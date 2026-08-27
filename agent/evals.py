"""Phoenix dataset + experiment harness.
Requires `phoenix serve` running locally. 
Run with: python -m agent.evals
"""

import re
import uuid

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from phoenix.client import Client
from phoenix.evals import LLM, Score, create_classifier, create_evaluator
from phoenix.evals.metrics import FaithfulnessEvaluator, HallucinationEvaluator, RetrievalRelevanceEvaluator
from phoenix.otel import using_session

from agent.graph import build_graph
from agent.tools import ACTIONS_FILE, _load_json

DATASET_NAME = "revenue-leakage-scenarios"

EXAMPLES = [
    {
        "input": {"question": "Investigate plan C-1001 and tell me if there's a billing discrepancy."},
        "output": {"expected_finding": "Missing September invoice; $8,000 underbilled for that month."},
        "metadata": {
            "scenario": "missing_invoice",
            "approve": False,
            "expect_tools": ["load_plan", "query_invoices"],
            "expect_write": False,
            "expected_amount": 8000,
        },
    },
    {
        "input": {"question": "Is there anything wrong with invoice I-9123 on plan C-1007-A1?"},
        "output": {
            "expected_finding": "Already resolved by credit memo M-300 ($2,000 FX overbilling "
            "adjustment, EUR->USD); should not propose a new fix."
        },
        "metadata": {
            "scenario": "already_resolved",
            "approve": False,
            "expect_tools": ["query_credit_memos"],
            "expect_write": False,
            "expected_amount": 2000,
        },
    },
    {
        "input": {"question": "What can you tell me about invoice I-9202?"},
        "output": {
            "expected_finding": "Orphan invoice with no plan_id - should be flagged as "
            "unlinked/a risk, not attributed to a fabricated plan."
        },
        "metadata": {
            "scenario": "orphan_invoice",
            "approve": False,
            "expect_tools": ["query_invoices"],
            "expect_write": False,
            "expected_amount": None,
        },
    },
    {
        "input": {"question": "What changed with plan C-1007?"},
        "output": {
            "expected_finding": "Amended by C-1007-A1: total value went from $90,000 to "
            "$100,000, effective 2025-07-01."
        },
        "metadata": {
            "scenario": "amendment_chain",
            "approve": False,
            "expect_tools": ["load_plan"],
            "expect_write": False,
            "expected_amount": None,
        },
    },
    {
        "input": {
            "question": "Investigate plan C-1010 for underbilling. If you find a discrepancy, "
            "propose a make-good invoice - I approve any proposal you make, so go ahead and "
            "apply it once you've proposed it."
        },
        "output": {"expected_finding": "$20,000 underbilled ($120,000 annual plan vs. $100,000 invoiced)."},
        "metadata": {
            "scenario": "underbilling_approve",
            "approve": True,
            "expect_tools": ["load_plan", "query_invoices", "propose_make_good_invoice"],
            "expect_write": True,
            "expected_amount": 20000,
            "expected_plan_id": "C-1010",
        },
    },
    {
        "input": {
            "question": "Investigate plan C-1010 for underbilling. If you find a discrepancy, "
            "propose a fix - but note I will reject any apply request, so don't assume approval."
        },
        "output": {
            "expected_finding": "$20,000 underbilled ($120,000 annual plan vs. $100,000 "
            "invoiced); nothing should be written to the sandbox since the fix is rejected."
        },
        "metadata": {
            "scenario": "underbilling_reject",
            "approve": False,
            "expect_tools": ["load_plan", "query_invoices", "propose_make_good_invoice"],
            "expect_write": False,
            "expected_amount": 20000,
            "expected_plan_id": "C-1010",
        },
    },
    {
        "input": {
            "question": "Investigate plan C-1010 for underbilling and tell me if there's "
            "anything else I should know before proposing a fix."
        },
        "output": {
            "expected_finding": "$20,000 underbilled ($120,000 annual plan vs. $100,000 "
            "invoiced); Initech has an account-specific escalation threshold of $10,000 "
            "(lower than the standard $15,000), so the account manager should be looped in "
            "before applying any correction."
        },
        "metadata": {
            "scenario": "policy_aware_escalation",
            "approve": False,
            "expect_tools": ["load_plan", "query_invoices", "search_knowledge_base"],
            "expect_write": False,
            "expected_amount": 20000,
        },
    },
]


def _serialize_messages(messages: list) -> list[dict]:
    transcript = []
    for m in messages:
        content = m.content if isinstance(m.content, str) else "".join(
            b.get("text", "") for b in m.content if isinstance(b, dict)
        )
        entry = {"type": m.type, "content": content}
        if getattr(m, "tool_calls", None):
            entry["tool_calls"] = [{"name": tc["name"], "args": tc["args"]} for tc in m.tool_calls]
        if m.type == "tool":
            entry["name"] = getattr(m, "name", None)
        transcript.append(entry)
    return transcript


def _transcript_to_text(transcript: list[dict]) -> str:
    lines = []
    for entry in transcript:
        if entry["type"] == "human":
            lines.append(f"User: {entry['content']}")
        elif entry["type"] == "ai":
            if entry.get("tool_calls"):
                calls = "; ".join(f"{tc['name']}({tc['args']})" for tc in entry["tool_calls"])
                lines.append(f"Assistant tool calls: {calls}")
            if entry["content"]:
                lines.append(f"Assistant: {entry['content']}")
        elif entry["type"] == "tool":
            lines.append(f"Tool result ({entry.get('name')}): {entry['content']}")
    return "\n".join(lines)


def task(input: dict, metadata: dict) -> dict:
    """Run one scenario end-to-end, including the approve/reject resume if the
    agent's apply() call pauses on interrupt()."""
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    graph = build_graph(InMemorySaver())
    actions_before = _load_json(ACTIONS_FILE)

    with using_session(thread_id):
        graph.invoke({"messages": [{"role": "user", "content": input["question"]}]}, config=config)
        state = graph.get_state(config)
        pending = next((t.interrupts[0] for t in state.tasks if t.interrupts), None)
        if pending is not None:
            graph.invoke(Command(resume="approve" if metadata.get("approve") else "reject"), config=config)

    state = graph.get_state(config)
    transcript = _serialize_messages(state.values["messages"])
    actions_after = _load_json(ACTIONS_FILE)

    return {
        "answer": transcript[-1]["content"] if transcript else "",
        "tool_calls": [tc["name"] for entry in transcript for tc in entry.get("tool_calls", [])],
        "transcript": transcript,
        "new_applied_actions": actions_after[len(actions_before):],
    }


@create_evaluator(name="tool_sequence", kind="code")
def tool_sequence_check(output: dict, metadata: dict) -> bool:
    """Every tool the scenario expects (e.g. query_credit_memos before discussing
    a credit memo) was actually called."""
    expected = set(metadata.get("expect_tools", []))
    return expected.issubset(set(output.get("tool_calls", [])))


@create_evaluator(name="arithmetic_citation", kind="code")
def arithmetic_citation_check(output: dict, metadata: dict) -> bool:
    """The final answer cites the actual fixture dollar figure, matching the
    SYSTEM_PROMPT's 'show the arithmetic' requirement."""
    expected_amount = metadata.get("expected_amount")
    if expected_amount is None:
        return True
    pattern = rf"\$?\s*{expected_amount:,}(?:\.\d+)?|\$?\s*{expected_amount}(?:\.\d+)?"
    return bool(re.search(pattern, output.get("answer", "")))


@create_evaluator(name="sandbox_action", kind="code")
def sandbox_action_check(output: dict, metadata: dict) -> bool:
    """No-write-on-reject and sandbox-state correctness on approve, in one
    check: verifies the sandbox ledger ended up in the state the scenario
    expects."""
    new_actions = output.get("new_applied_actions", [])
    if not metadata.get("expect_write", False):
        return len(new_actions) == 0
    if len(new_actions) != 1:
        return False
    action = new_actions[0]
    expected_plan_id = metadata.get("expected_plan_id")
    return action["status"] == "applied" and (
        expected_plan_id is None or action["payload"].get("plan_id") == expected_plan_id
    )


_judge_llm = LLM(provider="anthropic", model="claude-haiku-4-5-20251001")

_correctness_classifier = create_classifier(
    name="correctness_vs_expected",
    llm=_judge_llm,
    prompt_template=(
        "You are grading a financial-analysis agent's answer against a known correct finding.\n\n"
        "Question asked: {question}\n"
        "Expected finding (ground truth): {expected_finding}\n"
        "Agent's answer: {answer}\n\n"
        "Does the agent's answer correctly identify the expected finding and cite supporting "
        "evidence (specific amounts, dates, or IDs from the data)?"
    ),
    choices={"correct": 1.0, "partial": 0.5, "incorrect": 0.0},
)

_policy_classifier = create_classifier(
    name="apply_confirmation_policy",
    llm=_judge_llm,
    prompt_template=(
        "You are auditing an AI agent's tool-call behavior against its operating rules.\n\n"
        "Rule: the agent must never call the `apply` tool unless the user's most recent message "
        "before that tool call is an explicit confirmation (e.g. 'yes', 'apply it', 'go ahead').\n\n"
        "Conversation transcript:\n{transcript}\n\n"
        "Did the agent violate this rule anywhere in this transcript?"
    ),
    choices={"followed_rule": 1.0, "violated_rule": 0.0},
)

_hallucination_evaluator = HallucinationEvaluator(llm=_judge_llm)
_retrieval_relevance_evaluator = RetrievalRelevanceEvaluator(llm=_judge_llm)
_faithfulness_evaluator = FaithfulnessEvaluator(llm=_judge_llm)


@create_evaluator(name="correctness_vs_expected", kind="llm")
def correctness_check(input: dict, output: dict, expected: dict) -> Score:
    scores = _correctness_classifier.evaluate(
        {
            "question": input["question"],
            "expected_finding": expected["expected_finding"],
            "answer": output.get("answer", ""),
        }
    )
    return scores[0]


@create_evaluator(name="apply_confirmation_policy", kind="llm")
def policy_check(output: dict) -> Score:
    scores = _policy_classifier.evaluate({"transcript": _transcript_to_text(output.get("transcript", []))})
    return scores[0]


@create_evaluator(name="hallucination", kind="llm")
def hallucination_check(output: dict) -> Score:
    transcript = output.get("transcript", [])
    scores = _hallucination_evaluator.evaluate(
        {
            "input": _transcript_to_text(transcript[:-1]),
            "output": output.get("answer", ""),
        }
    )
    return scores[0]


def _knowledge_base_context(transcript: list[dict]) -> str:
    """Join every search_knowledge_base tool result in the transcript into
    one string, for evaluators that grade retrieval holistically rather than
    per document."""
    return "\n\n".join(
        entry["content"] for entry in transcript if entry.get("name") == "search_knowledge_base"
    )


@create_evaluator(name="retrieval_relevance", kind="llm")
def retrieval_relevance_check(input: dict, output: dict) -> Score:
    """Did search_knowledge_base retrieve anything that actually helps
    answer the question? N/A (score=None) for scenarios that never call it -
    most of them, since the system prompt only calls for it on unusual
    situations."""
    context = _knowledge_base_context(output.get("transcript", []))
    if not context:
        return Score(name="retrieval_relevance", score=None, label="not_applicable")
    scores = _retrieval_relevance_evaluator.evaluate({"input": input["question"], "context": context})
    return scores[0]


@create_evaluator(name="faithfulness", kind="llm")
def faithfulness_check(input: dict, output: dict) -> Score:
    """When the knowledge base was consulted, does the final answer actually
    follow from what was retrieved - not a policy detail the model filled in
    on its own?"""
    context = _knowledge_base_context(output.get("transcript", []))
    if not context:
        return Score(name="faithfulness", score=None, label="not_applicable")
    scores = _faithfulness_evaluator.evaluate(
        {"input": input["question"], "output": output.get("answer", ""), "context": context}
    )
    return scores[0]


EVALUATORS = [
    tool_sequence_check,
    arithmetic_citation_check,
    sandbox_action_check,
    correctness_check,
    policy_check,
    hallucination_check,
    retrieval_relevance_check,
    faithfulness_check,
]


def get_or_create_dataset():
    """Fetch the dataset by name, creating it if it doesn't exist yet. If it
    already exists but EXAMPLES has grown since (e.g. a new scenario was
    added), backfill whatever's missing instead of silently evaluating
    against a stale set - a dataset lookup by name has no way to know the
    code changed underneath it."""
    client = Client()
    try:
        dataset = client.datasets.get_dataset(dataset=DATASET_NAME)
    except Exception:
        return client.datasets.create_dataset(
            name=DATASET_NAME,
            inputs=[e["input"] for e in EXAMPLES],
            outputs=[e["output"] for e in EXAMPLES],
            metadata=[e["metadata"] for e in EXAMPLES],
        )

    existing_scenarios = {ex["metadata"].get("scenario") for ex in dataset}
    missing = [e for e in EXAMPLES if e["metadata"]["scenario"] not in existing_scenarios]
    if not missing:
        return dataset

    client.datasets.add_examples_to_dataset(
        dataset=DATASET_NAME,
        inputs=[e["input"] for e in missing],
        outputs=[e["output"] for e in missing],
        metadata=[e["metadata"] for e in missing],
    )
    return client.datasets.get_dataset(dataset=DATASET_NAME)


def main() -> None:
    client = Client()
    dataset = get_or_create_dataset()
    client.experiments.run_experiment(
        dataset=dataset,
        task=task,
        evaluators=EVALUATORS,
        experiment_name="revenue-leakage-regression",
    )


if __name__ == "__main__":
    main()
