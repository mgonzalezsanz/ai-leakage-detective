"""Shared agent-invocation for the eval harnesses (Phoenix/RAGAS in
agent/evals.py, DeepEval in tests/test_rag_quality.py): running one scenario
end-to-end through the real agent and extracting a structured result. Kept
separate from agent/evals.py so importing it doesn't pull in Phoenix's
dataset/experiment/judge machinery.
"""

import ast
import uuid
from contextlib import nullcontext
from unittest.mock import patch

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from phoenix.otel import using_session

from agent.graph import build_graph
from agent.tools import ACTIONS_FILE, _load_json

_MOCK_WEB_SEARCH_RESULT = [
    {
        "source": "https://example-news.test/initech-globex-holdings-investment",
        "title": "Globex Holdings completes minority-stake investment in Initech",
        "text": "Globex Holdings announced it has closed a 12% minority-stake investment in "
        "Initech in August 2026, part of a broader vendor-consolidation strategy.",
        "score": 0.83,
        "source_type": "web",
    }
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


def _retrieval_context(transcript: list[dict]) -> str:
    """Join every search_knowledge_base and search_web tool result in the
    transcript into one string, for evaluators that grade retrieval
    holistically regardless of whether it was internal or a web fallback."""
    return "\n\n".join(
        entry["content"] for entry in transcript
        if entry.get("name") in ("search_knowledge_base", "search_web")
    )


def retrieved_contexts(transcript: list[dict]) -> list[str]:
    """Like _retrieval_context, but as a list of discrete chunk texts rather
    than one joined blob - RAGAS's/DeepEval's context precision and recall
    metrics score per-chunk attribution, so collapsing everything into one
    string would make those two metrics meaningless.

    search_knowledge_base/search_web return list[dict]; LangChain stringifies
    that (via repr) into the ToolMessage content, so this parses it back with
    ast.literal_eval (safe here - the tool results only ever contain plain
    dicts/strings/floats/bools, no arbitrary objects) and pulls each chunk's
    "text" field. Falls back to the raw string as a single-element list if a
    result doesn't parse as expected.
    """
    contexts = []
    for entry in transcript:
        if entry.get("name") not in ("search_knowledge_base", "search_web"):
            continue
        content = entry["content"]
        try:
            parsed = ast.literal_eval(content)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, list) and all(isinstance(d, dict) for d in parsed):
            contexts.extend(d.get("text", "") for d in parsed if d.get("text"))
        elif content:
            contexts.append(content)
    return contexts


def run_scenario(input: dict, metadata: dict) -> dict:
    """Run one scenario end-to-end, including the approve/reject resume if the
    agent's apply() call pauses on interrupt()."""
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    graph = build_graph(InMemorySaver())
    actions_before = _load_json(ACTIONS_FILE)

    # Fakes the one genuinely networked tool for scenarios that need it, so the suite stays
    # offline/deterministic. Patches the alias bound into agent.tools,
    # not agent.web_search.search itself - tools.py already holds its own reference by the
    # time this runs.
    web_search_patch = (
        patch("agent.tools._search_web", return_value=_MOCK_WEB_SEARCH_RESULT)
        if metadata.get("mock_web_search")
        else nullcontext()
    )

    with using_session(thread_id), web_search_patch:
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
