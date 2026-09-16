"""DeepEval CI gate for RAG quality.
This suite asserts explicit pass/fail thresholds meant to fail a build, scoped to just
the scenarios that actually exercise search_knowledge_base/search_web.

Run with: pytest tests/test_rag_quality.py
      or: deepeval test run tests/test_rag_quality.py
"""

import pytest
from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.models import AnthropicModel
from deepeval.test_case import LLMTestCase

from agent.eval_runner import retrieved_contexts, run_scenario
from agent.eval_scenarios import EXAMPLES

_judge = AnthropicModel(model="claude-haiku-4-5-20251001", temperature=0)

# Faithfulness/relevancy sit above DeepEval's 0.5 default: for a financial agent, an
# ungrounded claim or an off-topic answer is the costly failure mode. Contextual
# precision/recall stay at the 0.5 default - the knowledge base is a handful of
# markdown files searched with k=3, so perfect ranking/coverage isn't a realistic
# bar; the point is catching a regression (e.g. a reranker/embedding change), not
# enforcing a research-grade ceiling.
FAITHFULNESS_THRESHOLD = 0.7
ANSWER_RELEVANCY_THRESHOLD = 0.7
CONTEXT_PRECISION_THRESHOLD = 0.5
CONTEXT_RECALL_THRESHOLD = 0.5


def _load_rag_cases() -> dict[str, LLMTestCase]:
    """Run every scenario once, keep only the ones that actually
    called search_knowledge_base/search_web - this is a RAG-quality gate."""
    cases = {}
    for example in EXAMPLES:
        output = run_scenario(example["input"], example["metadata"])
        contexts = retrieved_contexts(output.get("transcript", []))
        if contexts:
            cases[example["metadata"]["scenario"]] = LLMTestCase(
                input=example["input"]["question"],
                actual_output=output.get("answer", ""),
                expected_output=example["output"]["expected_finding"],
                retrieval_context=contexts,
            )
    return cases


_RAG_CASES = _load_rag_cases()


def test_rag_scenarios_exist():
    """Guards against a silent false-green build if a prompt/agent change
    stops every scenario from triggering retrieval at all."""
    assert _RAG_CASES, "no scenario exercised search_knowledge_base/search_web"


@pytest.mark.parametrize("scenario", sorted(_RAG_CASES))
def test_faithfulness(scenario):
    assert_test(_RAG_CASES[scenario], [FaithfulnessMetric(threshold=FAITHFULNESS_THRESHOLD, model=_judge)])


@pytest.mark.parametrize("scenario", sorted(_RAG_CASES))
def test_answer_relevancy(scenario):
    assert_test(_RAG_CASES[scenario], [AnswerRelevancyMetric(threshold=ANSWER_RELEVANCY_THRESHOLD, model=_judge)])


@pytest.mark.parametrize("scenario", sorted(_RAG_CASES))
def test_contextual_precision(scenario):
    assert_test(
        _RAG_CASES[scenario], [ContextualPrecisionMetric(threshold=CONTEXT_PRECISION_THRESHOLD, model=_judge)]
    )


@pytest.mark.parametrize("scenario", sorted(_RAG_CASES))
def test_contextual_recall(scenario):
    assert_test(_RAG_CASES[scenario], [ContextualRecallMetric(threshold=CONTEXT_RECALL_THRESHOLD, model=_judge)])
