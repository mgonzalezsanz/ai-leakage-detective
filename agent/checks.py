"""Assert-based self-checks for the real logic in agent/tools.py.
Run with: python -m agent.checks
"""

import os
from unittest.mock import patch

from agent.reranker import rerank
from agent.tools import (
    apply_impl,
    fx_convert,
    load_plan,
    propose_make_good_invoice,
    rollback,
)


def check_reranker_graceful_degradation():
    """No live Cohere call here - just the deterministic "don't crash, don't
    lie about being reranked" behavior: missing key or nothing to rerank both
    return None, not an empty/zeroed score list a caller could mistake for a
    real (if uniformly bad) result."""
    assert rerank("anything", []) is None, "no documents to rerank should return None, not []"

    with patch.dict("os.environ", {}, clear=False):
        os.environ.pop("COHERE_API_KEY", None)
        result = rerank("test query", ["doc one", "doc two"])
    assert result is None, result


def check_amendment_chain():
    current = load_plan.invoke({"plan_id": "C-1007"})
    assert current["plan_id"] == "C-1007-A1", current
    assert current["total_value"] == 100000, current

    pre_amendment = load_plan.invoke({"plan_id": "C-1007", "as_of_date": "2025-06-01"})
    assert pre_amendment["plan_id"] == "C-1007", pre_amendment
    assert pre_amendment["total_value"] == 90000, pre_amendment


def check_fx_convert():
    forward = fx_convert.invoke(
        {"amount": 25000, "from_ccy": "EUR", "to_ccy": "USD", "on_date": "2025-09-12"}
    )
    assert forward["converted_amount"] == 27000.0, forward

    inverse = fx_convert.invoke(
        {"amount": 27000, "from_ccy": "USD", "to_ccy": "EUR", "on_date": "2025-09-12"}
    )
    assert abs(inverse["converted_amount"] - 25000.0) < 0.01, inverse

    missing = fx_convert.invoke(
        {"amount": 100, "from_ccy": "GBP", "to_ccy": "JPY", "on_date": "2025-01-01"}
    )
    assert "error" in missing, missing


def check_apply_rollback_roundtrip():
    draft = propose_make_good_invoice.invoke(
        {"plan_id": "C-1001", "amount": 8000, "reason": "test check"}
    )
    applied = apply_impl(draft)
    assert applied["status"] == "applied", applied

    rolled_back = rollback.invoke({"action_id": applied["action_id"]})
    assert rolled_back["status"] == "rolled_back", rolled_back

    from agent.tools import _load_json, AUDIT_LOG_FILE

    audit_log = _load_json(AUDIT_LOG_FILE)
    assert any(
        e["action_id"] == applied["action_id"] and e["event"] == "rolled_back"
        for e in audit_log
    ), audit_log


if __name__ == "__main__":
    check_amendment_chain()
    check_fx_convert()
    check_apply_rollback_roundtrip()
    check_reranker_graceful_degradation()
    print("OK")
