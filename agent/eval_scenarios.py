"""Eval scenario fixtures - the single source of truth for the scenarios used
by agent/evals.py (Phoenix + RAGAS), tests/test_rag_quality.py (DeepEval),
and agent/checks.py's structural check.
"""

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
    {
        "input": {
            "question": "Before I dig into Initech's account, has there been any recent merger, "
            "acquisition, or ownership-change news involving Initech that could be relevant "
            "context? Check internal notes, and if we don't have anything on that, look it up."
        },
        "output": {
            "expected_finding": "Internal knowledge base has no coverage of external M&A/ownership "
            "news (only internal billing policy and account notes), so the agent should recognize "
            "low internal confidence and fall back to a real web search, then clearly cite the "
            "result as an external/web source rather than presenting it as internal policy or a "
            "solid internal match."
        },
        "metadata": {
            "scenario": "low_confidence_web_fallback",
            "approve": False,
            "expect_tools": ["search_knowledge_base", "search_web"],
            "expect_write": False,
            "expected_amount": None,
            "mock_web_search": True,
        },
    },
]
