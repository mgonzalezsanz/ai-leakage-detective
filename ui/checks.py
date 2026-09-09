"""Assert-based self-checks for the pure logic in ui/rendering.py.
Run with: python -m ui.checks
"""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ui.rendering import classify_badge, format_audit_entry, normalize_messages


def check_classify_badge():
    web_item = {"score": 0.72, "source_type": "web", "source": "https://example.test"}
    badge = classify_badge(web_item)
    assert badge["kind"] == "web", badge

    low_item = {"score": 0.21, "confidence": "low", "title": "Some Policy Doc"}
    badge = classify_badge(low_item)
    assert badge["kind"] == "warn", badge

    high_item = {"score": 0.82, "confidence": "high", "title": "Escalation Thresholds"}
    badge = classify_badge(high_item)
    assert badge["kind"] == "trust", badge

    error_item = {"error": "TAVILY_API_KEY not set"}
    assert classify_badge(error_item) is None, "an error item (no score) shouldn't get a badge"

    assert classify_badge("not a dict") is None
    assert classify_badge({"score": 0.5}) is not None, "a bare score with no confidence/source_type still gets a trust badge"


def check_normalize_messages_basic_roles():
    messages = [
        HumanMessage(content="Investigate plan C-1001"),
        AIMessage(content="Sure, let me check.", tool_calls=[]),
    ]
    out = normalize_messages(messages)
    assert out == [
        {"role": "user", "text": "Investigate plan C-1001"},
        {"role": "assistant", "text": "Sure, let me check.", "tool_calls": []},
    ], out


def check_normalize_messages_content_blocks():
    """Anthropic-style content can be a list of {"type": "text", "text": ...}
    blocks instead of a plain string - normalize_messages must extract just
    the text, same as the old Streamlit render_messages()."""
    messages = [AIMessage(content=[{"type": "text", "text": "Here's what I found."}], tool_calls=[])]
    out = normalize_messages(messages)
    assert out[0]["text"] == "Here's what I found.", out


def check_normalize_messages_tool_calls_and_badges():
    kb_result = [{"source": "policy.md", "title": "Policy", "score": 0.55, "confidence": "high"}]
    ai_msg = AIMessage(
        content="Checking the knowledge base.",
        tool_calls=[{"id": "call_1", "name": "search_knowledge_base", "args": {"query": "x"}}],
    )
    tool_msg = ToolMessage(content=json.dumps(kb_result), tool_call_id="call_1", name="search_knowledge_base")

    out = normalize_messages([ai_msg, tool_msg])
    tc = out[0]["tool_calls"][0]
    assert tc["name"] == "search_knowledge_base", tc
    assert tc["raw"] == json.dumps(kb_result), tc
    assert len(tc["badges"]) == 1 and tc["badges"][0]["kind"] == "trust", tc

    # A tool call to something other than search_knowledge_base/search_web
    # never gets badges, even if its result happens to look list-of-dicts-ish.
    other_ai = AIMessage(
        content="",
        tool_calls=[{"id": "call_2", "name": "load_plan", "args": {"plan_id": "C-1001"}}],
    )
    other_tool = ToolMessage(content=json.dumps(kb_result), tool_call_id="call_2", name="load_plan")
    out2 = normalize_messages([other_ai, other_tool])
    assert out2[0]["tool_calls"][0]["badges"] == [], out2


def check_format_audit_entry():
    applied = {
        "action_id": "f47bfaf3-d81e-42e7-bca3-2b39937d8835",
        "type": "make_good_invoice",
        "payload": {"plan_id": "C-1010", "amount": 20000.0, "reason": "Underbilling recovery."},
        "timestamp": "2026-08-25T17:14:44.834442+00:00",
        "event": "applied",
    }
    out = format_audit_entry(applied)
    assert out["label"] == "Make-Good Invoice", out
    assert out["amount"] == "$20,000", out
    assert out["ref"] == "C-1010", out
    assert out["when"] == "2026-08-25 17:14", out
    assert out["action_id"] == "f47bfaf3", out

    # credit_memo uses invoice_id instead of plan_id
    memo = {
        "action_id": "abc123",
        "type": "credit_memo",
        "payload": {"invoice_id": "I-9123", "amount": 2000, "reason": "FX overbilling."},
        "timestamp": "2026-09-01T10:00:00+00:00",
        "event": "applied",
    }
    out = format_audit_entry(memo)
    assert out["ref"] == "I-9123", out
    assert out["amount"] == "$2,000", out

    # plan_amendment has no "amount"/"reason" - falls back to change_set
    amendment = {
        "action_id": "xyz789",
        "type": "plan_amendment",
        "payload": {"plan_id": "C-1007", "change_set": {"total_value": 100000}},
        "timestamp": "2026-09-01T10:00:00+00:00",
        "event": "rolled_back",
    }
    out = format_audit_entry(amendment)
    assert out["amount"] is None, out
    assert "total_value" in out["reason"], out
    assert out["event"] == "rolled_back", out


def check_normalize_messages_empty_list_tool_result():
    """Regression check: LangGraph's ToolNode serializes an *empty* list tool
    result as the raw Python list [] (a vacuous-truth edge case in its own
    serializer), not the JSON string "[]" it uses for every other result.
    normalize_messages must not let that render as "(pending)"."""
    ai_msg = AIMessage(
        content="Checking for existing credit memos.",
        tool_calls=[{"id": "call_1", "name": "query_credit_memos", "args": {"plan_id": "C-1010"}}],
    )
    tool_msg = ToolMessage(content=[], tool_call_id="call_1", name="query_credit_memos")

    out = normalize_messages([ai_msg, tool_msg])
    tc = out[0]["tool_calls"][0]
    assert tc["raw"] == "[]", f"expected the JSON string '[]', got {tc['raw']!r}"


if __name__ == "__main__":
    check_classify_badge()
    check_normalize_messages_basic_roles()
    check_normalize_messages_content_blocks()
    check_normalize_messages_tool_calls_and_badges()
    check_normalize_messages_empty_list_tool_result()
    check_format_audit_entry()
    print("OK")
