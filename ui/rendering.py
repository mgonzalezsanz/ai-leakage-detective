"""Pure, FastAPI/Jinja-free rendering logic: turning raw LangGraph message
state into plain dicts templates can loop over, and classifying retrieval
results into badges."""

import json


def normalize_messages(messages: list) -> list[dict]:
    """Turn a LangGraph message list into plain dicts: {"role": "user"/"assistant",
    "text": ..., "tool_calls": [{"name", "args", "raw", "badges"}, ...]}."""
    # LangGraph's ToolNode serializes most tool outputs to a JSON string, but an
    # *empty* list is a vacuous-truth edge case in its own serializer (all(...) over
    # an empty sequence is True) and comes through as the raw Python list [] instead -
    # normalize everything to a string here so "no result yet" (None) can never be
    # confused with "the tool returned an empty result" (falsy but real, e.g. "[]").
    tool_results = {
        m.tool_call_id: m.content if isinstance(m.content, str) else json.dumps(m.content)
        for m in messages
        if m.type == "tool"
    }
    out = []
    for msg in messages:
        if msg.type == "human":
            out.append({"role": "user", "text": msg.content})
        elif msg.type == "ai":
            text = msg.content if isinstance(msg.content, str) else "".join(
                b.get("text", "") for b in msg.content if isinstance(b, dict)
            )
            tool_calls = []
            for tc in msg.tool_calls:
                raw = tool_results.get(tc["id"])
                tool_calls.append(
                    {
                        "name": tc["name"],
                        "args": tc["args"],
                        "raw": raw,
                        "badges": _extract_badges(raw) if tc["name"] in ("search_knowledge_base", "search_web") else [],
                    }
                )
            out.append({"role": "assistant", "text": text, "tool_calls": tool_calls})
    return out


def classify_badge(item: dict) -> dict | None:
    """Classify one retrieved item into a badge descriptor, or None if it
    isn't a real result (e.g. an {"error": ...} item with no score). Same
    three-way branching as the old _render_retrieval_badges: web fallback,
    low-confidence internal match, or a solid internal match."""
    if not isinstance(item, dict) or "score" not in item:
        return None
    if item.get("source_type") == "web":
        return {
            "kind": "web",
            "icon": "\U0001f310",
            "label": "Web fallback",
            "score": item["score"],
            "detail": item.get("source", ""),
        }
    if item.get("confidence") == "low":
        return {
            "kind": "warn",
            "icon": "⚠️",
            "label": "Low-confidence internal match",
            "score": item["score"],
            "detail": item.get("title", ""),
        }
    return {
        "kind": "trust",
        "icon": "\U0001f4c4",
        "label": "Internal doc",
        "score": item["score"],
        "detail": item.get("title", ""),
    }


_AUDIT_ACTION_LABELS = {
    "make_good_invoice": "Make-Good Invoice",
    "credit_memo": "Credit Memo",
    "plan_amendment": "Plan Amendment",
}


def format_audit_entry(entry: dict) -> dict:
    """Turn one raw data/sandbox/audit_log.json record into a compact,
    human-readable summary for the audit drawer - the full record is still
    available via "raw", for anyone who wants it."""
    payload = entry.get("payload", {}) or {}
    action_type = entry.get("type", "")
    amount = payload.get("amount")
    reason = payload.get("reason")
    if reason is None and "change_set" in payload:
        reason = ", ".join(f"{k} → {v}" for k, v in payload["change_set"].items())

    timestamp = entry.get("timestamp", "")  # "2026-08-25T17:14:44.834442+00:00"

    return {
        "event": entry.get("event", ""),
        "label": _AUDIT_ACTION_LABELS.get(action_type, action_type.replace("_", " ").title()),
        "ref": payload.get("plan_id") or payload.get("invoice_id") or "",
        "amount": f"${amount:,.0f}" if isinstance(amount, (int, float)) else None,
        "reason": reason,
        "when": timestamp[:16].replace("T", " ") if timestamp else "",
        "action_id": (entry.get("action_id") or "")[:8],
        "raw": json.dumps(entry, indent=2),
    }


def _extract_badges(raw: str | None) -> list[dict]:
    if not raw:
        return []
    try:
        items = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(items, list):
        return []
    badges = [classify_badge(item) for item in items]
    return [b for b in badges if b is not None]
