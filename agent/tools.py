"""Tools for the Revenue Leakage Agent: read-only lookups over /data, draft
proposals (no writes), and sandbox apply/rollback."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import tool

DATA_DIR = Path(__file__).parent.parent / "data"
SANDBOX_DIR = DATA_DIR / "sandbox"
ACTIONS_FILE = SANDBOX_DIR / "actions.json"
AUDIT_LOG_FILE = SANDBOX_DIR / "audit_log.json"


def _load_json(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@tool
def load_plan(plan_id: str, as_of_date: str | None = None) -> dict:
    """Load a billing plan by ID, resolving any amendment chain.

    If as_of_date (YYYY-MM-DD) is given, returns the version of the plan in
    effect on that date. Otherwise returns the most current (latest amended)
    version. The result includes "amendment_chain" listing every plan_id in
    the chain from original to most current, for citation purposes.
    """
    plans = _load_json(DATA_DIR / "billing_plans.json")
    by_id = {p["plan_id"]: p for p in plans}
    if plan_id not in by_id:
        return {"error": f"plan_id {plan_id} not found"}

    chain = [by_id[plan_id]]
    current = by_id[plan_id]
    for _ in range(10):
        nxt = next((p for p in plans if p.get("amends") == current["plan_id"]), None)
        if nxt is None:
            break
        chain.append(nxt)
        current = nxt

    if as_of_date is not None:
        eligible = [p for p in chain if p["start_date"] <= as_of_date]
        resolved = eligible[-1] if eligible else chain[0]
    else:
        resolved = chain[-1]

    return {**resolved, "amendment_chain": [p["plan_id"] for p in chain]}


@tool
def query_invoices(
    plan_id: str | None = None,
    customer_name: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict]:
    """Filter invoices by plan_id, customer_name, and/or issue_date range
    (date_from/date_to are inclusive, YYYY-MM-DD). Omit a filter to not
    restrict on that field."""
    invoices = _load_json(DATA_DIR / "invoices.json")
    return [
        inv
        for inv in invoices
        if (plan_id is None or inv["plan_id"] == plan_id)
        and (customer_name is None or inv["customer_name"] == customer_name)
        and (date_from is None or inv["issue_date"] >= date_from)
        and (date_to is None or inv["issue_date"] <= date_to)
    ]


@tool
def query_credit_memos(plan_id: str | None = None, invoice_id: str | None = None) -> list[dict]:
    """Look up existing credit memos, optionally filtered by plan_id and/or
    invoice_id. Always check this before proposing a credit memo or
    explaining an overbilling issue - it may already be resolved."""
    memos = _load_json(DATA_DIR / "credit_memos.json")
    return [
        m
        for m in memos
        if (plan_id is None or m["plan_id"] == plan_id)
        and (invoice_id is None or m["invoice_id"] == invoice_id)
    ]


@tool
def fx_convert(amount: float, from_ccy: str, to_ccy: str, on_date: str) -> dict:
    """Convert amount from from_ccy to to_ccy using the rate in effect on
    on_date (YYYY-MM-DD). Only exact currency-pair matches (or their inverse)
    in exchange_rates.json are supported; returns an "error" key if no rate
    is available."""
    if from_ccy == to_ccy:
        return {"converted_amount": amount, "rate": 1.0, "rate_date": on_date}

    rates = _load_json(DATA_DIR / "exchange_rates.json")
    for r in rates:
        if r["from_currency"] == from_ccy and r["to_currency"] == to_ccy:
            return {
                "converted_amount": round(amount * r["rate"], 2),
                "rate": r["rate"],
                "rate_date": r["date"],
            }
        if r["from_currency"] == to_ccy and r["to_currency"] == from_ccy:
            inv_rate = 1 / r["rate"]
            return {
                "converted_amount": round(amount * inv_rate, 2),
                "rate": inv_rate,
                "rate_date": r["date"],
            }
    return {"error": f"no exchange rate available for {from_ccy}->{to_ccy}"}


@tool
def propose_make_good_invoice(plan_id: str, amount: float, reason: str) -> dict:
    """Draft a new invoice to recover missed/underbilled revenue. Does not
    write anything; call apply() with this draft to commit it after human
    approval."""
    return {
        "action_id": str(uuid.uuid4()),
        "type": "make_good_invoice",
        "payload": {"plan_id": plan_id, "amount": amount, "reason": reason},
        "status": "draft",
        "created_at": _now(),
    }


@tool
def propose_credit_memo(invoice_id: str, amount: float, reason: str) -> dict:
    """Draft a credit memo reducing what's owed on an invoice (overbilling /
    pricing error). Does not write anything; call apply() with this draft to
    commit it after human approval."""
    return {
        "action_id": str(uuid.uuid4()),
        "type": "credit_memo",
        "payload": {"invoice_id": invoice_id, "amount": amount, "reason": reason},
        "status": "draft",
        "created_at": _now(),
    }


@tool
def propose_plan_amendment(plan_id: str, change_set: dict) -> dict:
    """Draft a billing plan update (e.g. new total_value, cadence, or
    entitlements). Does not write anything; call apply() with this draft to
    commit it after human approval."""
    return {
        "action_id": str(uuid.uuid4()),
        "type": "plan_amendment",
        "payload": {"plan_id": plan_id, "change_set": change_set},
        "status": "draft",
        "created_at": _now(),
    }


def apply_impl(action_draft: dict) -> dict:
    """Write an approved action draft to the sandbox ledger + audit log."""
    actions = _load_json(ACTIONS_FILE)
    applied = {**action_draft, "status": "applied", "applied_at": _now()}
    actions.append(applied)
    _save_json(ACTIONS_FILE, actions)

    audit_log = _load_json(AUDIT_LOG_FILE)
    audit_log.append(
        {
            "action_id": applied["action_id"],
            "type": applied["type"],
            "payload": applied["payload"],
            "timestamp": _now(),
            "event": "applied",
        }
    )
    _save_json(AUDIT_LOG_FILE, audit_log)
    return applied


@tool
def rollback(action_id: str | None = None) -> dict:
    """Undo an applied action. If action_id is omitted, undoes the most
    recently applied action."""
    actions = _load_json(ACTIONS_FILE)
    applied_actions = [a for a in actions if a["status"] == "applied"]
    if action_id is not None:
        target = next((a for a in applied_actions if a["action_id"] == action_id), None)
    else:
        target = max(applied_actions, key=lambda a: a["applied_at"], default=None)

    if target is None:
        return {"error": "no matching applied action found to roll back"}

    target["status"] = "rolled_back"
    target["rolled_back_at"] = _now()
    _save_json(ACTIONS_FILE, actions)

    audit_log = _load_json(AUDIT_LOG_FILE)
    audit_log.append(
        {
            "action_id": target["action_id"],
            "type": target["type"],
            "payload": target["payload"],
            "timestamp": _now(),
            "event": "rolled_back",
        }
    )
    _save_json(AUDIT_LOG_FILE, audit_log)
    return target
