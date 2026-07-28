"""Revenue Leakage Agent graph: a ReAct agent whose apply() tool is gated by
a real LangGraph interrupt() so nothing writes to the sandbox without an
explicit human resume."""

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt

from agent.tools import (
    apply_impl,
    fx_convert,
    load_plan,
    propose_credit_memo,
    propose_make_good_invoice,
    propose_plan_amendment,
    query_credit_memos,
    query_invoices,
    rollback,
)

load_dotenv()


@tool
def apply(action_draft: dict) -> dict:
    """Write an approved action draft (from a propose_* tool) to the
    sandbox. This pauses for human approval before writing anything -
    only call it after the user has explicitly confirmed in this
    conversation that they want the draft applied."""
    decision = interrupt({"action": "apply", "draft": action_draft})
    if decision != "approve":
        return {"status": "rejected", "reason": decision}
    return apply_impl(action_draft)


TOOLS = [
    load_plan,
    query_invoices,
    query_credit_memos,
    fx_convert,
    propose_make_good_invoice,
    propose_credit_memo,
    propose_plan_amendment,
    apply,
    rollback,
]

SYSTEM_PROMPT = """You are the Revenue Leakage Agent, a financial detective \
that investigates discrepancies between billing plans and invoices.

Rules you must follow:
- Before asserting any discrepancy exists, call load_plan and/or \
query_invoices to fetch the real numbers - never state a figure you haven't \
actually fetched in this conversation.
- When you claim a discrepancy, cite the evidence: the plan's expected \
amount and currency, the invoice amount(s)/currency/date(s), and if a \
currency conversion was involved, the FX rate and date used. Show the \
arithmetic, e.g. "25000 EUR x 1.08 = 27000 USD vs 25000 USD expected -> \
$2000 overbilled".
- If a plan has an amendment_chain, mention which version is in effect and \
since when.
- Before proposing a credit memo, or whenever discussing a possible \
overbilling issue, call query_credit_memos for the relevant plan_id/invoice_id \
- if a credit memo already covers it, explain that it's already resolved \
(cite the memo_id, amount, and reason) rather than re-proposing a fix.
- propose_make_good_invoice, propose_credit_memo, and propose_plan_amendment \
only draft an action; they never write anything. Only apply() writes to the \
sandbox, and apply() will pause for human approval - never call apply() \
unless the user's most recent message is an explicit confirmation (e.g. \
"yes", "apply it", "go ahead") of a proposal you already made. If in doubt, \
ask "Would you like me to apply this?" and wait for their answer.
- After apply() runs, tell the user what was written (action id and type) \
and that it's in the sandbox now.
- rollback() undoes the most recently applied action, or a specific one by \
action_id.
"""

model = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)


def build_graph(checkpointer=None):
    """Compile the agent graph. `langgraph dev` / LangGraph Studio inject
    their own persistence and reject a custom checkpointer, so the
    module-level `graph` below passes none. Standalone callers (e.g. the
    Streamlit app) need a real checkpointer for interrupt() to work and
    should call this directly with one."""
    return create_react_agent(model, TOOLS, prompt=SYSTEM_PROMPT, checkpointer=checkpointer)


graph = build_graph()
