# Revenue Leakage Agent

A stateful, conversational AI agent that investigates revenue leakage in billing data — missing invoices, FX-driven overbilling, stale plan amendments — and proposes (and, only with explicit human approval, applies) fixes to a sandbox ledger.

Built with **LangGraph** (a ReAct agent with an `interrupt()`-gated approval step) and instrumented end-to-end with **Arize Phoenix** for tracing and offline evaluation.

📖 I wrote up how the observability/evaluation layer works in more detail here: [Observability and Evaluation for AI Agents, with Arize Phoenix](https://killosmind.com/2026/08/27/ai-agent-observability-evaluation-arize-phoenix/)

---

## What it does

- **Investigates** discrepancies between billing plans and invoices (missing invoices, FX mismatches, orphaned records)
- **Proposes** corrective actions — a make-good invoice, a credit memo, or a plan amendment
- **Applies** those actions to a writable sandbox only after explicit human confirmation, via a LangGraph `interrupt()`
- **Explains** every finding by citing the actual numbers, dates, and IDs it pulled from the data in that conversation — never a figure it didn't fetch
- **Maintains context** across a conversation, so follow-up questions ("what about the other months?") work naturally

The agent is conversational and stateful: it can be asked open-ended questions, remembers prior turns on the same thread, and adapts based on how the conversation goes.

## Key concepts

| Term | Meaning | Example |
|------|----------|--------------|
| **Credit Memo** | A negative invoice that reduces what the customer owes — used when overbilled or a pricing error occurred. | Overbilled €25,000 → $27,000 → issue a $2,000 USD credit memo |
| **Plan Amendment** | An update to the contract/billing plan (total, cadence, entitlements) when the agreement itself changes. | Upgrade plan from $90,000 → $100,000, or add "Premium Support" |
| **Make-Good Invoice** | A new invoice to recover missed or underbilled revenue. | Missing September billing → invoice $8,000 USD |

## Example conversation

```
User: Can you check if there are any revenue leakage issues with plan P-12345?

Agent: I'll investigate plan P-12345 for you. Let me load the plan details and check the invoices...
[Agent uses load_plan("P-12345") and query_invoices(...)]
I found a discrepancy! Plan P-12345 should bill $10,000 monthly, but I only see invoices for September and November. October is missing. This represents $10,000 in missed revenue.

User: What currency is that plan in?

Agent: Plan P-12345 is in USD. The expected monthly amount is $10,000 USD.

User: Can you create a make-good invoice for the missing October billing?

Agent: I'll propose a make-good invoice for $10,000 USD for the missing October 2024 billing period.
[Agent uses propose_make_good_invoice("P-12345", 10000, "Missing October 2024 billing")]
Proposal created. Would you like me to apply this to the sandbox?

User: Yes, apply it.

Agent: [Agent uses apply(...)] Make-good invoice has been applied to the sandbox. The invoice ID is INV-MG-001.

User: What about the other months? Are they correct?

Agent: Let me check the other invoices for this plan... September shows $10,000 USD which matches the plan. November also shows $10,000 USD. Both are correct. The only issue was the missing October invoice, which we've now addressed.
```

Notice how the agent maintains context about plan P-12345 throughout, answers follow-up questions that reference earlier turns, understands references to previous actions, and never writes to the sandbox until the user explicitly says to.

## Architecture

- [`agent/tools.py`](agent/tools.py) — read-only lookups (`load_plan`, `query_invoices`, `query_credit_memos`, `fx_convert`) and side-effect-free `propose_*` drafting tools; `apply_impl`/`rollback` are the only functions that touch the sandbox ledger.
- [`agent/graph.py`](agent/graph.py) — compiles a LangGraph `create_react_agent` over those tools, plus a wrapped `apply()` tool that calls `interrupt()` before writing anything, so the graph pauses until a human resumes with `"approve"` or `"reject"`.
- [`agent/knowledge_base.py`](agent/knowledge_base.py) — a small local RAG layer: sentence-transformer embeddings + a FAISS index over `data/knowledge_base`, searched via the `search_knowledge_base` tool for policy/account-specific context an investigation might need.
- [`agent/observability.py`](agent/observability.py) — wires OpenTelemetry tracing to a local Arize Phoenix collector and auto-instruments every LLM call and tool invocation.
- [`agent/evals.py`](agent/evals.py) — a Phoenix dataset + experiment harness: seven scenarios run end-to-end through the real agent, graded by eight evaluators (three deterministic, five LLM-judged, including RAG-specific retrieval-relevance and faithfulness checks).
- [`streamlit_app.py`](streamlit_app.py) — a minimal chat UI on top of the same compiled graph, with an approve/reject control for pending actions and a live audit-log viewer.

## Data (`/data`)

| File | Description |
|------|--------------|
| `billing_plans.json` | Expected billing plans (contracts), including amendment chains |
| `invoices.json` | Issued invoices |
| `credit_memos.json` | Existing credit memos |
| `exchange_rates.json` | FX rates |
| `sandbox/*.json` | Writable ledger + audit log for applied actions (gitignored) |
| `knowledge_base/*.md`, `knowledge_base/internal_notes/*.md` | Internal policy docs and account notes, retrieved via RAG (`search_knowledge_base`) |

## Tools available to the agent

| Tool | Purpose |
|------|----------|
| `load_plan(plan_id, as_of_date=None)` | Read a plan, resolving its amendment chain |
| `query_invoices(...)` | Filter invoices by plan, customer, date range |
| `query_credit_memos(...)` | Filter existing credit memos |
| `fx_convert(amount, from_ccy, to_ccy, on_date)` | Currency conversion using dated FX rates |
| `search_knowledge_base(query, k=3)` | Semantic search over internal policy docs and account notes (RAG, local embeddings) |
| `propose_make_good_invoice(...)` | Draft a new invoice (no write) |
| `propose_credit_memo(...)` | Draft a credit memo (no write) |
| `propose_plan_amendment(...)` | Draft a plan update (no write) |
| `apply(action_draft)` | Write a drafted action to the sandbox — pauses for human approval |
| `rollback(action_id=None)` | Undo the most recent (or a specific) applied action |

## Observability & evaluation

Every LLM call and tool invocation is traced via OpenTelemetry into [Arize Phoenix](https://github.com/Arize-ai/phoenix), so a full conversation shows up as one session with every reasoning step, tool call, and result inspectable in the Phoenix UI.

On top of that, [`agent/evals.py`](agent/evals.py) defines a small regression suite: seven scenarios (missing invoice, already-resolved via credit memo, orphan invoice, amendment chain, approve-and-apply, reject-and-don't-apply, and a policy-aware escalation case) run end-to-end through the real agent — including the human-approval interrupt — and are graded by:

- **Deterministic checks** — the right tools were called, the right dollar figure was cited, the sandbox ended up in the expected state (written on approval, untouched on rejection)
- **LLM-judged checks** — the answer matches the expected finding, the agent never called `apply()` without explicit confirmation, it didn't hallucinate facts not in the data, and — when `search_knowledge_base` was consulted — the retrieved policy/notes were actually relevant and the answer stayed faithful to them rather than inventing policy details

See the [blog post](https://killosmind.com/2026/08/27/ai-agent-observability-evaluation-arize-phoenix/) for a full walkthrough of how this is wired up and why.

## Getting started

```bash
git clone https://github.com/mmmgggsss/ai-leakage-detective.git
cd ai-leakage-detective
python -m venv .venv && source .venv/bin/activate   # macOS/Linux
python -m venv .venv && .venv\Scripts\activate       # Windows
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY
```

The first call to `search_knowledge_base` downloads its embedding model (`all-MiniLM-L6-v2`, ~90MB, one-time, cached locally afterward) — the first investigation in a fresh clone will be slower than the rest.

Start Phoenix in its own terminal (tracing/evals no-op silently if it isn't running):

```bash
phoenix serve   # UI at http://localhost:6006
```

Then run the chat UI:

```bash
streamlit run streamlit_app.py
```

...or open the graph in LangGraph Studio instead:

```bash
langgraph dev
```

Run the plain assert-based unit checks:

```bash
python -m agent.checks
```

Run the full Phoenix eval suite (requires Phoenix running):

```bash
python -m agent.evals
```

## Project structure

```
agent/
  tools.py             # read-only lookups, propose_* drafting tools, apply/rollback
  graph.py             # LangGraph ReAct agent + interrupt()-gated apply()
  knowledge_base.py    # local RAG: sentence-transformer embeddings + FAISS
  observability.py     # Phoenix/OTel tracing setup
  evals.py             # Phoenix dataset + experiment harness
  checks.py            # plain assert-based unit checks
data/                   # fixture data + writable sandbox ledger
streamlit_app.py        # chat UI
langgraph.json          # LangGraph Studio config
```
