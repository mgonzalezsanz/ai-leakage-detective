"""FastAPI + htmx chat UI, on top of the same compiled LangGraph agent used by
`langgraph dev`. Run with: python -m ui.app

Architecture: one background thread per conversation turn runs graph.stream(),
writing live per-node progress into a shared in-memory dict guarded by a lock.
The browser polls via htmx (hx-get ... every 1s), swapping only the volatile
tail of the page while a turn is running, and swapping the whole turn once
when it finishes - so the message thread above never flickers.

This app reuses the *same* checkpointer/thread_id across turns (via InMemorySaver),
because the apply() tool's interrupt()/Command(resume=...) pattern needs to
resume the exact paused graph run, not just replay prior messages.
"""

import json
import re
import threading
import uuid
from html import escape
from pathlib import Path

import markdown as _markdown
import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from phoenix.otel import using_session

from agent.graph import build_graph
from ui.rendering import format_audit_entry, normalize_messages

AUDIT_LOG_FILE = Path(__file__).parent.parent / "data" / "sandbox" / "audit_log.json"


def _render_markdown(text: str | None) -> str:
    # escape first so any stray HTML in an LLM message renders inert, then let
    # Markdown turn **bold** / lists / etc. into real elements
    safe = escape(text or "")
    # LLMs routinely omit the blank line Markdown wants before a list
    safe = re.sub(r"(?<=\S)\n(?=(?:[-*+]|\d+\.)\s)", "\n\n", safe)
    html = _markdown.markdown(safe, extensions=["sane_lists"])
    # Make links open in a new tab
    html = re.sub(r'<a href="([^"]*)">', r'<a href="\1" target="_blank" rel="noopener">', html)
    return html


app = FastAPI(title="Revenue Leakage Agent")
templates = Jinja2Templates(directory="ui/templates")
templates.env.filters["markdown"] = _render_markdown

graph = build_graph(InMemorySaver())

# Conversations for this process run, keyed by id (== LangGraph thread_id).
# In-memory only - same lifetime/durability as the old Streamlit app's
# st.session_state (gone on restart), and as InMemorySaver itself.
_conversations: dict[str, dict] = {}
_lock = threading.Lock()


def _execute(conv_id: str, *, message: str | None, resume: str | None) -> None:
    config = {"configurable": {"thread_id": conv_id}}
    try:
        with using_session(conv_id):
            stream_input = Command(resume=resume) if resume else {
                "messages": [{"role": "user", "content": message}]
            }
            for update in graph.stream(stream_input, config=config, stream_mode="updates"):
                for node_name in update:
                    with _lock:
                        _conversations[conv_id]["log"].append({"node": node_name})
    except Exception as exc:  # surfaced in the UI, not swallowed
        with _lock:
            _conversations[conv_id]["error"] = str(exc)

    with _lock:
        _conversations[conv_id]["running"] = False


def _start_turn(conv_id: str, *, message: str | None = None, resume: str | None = None) -> None:
    with _lock:
        _conversations[conv_id].update(log=[], error=None, running=True)
    threading.Thread(target=_execute, args=(conv_id,), kwargs={"message": message, "resume": resume}, daemon=True).start()


def _conversation_context(conv: dict) -> dict:
    config = {"configurable": {"thread_id": conv["id"]}}
    state = graph.get_state(config)
    pending = next((t.interrupts[0] for t in state.tasks if t.interrupts), None)
    raw_audit_log = json.loads(AUDIT_LOG_FILE.read_text()) if AUDIT_LOG_FILE.exists() else []
    # most-recent-first: what did I just do is the more common question than
    # what happened first
    audit_log = [format_audit_entry(e) for e in reversed(raw_audit_log)]
    return {
        "conv_id": conv["id"],
        "messages": normalize_messages(state.values.get("messages", []) if state.values else []),
        "pending_interrupt": pending.value["draft"] if pending else None,
        "running": conv["running"],
        "error": conv["error"],
        "log": conv["log"],
        "audit_log": audit_log,
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.post("/conversations")
def create_conversation(message: str = Form(...)):
    message = message.strip()
    conv_id = uuid.uuid4().hex[:8]
    with _lock:
        _conversations[conv_id] = {"id": conv_id, "running": False, "error": None, "log": []}
    _start_turn(conv_id, message=message)
    return RedirectResponse(f"/conversations/{conv_id}", status_code=303)


@app.post("/conversations/{conv_id}/messages")
def add_message(conv_id: str, message: str = Form(...)):
    conv = _conversations.get(conv_id)
    if conv is not None and not conv["running"]:
        message = message.strip()
        if message:
            _start_turn(conv_id, message=message)
    return RedirectResponse(f"/conversations/{conv_id}", status_code=303)


@app.post("/conversations/{conv_id}/resume")
def resume_conversation(conv_id: str, decision: str = Form(...)):
    conv = _conversations.get(conv_id)
    if conv is not None and not conv["running"] and decision in ("approve", "reject"):
        _start_turn(conv_id, resume=decision)
    return RedirectResponse(f"/conversations/{conv_id}", status_code=303)


@app.get("/conversations/{conv_id}", response_class=HTMLResponse)
def get_conversation(request: Request, conv_id: str):
    conv = _conversations.get(conv_id)
    if conv is None:
        return RedirectResponse("/", status_code=303)
    ctx = _conversation_context(conv)
    if not request.headers.get("hx-request"):
        return templates.TemplateResponse(request, "conversation.html", ctx)
    # htmx poll during a run: swap only the volatile tail (log + reply box), so
    # the message thread above is left untouched - no flash, no scroll jump.
    if conv["running"]:
        return templates.TemplateResponse(request, "turn_tail.html", ctx)
    # run finished: replace the whole #turn once so the new agent message
    # lands, and polling stops (the fresh tail carries no hx-get).
    resp = templates.TemplateResponse(request, "turn_fragment.html", ctx)
    resp.headers["HX-Retarget"] = "#turn"
    resp.headers["HX-Reswap"] = "outerHTML"
    return resp


app.mount("/static", StaticFiles(directory="ui/static"), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
