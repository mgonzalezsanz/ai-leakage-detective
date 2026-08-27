"""Simple chat UI for the Revenue Leakage Agent, on top of the same compiled
LangGraph agent used by `langgraph dev`. Run with: streamlit run streamlit_app.py
"""

import json
import uuid
from pathlib import Path

import streamlit as st
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from phoenix.otel import using_session

from agent.graph import build_graph

AUDIT_LOG_FILE = Path(__file__).parent / "data" / "sandbox" / "audit_log.json"

st.set_page_config(page_title="Revenue Leakage Agent", page_icon="\U0001f4b0")


@st.cache_resource
def get_graph():
    return build_graph(InMemorySaver())


graph = get_graph()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

config = {"configurable": {"thread_id": st.session_state.thread_id}}

st.title("\U0001f4b0 Revenue Leakage Agent")
st.caption(
    "Investigates billing plan vs. invoice discrepancies. Nothing is written "
    "to the sandbox without your approval."
)

with st.sidebar:
    if st.button("New conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.subheader("Sandbox audit log")
    if AUDIT_LOG_FILE.exists():
        st.json(json.loads(AUDIT_LOG_FILE.read_text()))
    else:
        st.caption("No actions applied yet.")


def render_messages(messages):
    tool_results = {m.tool_call_id: m.content for m in messages if m.type == "tool"}
    for msg in messages:
        if msg.type == "human":
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif msg.type == "ai":
            text = msg.content if isinstance(msg.content, str) else "".join(
                block.get("text", "") for block in msg.content if isinstance(block, dict)
            )
            with st.chat_message("assistant"):
                if text:
                    st.markdown(text)
                for tc in msg.tool_calls:
                    with st.expander(f"\U0001f50d {tc['name']}({tc['args']})"):
                        st.code(tool_results.get(tc["id"], "(pending)"), language="json")


state = graph.get_state(config)
messages = state.values.get("messages", []) if state.values else []
render_messages(messages)

pending_interrupt = next((t.interrupts[0] for t in state.tasks if t.interrupts), None)

if pending_interrupt:
    draft = pending_interrupt.value["draft"]
    st.info(f"**Approval needed** - apply this action?\n\n```json\n{json.dumps(draft, indent=2)}\n```")
    col1, col2 = st.columns(2)
    if col1.button("✅ Approve", type="primary"):
        with using_session(st.session_state.thread_id):
            graph.invoke(Command(resume="approve"), config=config)
        st.rerun()
    if col2.button("❌ Reject"):
        with using_session(st.session_state.thread_id):
            graph.invoke(Command(resume="reject"), config=config)
        st.rerun()
else:
    if user_input := st.chat_input("Ask about a plan, invoice, or discrepancy..."):
        with using_session(st.session_state.thread_id):
            graph.invoke({"messages": [{"role": "user", "content": user_input}]}, config=config)
        st.rerun()
