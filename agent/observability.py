"""Phoenix tracing setup for the Revenue Leakage Agent.

Requires `phoenix serve` running locally.
This only wires the OTel exporter and auto-instruments LangChain (which covers 
LLM calls and every @tool invocation); it does not start a Phoenix server itself.
If Phoenix isn't running, span export fails silently and the app keeps working.
"""

from phoenix.otel import register


def setup_tracing() -> None:
    register(auto_instrument=True)
