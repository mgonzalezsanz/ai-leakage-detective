"""Workaround for a ragas==0.4.3 bug: ragas/llms/base.py does an unconditional
`from langchain_community.chat_models.vertexai import ChatVertexAI` at import
time, but that submodule has been removed from recent langchain-community
releases - breaking `import ragas` for every non-Vertex user.
"""

import sys
import types

if "langchain_community.chat_models.vertexai" not in sys.modules:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("ChatVertexAI stub - Vertex AI is not used in this project")

    _stub.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _stub
