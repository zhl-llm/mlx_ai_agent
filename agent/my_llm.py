import requests
import asyncio
from typing import List, Optional, Union
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration

# ---------------------------
# Request helper
# ---------------------------
def mychatllm_calling(messages: List[dict], max_tokens: int = 1024) -> str:
    """
    Sends conversation messages to the FastAPI LLM server and returns text.
    """
    url = 'http://localhost:8000/chat'

    # Convert API messages to the new "messages" field expected by app.py
    payload = {
        "messages": messages,
        "max_tokens": max_tokens
    }

    response = requests.post(
        url,
        headers={
            "Content-Type": "application/json",
            "USER_AGENT": "custom-agent 1.0"
        },
        json=payload
    )

    try:
        result = response.json()
        # Our app.py returns:
        # { "response": "...", "messages": [...] }
        return result.get("response", "")
    except Exception:
        return response.text

# ---------------------------
# MyChatLLM class
# ---------------------------
class MyChatLLM(BaseChatModel):
    """Minimal custom chat model for LangChain / LangGraph."""

    def _format_messages(self, messages: List) -> List[dict]:
        """
        Convert LangChain messages → backend format for app.py:
        [{"type": "HumanMessage"/"AIMessage", "content": "..."}]
        """
        api_messages = []
        for m in messages:
            if isinstance(m, HumanMessage):
                msg_type = "HumanMessage"
            elif isinstance(m, AIMessage):
                msg_type = "AIMessage"
            elif isinstance(m, SystemMessage):
                msg_type = "SystemMessage"
            else:
                msg_type = "HumanMessage"

            api_messages.append({
                "type": msg_type,
                "content": m.content
            })
        return api_messages

    # ---------------------------
    #  Synchronous generation
    # ---------------------------
    def _generate(self, messages: List, stop=None):
        api_messages = self._format_messages(messages)
        output_text = mychatllm_calling(api_messages)
        ai_msg = AIMessage(content=output_text)
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    # ---------------------------
    #  Async generation
    # ---------------------------
    async def _agenerate(self, messages: List, stop=None):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._generate(messages, stop))

    @property
    def _llm_type(self) -> str:
        return "custom-chat"
