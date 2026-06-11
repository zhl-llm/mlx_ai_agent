import requests
import asyncio
from typing import List
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration

DEFAULT_LLM_ENDPOINT = "http://localhost:8000/chat"


def call_llm_server(messages: List[dict], max_tokens: int = 512) -> str:
    """Send conversation messages to the local LLM server."""
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
    }

    response = requests.post(
        DEFAULT_LLM_ENDPOINT,
        headers={
            "Content-Type": "application/json",
            "USER_AGENT": "mlx-ai-agent/0.1 (local)",
        },
        json=payload,
    )

    try:
        result = response.json()
        return result.get("response", "")
    except Exception:
        return response.text


class MyChatLLM(BaseChatModel):
    """Minimal custom chat model for LangChain / LangGraph."""

    def _format_messages(self, messages: List) -> List[dict]:
        """
        Convert LangChain messages to the backend format:
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
                "content": m.content,
            })
        return api_messages

    def _generate(self, messages: List, stop=None):
        api_messages = self._format_messages(messages)
        output_text = call_llm_server(api_messages)
        ai_msg = AIMessage(content=output_text)
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    async def _agenerate(self, messages: List, stop=None):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._generate(messages, stop))

    @property
    def _llm_type(self) -> str:
        return "custom-chat"
