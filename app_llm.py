from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Dict
from mlx_lm import load, generate
import json

# ---------------------------
# App
# ---------------------------
app = FastAPI(title="MLX LLM Server")

# ---------------------------
# Model loading (once)
# ---------------------------
MODEL_PATH = "/Users/zhlsunshine/Projects/inference/models/qwen2.5-14b-instruct-bits-8"

print("Loading model...")
model, tokenizer = load(MODEL_PATH)
print("Model loaded.")

# ---------------------------
# Pydantic schemas
# ---------------------------

class Message(BaseModel):
    type: str = Field(..., description="Message type: HumanMessage or AIMessage")
    content: str

class ChatMessage(BaseModel):
    messages: List[Message] = Field(..., description="List of conversation messages")
    max_tokens: int = Field(128, description="Maximum number of tokens to generate")

# ---------------------------
# Prompt builder
# ---------------------------
def build_prompt_from_messages(messages: List[Dict]) -> str:
    """
    Converts LangChain-style messages into a plain text prompt for the LLM.
    """
    prompt = ""
    for msg in messages:
        role = msg.get("type", "HumanMessage").replace("Message", "").capitalize()
        content = msg.get("content", "")
        prompt += f"{role}: {content}\n"
    prompt += "Assistant: "
    return prompt

# ---------------------------
# Health / info
# ---------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    return "<h2>MLX LLM Server (standardized messages)</h2>"

# ---------------------------
# Streaming chat endpoint
# ---------------------------
@app.post("/chat/stream")
async def chat_stream(item: ChatMessage):
    full_prompt = build_prompt_from_messages([msg.dict() for msg in item.messages])

    def token_generator():
        assistant_reply = ""

        for token in generate(
            model,
            tokenizer,
            prompt=full_prompt,
            max_tokens=item.max_tokens,
        ):
            assistant_reply += token
            yield json.dumps({"token": token}) + "\n"

        # Final message with updated conversation
        yield json.dumps({
            "finished": True,
            "messages": [*item.messages, {"type": "AIMessage", "content": assistant_reply}],
        }) + "\n"

    return StreamingResponse(token_generator(), media_type="application/json")

# ---------------------------
# Non-streaming chat endpoint
# ---------------------------
@app.post("/chat")
async def chat(item: ChatMessage):
    full_prompt = build_prompt_from_messages([msg.dict() for msg in item.messages])

    tokens = []
    for token in generate(
        model,
        tokenizer,
        prompt=full_prompt,
        max_tokens=item.max_tokens,
    ):
        tokens.append(token)

    response = "".join(tokens)

    return {
        "response": response,
        "messages": [*item.messages, {"type": "AIMessage", "content": response}],
    }
