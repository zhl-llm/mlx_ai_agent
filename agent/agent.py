from fastapi import FastAPI
from pydantic import BaseModel
from agent_service import run_agent_with_history

app = FastAPI()

class ChatRequest(BaseModel):
    history: list[list[str]]  # [[user, assistant], ...]

@app.post("/chat")
def chat(req: ChatRequest):
    answer = run_agent_with_history(req.history)
    return {"answer": answer}

