import gradio as gr
import requests

API_URL = "http://localhost:5050/chat"

def chat_fn(message, history):
    """
    history: list of {"role": "...", "content": "..."}
    """
    agent_history = []

    i = 0
    while i < len(history) - 1:
        if history[i]["role"] == "user" and history[i + 1]["role"] == "assistant":
            agent_history.append([
                history[i]["content"],
                history[i + 1]["content"]
            ])
            i += 2
        else:
            i += 1

    agent_history.append([message, ""])

    resp = requests.post(
        API_URL,
        json={"history": agent_history},
        timeout=300
    )

    answer = resp.json()["answer"]
    return answer


demo = gr.ChatInterface(
    fn=chat_fn,
    title="🤖 Local ReAct Agent",
    description="Local LLM + tools + memory",
)

demo.launch(server_name="0.0.0.0", server_port=7860)
