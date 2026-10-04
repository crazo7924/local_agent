import ollama
from fastapi import FastAPI
from pydantic import BaseModel

from src.agent import run_agent_loop
from src.config import MODEL_NAME

app = FastAPI(title="Local LLM Agent System")


class AgentRequest(BaseModel):
    prompt: str


@app.post("/agent/chat")
async def agent_chat(request: AgentRequest):
    response = run_agent_loop(request.prompt)
    return {"response": response}


@app.get("/")
def health_check():
    ps_response = ollama.ps()
    for model in ps_response.models:
        if model.name == MODEL_NAME:
            return {"status": "Agent is online", "model": MODEL_NAME}

    return {"status": "Agent is NOT running", "model": MODEL_NAME}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
