"""FastAPI application for the LLM Agent System."""

import os
from fastapi import FastAPI
from pydantic import BaseModel

from src.agent import run_agent_loop
from src.providers import DummyProvider, LLMProvider, get_provider

app = FastAPI(title="Local LLM Agent System")

# Active provider initialized from environment variable LLM_PROVIDER or default DummyProvider
provider_name = os.getenv("LLM_PROVIDER", "dummy")
try:
    active_provider: LLMProvider = get_provider(provider_name)
except Exception:
    active_provider = DummyProvider()


class AgentRequest(BaseModel):
    prompt: str


@app.post("/agent/chat")
async def agent_chat(request: AgentRequest):
    response = run_agent_loop(request.prompt, provider=active_provider)
    return {"response": response}


@app.get("/")
def health_check():
    health = active_provider.check_health()
    return {
        "status": health.status,
        "is_online": health.is_online,
        "model": health.model,
        "provider": active_provider.__class__.__name__,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
