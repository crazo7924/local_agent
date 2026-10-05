"""Tests for FastAPI endpoints."""

from unittest.mock import patch
from fastapi.testclient import TestClient

from main import app
from src.providers import MockProvider

client = TestClient(app)


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "is_online" in data
    assert "model" in data
    assert "provider" in data


def test_agent_chat_endpoint():
    mock_provider = MockProvider(default_content="Test response")
    with patch("main.active_provider", mock_provider):
        response = client.post("/agent/chat", json={"prompt": "Hello"})
        assert response.status_code == 200
        assert response.json() == {"response": "Test response"}
