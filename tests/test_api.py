import pytest
from fastapi.testclient import TestClient
from app.fastapi_app import app
from app.core.model import classifier

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_model():
    """Ensure model is loaded for tests."""
    if classifier.model is None:
        classifier.load_model()
    yield

def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_list_intents():
    """Test listing available intents."""
    response = client.get("/intents")
    assert response.status_code == 200
    intents = response.json()
    assert isinstance(intents, list)
    assert len(intents) > 0
    assert all(isinstance(intent, str) for intent in intents)

def test_classify_simple_message():
    """Test classification of a simple message."""
    payload = {
        "message": "What time is it?",
        "history": []
    }
    response = client.post("/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "intent" in data
    assert "confidence" in data
    assert isinstance(data["confidence"], float)
    assert 0 <= data["confidence"] <= 1

def test_classify_with_history():
    """Test classification with conversation history."""
    payload = {
        "message": "Yes, that's correct",
        "history": [
            {
                "user": "What time is it?",
                "assistant": "It's 3:00 PM"
            }
        ]
    }
    response = client.post("/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "intent" in data
    assert "confidence" in data
    assert "rationale" in data

def test_invalid_message():
    """Test classification with invalid input."""
    payload = {
        "message": "",  # Empty message
        "history": []
    }
    response = client.post("/classify", json=payload)
    assert response.status_code != 200  # Should fail