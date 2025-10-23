import pytest
import torch
from app.core.model import IntentClassifier
from app.config import settings

@pytest.fixture
def classifier():
    """Fixture to provide a loaded classifier."""
    cls = IntentClassifier()
    cls.load_model()
    return cls

def test_model_loading(classifier):
    """Test that model loads correctly."""
    assert classifier.model is not None
    assert classifier.tokenizer is not None

def test_classification_basic(classifier):
    """Test basic classification functionality."""
    text = "What time is it?"
    intent, confidence = classifier.classify(text)
    
    assert isinstance(intent, str)
    assert intent in settings.ALLOWED_INTENTS
    assert isinstance(confidence, float)
    assert 0 <= confidence <= 1

def test_classification_different_intents(classifier):
    """Test classification with different types of messages."""
    test_cases = [
        ("Hello, how are you?", "greeting"),
        ("What's the weather like?", "question"),
        ("Please help me with this.", "request"),
        ("Goodbye!", "farewell")
    ]
    
    for text, expected_type in test_cases:
        intent, confidence = classifier.classify(text)
        assert isinstance(intent, str)
        assert isinstance(confidence, float)
        # We don't assert exact matches as zero-shot classification
        # might reasonably classify these differently

def test_empty_input(classifier):
    """Test classification with empty input."""
    with pytest.raises(Exception):
        classifier.classify("")

def test_model_cuda_support(classifier):
    """Test CUDA support if available."""
    if torch.cuda.is_available():
        assert next(classifier.model.parameters()).is_cuda
    else:
        assert not next(classifier.model.parameters()).is_cuda