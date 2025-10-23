import pytest
from app.core.prompt_builder import build_zero_shot_prompt, extract_relevant_context
from app.config import settings

def test_build_zero_shot_prompt_simple():
    """Test prompt building with a simple message."""
    message = "Hello there!"
    history = []
    prompt = build_zero_shot_prompt(message, history)
    
    assert isinstance(prompt, str)
    assert message in prompt
    assert all(intent in prompt for intent in settings.ALLOWED_INTENTS)

def test_build_zero_shot_prompt_with_history():
    """Test prompt building with conversation history."""
    message = "Yes, please do that"
    history = [
        {"user": "Can you help me?", "assistant": "Of course! What do you need?"}
    ]
    prompt = build_zero_shot_prompt(message, history)
    
    assert isinstance(prompt, str)
    assert message in prompt
    assert history[0]["user"] in prompt
    assert history[0]["assistant"] in prompt

def test_build_zero_shot_prompt_history_limit():
    """Test that prompt respects MAX_HISTORY_TURNS setting."""
    message = "Final message"
    history = [
        {"user": f"Message {i}", "assistant": f"Response {i}"}
        for i in range(settings.MAX_HISTORY_TURNS + 5)
    ]
    prompt = build_zero_shot_prompt(message, history)
    
    # Check that only MAX_HISTORY_TURNS are included
    history_turns = len([line for line in prompt.split("\n") 
                        if line.startswith("User:") or line.startswith("Assistant:")])
    assert history_turns <= (settings.MAX_HISTORY_TURNS * 2 + 1)  # *2 for user+assistant, +1 for current message

def test_extract_relevant_context():
    """Test context extraction from history."""
    message = "Yes, please"
    history = [
        {"user": "Can you help?", "assistant": "Sure!"}
    ]
    context = extract_relevant_context(history, message)
    
    assert isinstance(context, str)
    assert message in context
    assert history[0]["user"] in context