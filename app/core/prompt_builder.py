from typing import List, Dict, Any
from app.config import settings

def build_zero_shot_prompt(current_message: str, history: List[Dict[str, Any]]) -> str:
    """Build a prompt for zero-shot classification using conversation history."""
    # Take the last N turns based on MAX_HISTORY_TURNS setting
    recent_history = history[-settings.MAX_HISTORY_TURNS:]
    
    # Construct context from history
    context = []
    for turn in recent_history:
        context.append(f"User: {turn['user']}")
        if turn.get('assistant'):
            context.append(f"Assistant: {turn['assistant']}")
    
    # Add current message
    context.append(f"User: {current_message}")
    
    # Build the final prompt
    prompt = (
        "Based on the conversation history below, classify the last user message into one of "
        f"these intents: {', '.join(settings.ALLOWED_INTENTS)}.\n\n"
        "Conversation:\n" + "\n".join(context)
    )
    
    return prompt

def extract_relevant_context(history: List[Dict[str, Any]], current_message: str) -> str:
    """Extract relevant context from conversation history for intent classification."""
    return build_zero_shot_prompt(current_message, history)