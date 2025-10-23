from typing import Dict, List, Any
from app.config import settings

def generate_rationale(
    intent: str,
    confidence: float,
    message: str,
    history: List[Dict[str, Any]]
) -> str:
    """Generate a human-readable rationale for the intent classification."""
    if not settings.ENABLE_RATIONALE:
        return ""
        
    # Start with base explanation
    rationale = [
        f"Intent '{intent}' was detected",
        f"Confidence: {confidence:.2%}"
    ]
    
    # Add context-specific reasoning
    if confidence >= settings.CONFIDENCE_THRESHOLD:
        rationale.append("This classification is based on:")
        
        # Add message-specific reasoning
        if "question" in intent.lower() and "?" in message:
            rationale.append("- The message ends with a question mark")
        elif "greeting" in intent.lower() and any(word in message.lower() for word in ["hi", "hello", "hey"]):
            rationale.append("- The message contains common greeting words")
        elif "farewell" in intent.lower() and any(word in message.lower() for word in ["bye", "goodbye", "see you"]):
            rationale.append("- The message contains common farewell expressions")
            
        # Add context from conversation history if available
        if history:
            recent_context = history[-settings.MAX_HISTORY_TURNS:]
            if len(recent_context) > 0:
                rationale.append("- Conversation context was considered")
                
    else:
        rationale.append("Note: Classification confidence is below the threshold")
        
    return "\n".join(rationale)