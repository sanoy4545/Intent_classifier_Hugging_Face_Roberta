

def build_zero_shot_prompt(history_text: str, last_msg_text: str, intent: str) -> str:
    """Build a prompt for zero-shot classification using conversation history and last message"""
    
    prompt = (
    f"Analyze the following multi-turn conversation between a user and a business:\n\n"
    f"{history_text}\n\n"
    f"The last message was:\n\"{last_msg_text}\"\n\n"
    f"Question: Does the overall conversation, indicate the user's intent is '{intent}'?"
    )

    return prompt

