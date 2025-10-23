

def build_zero_shot_prompt(history_text: str, last_msg_text: str, intent: str) -> str:
    """Build a prompt for zero-shot classification using conversation history."""
    '''nli_input = (
                f"Premise: In a conversation, the user previously discussed: {history_text}\n\n"
                f"Hypothesis: The user's current message '{last_msg_text}' expresses the intent to {intent}.\n\n"
                f"Does the hypothesis logically follow from the premise?"
            )'''
    
    '''prompt = (
        "Based on the conversation history below, classify the last user message into one of "
        f"these intents: {', '.join(settings.ALLOWED_INTENTS)}.\n\n"
        "Conversation:\n" + "\n".join(context)
    )'''
    
    prompt = (
        f"Given the conversation context: {history_text}\n"
        f"The statement '{last_msg_text}' indicates intent: {intent}"
    )
    return prompt

