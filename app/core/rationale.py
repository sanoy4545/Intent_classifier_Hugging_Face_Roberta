from app.config import settings  
    
def generate_rationale(conversation_text, predicted_intent):
        """Extract keywords and relevant quote
        Note: Intent keywords should be defined in settings.py as a dictionary mapping intents to lists of keywords."""

        
        intent_keywords = settings.INTENT_KEYWORDS
        
        # Find matching keywords
        matched_keywords = [
            kw for kw in intent_keywords[predicted_intent]
            if kw in conversation_text
        ]
        
        # Extract most relevant user message
        user_messages = [
            msg.strip() for msg in conversation_text.split('|')
            if 'user:' in msg.lower()
        ]
        
        # Find message with most keyword matches
        best_message = None
        max_matches = 0
        for msg in user_messages:
            matches = sum(1 for kw in matched_keywords if kw in msg.lower())
            if matches > max_matches:
                max_matches = matches
                best_message = msg.replace('user:', '').replace('User:', '').strip()
        
        # Truncate long messages
        if best_message and len(best_message) > 80:
            best_message = best_message[:77] + "..."
        
        # Generate rationale
        if matched_keywords and best_message:
            return f"The user mentioned '{matched_keywords[0]}' in: \"{best_message}\""
        elif matched_keywords:
            return f"Keywords detected: '{', '.join(matched_keywords[:2])}' indicating {predicted_intent}"
        else:
            return f"Conversation pattern and context indicate {predicted_intent}"
