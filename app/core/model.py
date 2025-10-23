from typing import List, Tuple, Optional
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class IntentClassifier:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load_model(self) -> None:
        """Load the Gemma (or similar) model for intent classification."""
        try:
            model_path = settings.MODEL_PATH or settings.MODEL_NAME
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            self.model.eval()
            logger.info(f"Model loaded successfully: {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def classify(
    self,
    data: dict,  # {'conversation_id': ..., 'history': ..., 'last_message': ...}
    candidate_intents: Optional[List[str]] = None
) -> dict:
        """
        Classify the last user message in a multi-turn conversation.

        Args:
            data: Dict containing 'conversation_id', 'history', 'last_message'.
            candidate_intents: List of allowed intents.

        Returns:
            Dict with predicted_intent, confidence, and rationale.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        intents = candidate_intents or settings.ALLOWED_INTENTS
        conversation_id = data.get("conversation_id", "")
        history_text = data.get("history", "")
        last_msg_text = data.get("last_message", "")

        best_intent = None
        best_score = float('-inf')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        for intent in intents:
            # Improved zero-shot prompt
            nli_input = (
            f"You are an intent classifier. Your task is to classify the last user message "
            f"in a conversation into one of these intents: {', '.join(intents)}.\n\n"
            f"Conversation context:\n{history_text}\n\n"
            f"Last user message:\n{last_msg_text}\n\n"
            f"Question: Does the last message express the intent '{intent}'? Answer Yes or No."
        )


            inputs = self.tokenizer(
                nli_input, return_tensors="pt", truncation=True, padding=True, max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.softmax(outputs.logits, dim=1)
                entailment_score = scores[0][1].item()  # Probability of entailment

            if entailment_score > best_score:
                best_score = entailment_score
                best_intent = intent

        # Placeholder rationale
        rationale = f"The intent is predicted based on the last user message and conversation history."

        return {
            "conversation_id": conversation_id,
            "predicted_intent": best_intent,
            "confidence": best_score,
            "rationale": rationale
        }


# Global instance
classifier = IntentClassifier()
'''classifier.load_model()
sample_input = {
    "conversation_id": "conv_001",
    "history": (
        "user: Hi, I'm looking for a 2BHK in Dubai "
        "agent: Great! Any specific area in mind? "
        "user: Preferably Marina or JVC "
        "agent: What's your budget?"
    ),
    "last_message": "user: Max 120k. Can we do a site visit this week?"
}
sample_input_2 = {
    "conversation_id": "conv_002",
    "history": (
        "user: Hello, I have an issue with my recent order "
        "agent: Sorry to hear that! Can you describe the problem? "
        "user: The item arrived damaged "
        "agent: We can replace it or issue a refund. Which do you prefer?"
    ),
    "last_message": "user: I would like a replacement, please."
}


print(classifier.classify(sample_input))  # Example usage
print(classifier.classify(sample_input_2))  # Example usage'''