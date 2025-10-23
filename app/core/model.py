from typing import List, Tuple, Optional
import torch
from app.core.prompt_builder import build_zero_shot_prompt
from transformers import AutoModelForSequenceClassification, AutoTokenizer,pipeline
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class IntentClassifier:
    def __init__(self):
        self.model = []
        self.tokenizer = []


    def load(self, model_name) -> None:
        """Load the models for intent classification dynamically."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            generator = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info(f"Model loaded successfully from {model_name} ")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
        return model, tokenizer
    

    def load_model(self) -> None:
        """Load all models specified in settings dynamically."""
        for model_name in settings.MODELS:
            model, tokenizer = self.load(model_name)
            self.model.append(model)
            self.tokenizer.append(tokenizer)

    def model_response(self, intent: str, history_text: str, last_msg_text: str, model, tokenizer, device) -> float:
        model.to(device)
        """Generate model response and return confidence score. This is based on sequence classification models. Change if required."""
        prompt = build_zero_shot_prompt(history_text, last_msg_text, intent)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)
            confidence = scores[0][1].item()  # Get confidence score for the positive class
        return confidence


    def prediction_logic(self, intents, history_text, last_msg_text, device):
        """Core prediction logic to determine best intent and confidence. Change Logic here if needed."""

        best_intent = None
        best_score = float('-inf')
        confidence_scores = {}

        for intent in intents:

            for model, tokenizer in zip(self.model, self.tokenizer):
                confidence = self.model_response(intent, history_text, last_msg_text, model, tokenizer, device)
                confidence_scores[intent] = (confidence_scores.get(intent, 0) + confidence) / 2

                if confidence > best_score:
                    best_score = confidence
                    best_intent = intent

        return best_intent, best_score

    def classify(
    self,
    data: dict,  # {'conversation_id': ..., 'history': ..., 'last_message': ...}
) -> dict:
        """
        Classify the last user message in a multi-turn conversation.

        Args:
            data: Dict containing 'conversation_id', 'history', 'last_message'.
            candidate_intents: List of allowed intents.

        Returns:
            Dict with predicted_intent, and confidence
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        intents = settings.ALLOWED_INTENTS
        conversation_id = data.get("conversation_id", "")
        history_text = data.get("history", "")
        last_msg_text = data.get("last_message", "")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        best_intent,best_score = self.prediction_logic(intents,history_text,last_msg_text,device)

        

        return {
            "conversation_id": conversation_id,
            "predicted_intent": best_intent,
            "confidence": best_score,          #change before submitting
        }



# Global instance
classifier = IntentClassifier()
'''classifier.load_model()
print(len(classifier.model))
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