from pydantic_settings import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    # Model configuration (Gemma, DistilBERT, RoBERTa, etc.)
    MODELS: List[str] = ["roberta-large-mnli","microsoft/deberta-base"]  # or any open-source model you choose
    # Allowed final intents for classification
    ALLOWED_INTENTS: List[str] = [
        "Book Appointment",
        "Product Inquiry",
        "Pricing Negotiation",
        "Support Request",
        "Follow-Up"
    ]
    INTENT_KEYWORDS: dict = {
        "Book Appointment": ["schedule", "appointment", "visit", "viewing", "tour", "meet", "book", "come see"],
        "Product Inquiry": ["looking for", "need", "bhk", "property", "details", "specifications", "tell me about"],
        "Pricing Negotiation": ["budget", "price", "cost", "negotiate", "discount", "max", "afford", "deal"],
        "Support Request": ["issue", "problem", "help", "support", "not working", "error", "fix", "urgent"],
        "Follow-Up": ["following up", "update", "status", "waiting", "checking in", "any news", "previously"]
    }

    # Classification parameters
    MAX_HISTORY_TURNS: int = 5

    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
