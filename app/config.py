from pydantic_settings import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    # Model configuration (Gemma, DistilBERT, RoBERTa, etc.)
    MODEL_NAME: str = "distilbert-base-uncased-finetuned-sst-2-english"  # or any open-source model you choose
    MODEL_PATH: Optional[str] = None  # Path to local model directory if downloaded

    # Allowed final intents for classification
    ALLOWED_INTENTS: List[str] = [
        "Book Appointment",
        "Product Inquiry",
        "Pricing Negotiation",
        "Support Request",
        "Follow-Up"
    ]

    # Classification parameters
    MAX_HISTORY_TURNS: int = 5
    CONFIDENCE_THRESHOLD: float = 0.7
    ENABLE_RATIONALE: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
