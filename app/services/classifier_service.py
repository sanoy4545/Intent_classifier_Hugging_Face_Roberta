from typing import Dict, Any, List
import logging
from app.core.model import classifier
from app.core.prompt_builder import build_zero_shot_prompt
from app.core.rationale import generate_rationale
from app.core.utils import preprocess_data, format_classification_result
from app.config import settings

logger = logging.getLogger(__name__)

class ClassifierService:
    def classify_message(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Orchestrate the classification process:
        1. Preprocess the message
        2. Validate and clean history
        3. Build the classification prompt
        4. Run inference
        5. Generate rationale
        6. Format and return result
        """
        try:
            # Preprocess message
            cleaned_data = preprocess_data(data)
            
            # Build prompt with context
            #prompt = build_zero_shot_prompt(cleaned_data)

            # Run classification
            intent, confidence = classifier.classify(cleaned_data,settings.ALLOWED_INTENTS)
            
            # Generate rationale if enabled
            rationale = generate_rationale(
                intent=intent,
                confidence=confidence,
                message=cleaned_message,
                history=cleaned_history
            ) if settings.ENABLE_RATIONALE else None
            
            # Format result
            result = format_classification_result(
                intent=intent,
                confidence=confidence,
                rationale=rationale,
                message=cleaned_message
            )
            
            logger.info(f"Successfully classified message with intent: {intent}")
            return result
            
        except Exception as e:
            logger.error(f"Error in classification service: {str(e)}")
            raise