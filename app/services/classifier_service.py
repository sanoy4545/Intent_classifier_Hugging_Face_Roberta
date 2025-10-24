from typing import Dict, Any, List
import logging
from app.core import rationale
from app.core.model import classifier
from app.core.prompt_builder import build_zero_shot_prompt
from app.core.rationale import generate_rationale
from app.core.utils import preprocess_data, output_writer
from app.config import settings

logger = logging.getLogger(__name__)

class ClassifierService:
    def classify_conversations(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Orchestrate the classification process:
        Args:
            data: List of conversations to classify
        """
        try:
            # Preprocess message
            cleaned_data = preprocess_data(data)
            logger.info(f"Preprocessed {len(cleaned_data)} conversations for classification")

            # Run classification
            results = []
            for convo in cleaned_data:
                result = classifier.classify(convo)

                # Generate rationale 
                result['rationale'] = generate_rationale(
                    conversation_text=convo.get("history", "") + convo.get("last_message", ""),
                    predicted_intent=result.get("predicted_intent", "")
                )

                results.append(result)
            
            # Format result
            output=output_writer(results)

            logger.info(f"Successfully classified message with intent")

            #return zip file path 
            return output

        except Exception as e:
            logger.error(f"Error in classification service: {str(e)}")
            raise