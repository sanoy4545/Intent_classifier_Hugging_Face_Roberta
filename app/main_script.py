import json
import logging
import argparse
from app.core.model import classifier
from app.core.utils import setup_logging
from app.services.classifier_service import ClassifierService

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize classifier service
classifier_service = ClassifierService()


def main(input_json: str, json_output: str = "predictions.json", csv_output: str = "predictions.csv"):
    """
    Read a JSON file containing multiple conversations, classify intents,
    and write results to JSON and CSV.
    """
    try:
        # Load input JSON
        with open(input_json, "r", encoding="utf-8") as f:
            conversations = json.load(f)
        
        logger.info(f"Loaded {len(conversations)} conversations from {input_json}")

        # Ensure model is loaded
        classifier.load_model()
        logger.info("Model loaded successfully")

        # Classify all conversations
        classifier_service.classify_conversations(conversations)


    except FileNotFoundError:
        logger.error(f"File not found: {input_json}")
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON file: {input_json}")
    except Exception as e:
        logger.error(f"Error processing conversations: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch classify conversation intents")
    parser.add_argument("input_json", help="Path to input JSON file containing conversations")
    parser.add_argument("--json_output", default="predictions.json", help="Name of output JSON file")
    parser.add_argument("--csv_output", default="predictions.csv", help="Name of output CSV file")
    args = parser.parse_args()

    main(args.input_json, args.json_output, args.csv_output)
