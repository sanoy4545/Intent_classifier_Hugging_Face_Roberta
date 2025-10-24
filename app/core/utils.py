import csv,os,json,re,emoji
import zipfile
import logging
from app.config import settings
from typing import Dict, Any, List

# Configure logging
def setup_logging(enable_logging: bool = True):
    """
    Sets up global logging. 
    If enable_logging=False, disables all logging across the app.
    """
    if not enable_logging:
        logging.disable(logging.CRITICAL)
        return

    # Create logs directory if not exists
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "app.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

def clean_text(text: str) -> str:
    """
    Clean a single message:
    - Lowercase
    - Remove emojis
    - Remove special characters except basic punctuation
    - Trim extra spaces
    """
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    text = text.lower()
    text = ' '.join(text.split())
    return text.strip()


def truncate_messages(messages: List[Dict[str, Any]], max_messages: int = 5) -> List[Dict[str, Any]]:
    """
    Keep only the last N messages.
    """
    return messages[-max_messages:]


def format_conversation(messages: List[Dict[str, Any]]) -> Any:
    """
    Concatenate messages into a Python-style parenthesis string for history
    and return last message separately.
    """
    cleaned_texts = []
    for msg in messages:
        sender = msg.get("sender", "").lower()
        text = clean_text(msg.get("text", ""))
        cleaned_texts.append(f"{sender}: {text}")

    if len(cleaned_texts) == 1:
        return "", cleaned_texts[0]

    # All except last message, concatenated with spaces
    history = " ".join(f'{m}' for m in cleaned_texts[:-1]) 
    last_message = cleaned_texts[-1]
    return history, last_message


def preprocess_data(data: List[Dict[str, Any]], max_messages: int = 5) -> List[Dict[str, Any]]:
    processed = []

    for convo in data:
        conv_id = convo.get("conversation_id", "unknown_id")
        messages = convo.get("messages", [])

        if not messages:
            continue

        truncated = truncate_messages(messages, max_messages)
        history, last_message = format_conversation(truncated)

        processed.append({
            "conversation_id": conv_id,
            "history": history,
            "last_message": last_message
        })

    return processed


def output_writer(predictions: List[Dict[str, str]]) -> None:
    """Write classification results to JSON and CSV files."""

    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "classification_results.json")
    csv_path = os.path.join(output_dir, "classification_results.csv")
    zip_path = os.path.join(output_dir, "classification_results.zip")

    # --- Write JSON ---
    with open(json_path, "w", encoding="utf-8") as f_json:
        json.dump(predictions, f_json, indent=2, ensure_ascii=False)
    
    # --- Write CSV ---
    if predictions:  # Only if the list is not empty
        with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
            fieldnames = ["conversation_id", "predicted_intent", "rationale"]
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()
            for row in predictions:
                # Ensure only the desired keys are written
                writer.writerow({key: row.get(key, "") for key in fieldnames})
    
    with zipfile.ZipFile(zip_path, "w") as zipf:
        zipf.write(json_path, "predictions.json")
        zipf.write(csv_path, "predictions.csv")
    
    return zip_path

# Initialize logging when module is imported
setup_logging()


