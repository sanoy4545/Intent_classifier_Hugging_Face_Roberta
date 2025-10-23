# Multi-turn Intent Classifier

A FastAPI-based service that performs zero-shot intent classification for multi-turn conversations using transformer models.

## Features

- Zero-shot intent classification for natural language messages
- Multi-turn conversation context support
- Detailed classification rationale generation
- RESTful API with FastAPI
- Containerized deployment support

## Setup Instrcutions

1. Clone the repository:
```bash
git clone <repository-url>
cd multi_turn_intent_classifier
```
2. Create Virtual Environment:
```bash
python -m venv venv
venv\Scripts\activate      # On Windows  
source venv/bin/activate   # On Linux/Mac

```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

## Using Scripts
Place your input JSON file inside the input/ folder.
Example file:
    [
  {
    "conversation_id": "conv_001",
    "history": "Hi, I ordered a phone last week but it hasn’t arrived yet.",
    "last_message": "Can you check the status of my order?"
  },
  {
    "conversation_id": "conv_002",
    "history": "I tried logging in but it failed.",
    "last_message": "I think I forgot my password."
  }
]

'''bash
python -m app.main_script input/convo.json
'''
Logging enabled by default.
Creates /logs/ folder with a timestamped log file.

'''bash
python -m app.main_script input/convo.json --no_log
'''
Logging disabled

Outputs saved to:

    outputs/predictions.json

    outputs/predictions.csv


### FastAPI 

1. Start the FastAPI server:
```bash
uvicorn app.main:app --reload
```

2. Access the API documentation at http://localhost:8000/docs

### Using Docker

1. Build the Docker image:
```bash
docker build -t intent-classifier .
```

2. Run the container:
```bash
docker run -p 8000:8000 intent-classifier
```

## API Endpoints

- `POST /classify`: Classify the intent of a message with conversation context

### Example Request

```json
[{
  "conversation_id": "conv_001",
  "history": "Hi, I ordered a phone last week but it hasn’t arrived yet.",
  "last_message": "Can you check the status of my order?"
},
{
  "conversation_id": "conv_002",
  "history": "Hi, I ordered a phone last week but it hasn’t arrived yet.",
  "last_message": "Can you check the status of my order?"
}
]

```

### Example Response

```json
{
    "intent": "question",
    "confidence": 0.95,
    "rationale": "Intent 'question' was detected\nConfidence: 95%\nThis classification is based on:\n- The message ends with a question mark\n- Conversation context was considered",
    "message": "What time is it?",
    "timestamp": "2025-10-22T10:30:00.000Z"
}
```

## Configuration

Configuration options can be set through environment variables or `.env` file:

- `MODEL_NAME`: HuggingFace model to use (default: "facebook/bart-large-mnli")
- `MODEL_PATH`: Path to local model (optional)
- `MAX_HISTORY_TURNS`: Maximum conversation turns to consider (default: 5)
- `CONFIDENCE_THRESHOLD`: Minimum confidence for rationale generation (default: 0.7)
- `ENABLE_RATIONALE`: Enable/disable rationale generation (default: true)

## Testing

Run the test suite:

```bash
pytest tests/
```

## License

[Insert License Information]

## Contributing

[Insert Contribution Guidelines]