# Multi-turn Intent Classifier

A FastAPI-based service that performs zero-shot intent classification for multi-turn conversations using transformer models.

## Features

- Zero-shot intent classification for natural language messages
- Multi-turn conversation context support
- Detailed classification rationale generation
- RESTful API with FastAPI
- Containerized deployment support
- Comprehensive test suite

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd multi_turn_intent_classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Locally

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
- `GET /intents`: List all available intent categories
- `GET /health`: Check service health status

### Example Request

```json
{
    "message": "What time is it?",
    "history": [
        {
            "user": "Hello",
            "assistant": "Hi there! How can I help you?"
        }
    ]
}
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