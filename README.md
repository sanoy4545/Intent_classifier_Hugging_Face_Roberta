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
Note: Remove fastapi,uvicorn and streamlit if not using fastapi

## Usage

### Using Scripts
Place your input JSON file inside the input/ folder.
Example file:

```json
[{
  "conversation_id": "conv_001",
  "history": "Hi, I ordered a phone last week but it hasn’t arrived yet.",
  "last_message": "Can you check the status of my order?"
},
{
    "conversation_id": "conv_002",
    "messages": [
      {"sender": "user", "text": "Hello, I need help with my laptop."},
      {"sender": "agent", "text": "Sure! Can you describe the issue?"},
      {"sender": "user", "text": "It won't start and shows a blue screen."}
    ]
  }
]

```

Logging enabled by default.Creates /logs/ folder with a timestamped log file.

```bash
python -m app.main_script input/file_name.json
```


Logging disabled

```bash
python -m app.main_script input/file_name.json --no_log
```
Outputs saved to:

    outputs/predictions.json
    Example:
```json
[{
    "conversation_id": "conv_001",
    "predicted_intent": "Book Appointment",
    "rationale": "The user mentioned 'visit' in: \"hi, im looking for a 2bhk in dubai\nagent: great! any specific area in mind?\n ...\""
  },
  {
    "conversation_id": "conv_002",
    "predicted_intent": "Support Request",
    "rationale": "The user mentioned 'issue' in: \"hello, i need help with my laptop.\nagent: sure! can you describe the issue? i...\""
  },
]

```
    outputs/predictions.csv
    Example:
    conversation_id,predicted_intent,rationale
    conv_001,Book Appointment,"The user mentioned 'visit' in: ""hi, im looking for a 2bhk in dubai
    agent: great! any specific area in mind?
    ..."""
    conv_002,Support Request,"The user mentioned 'issue' in: ""hello, i need help with my laptop.
    agent: sure! can you describe the issue? i..."""
    conv_003,Follow-Up,Conversation pattern and context indicate Follow-Up


### FastAPI 

1. Start the FastAPI server:
```bash
uvicorn app.fastapi_app:app --reload
```
2. Start Streamlit client:
```bash
streamlit run frontend/client.py
```

3. Access the API documentation at http://localhost:8000/docs

### Using Docker (Not Recommended due to large size)

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
    "messages": [
      {"sender": "user", "text": "Hello, I need help with my laptop."},
      {"sender": "agent", "text": "Sure! Can you describe the issue?"},
      {"sender": "user", "text": "It won't start and shows a blue screen."}
    ]
  }
]

```

### Example Response
zip file containing JSON and CSV files of output

## Configuration

All dynamic configurations for the project are centralized in the Settings class.
This ensures that the models, allowed intents, and classification parameters can be easily updated without modifying other parts of the codebase.

Configurable Parameters

- "MODELS": List of HuggingFace models used for intent classification. Use only ungated sequential classification models from hugging face to use without change in codebase.(default: ["roberta-large-mnli", "microsoft/deberta-base"])

- "ALLOWED_INTENTS": List of possible intent categories for classification (default: ["Book Appointment", "Product Inquiry", "Pricing Negotiation", "Support Request", "Follow-Up"])

- "INTENT_KEYWORDS": Mapping of intents to relevant keywords for fallback keyword-based matching

- "MAX_HISTORY_TURNS": Maximum number of previous conversation turns to consider as context (default: 5)

## 🧠 Model Choice Explanation

For intent classification, this project uses a **combination of two transformer models — RoBERTa (`roberta-large-mnli`)** and **DeBERTa (`microsoft/deberta-base`)** — blended together using **soft voting** to get more stable and accurate predictions.

### Why RoBERTa?
RoBERTa is a stronger, fine-tuned version of BERT that performs really well on understanding sentence meaning and relationships (NLI tasks).  
It’s great at picking up **context and tone** from conversation history, which makes it suitable for identifying user intent even when the message is short or indirect.

### Why DeBERTa?
DeBERTa is a newer model that improves how attention is handled in transformers. It understands the **structure and semantics** of sentences better by separating word meaning from position.  
This helps it catch details that other models might miss, like word order or emphasis, which are often important in chat-based messages.

### Why Use Both (Soft Voting)?
Both models have different strengths — RoBERTa is more context-aware, while DeBERTa is better at fine-grained understanding.  
By averaging their confidence scores (soft voting), we combine their advantages:
- Smaller models produce output faster than large models like Gemma
- Reduces random misclassifications from one model  
- Gives smoother, more confident predictions  
- Improves overall reliability across different types of conversations  

### In Simple Terms
| Model | What It’s Good At | Why We Use It |
|--------|-------------------|---------------|
| **RoBERTa-large-MNLI** | Understands sentence meaning and intent context | Strong at general conversation understanding |
| **DeBERTa-base** | Captures deeper structure and word relationships | Helps refine intent decisions |
| **Soft Voting** | Averages both model predictions | Gives balanced, stable final results |

## ⚠️ Limitations & Edge Cases

While this intent classification system is designed to be robust, there are some scenarios where predictions may be less accurate:

### 1. Ambiguous Messages
- Messages that could fit multiple intents (e.g., "Can we discuss the price and schedule a meeting?")  
- The classifier may pick only one intent, potentially missing secondary ones.

### 2. Very Short Inputs
- Single-word messages or vague phrases (e.g., "Help", "Update")  
- Models may rely heavily on conversation history to infer intent, which can reduce confidence.

### 3. Sarcasm or Figurative Language
- Expressions like "Yeah, right, I totally want to book" may be misclassified due to literal interpretation.

### 4. Domain-Specific Terms
- Specialized words, abbreviations, or local phrases not in pretraining data  
- Example: "BHK" in property queries might require keyword mapping for correct intent.

### 5. Long Conversation Histories
- Extremely long histories can exceed the model’s maximum token limit (512 tokens for RoBERTa)  
- Only the latest turns are considered (`MAX_HISTORY_TURNS`), so very old context may be ignored.

### 6. Multi-Intent Messages
- Users sending multiple requests in one message can confuse the model  
- Example: "I want to book an appointment and check the pricing" may only return one intent.

### 7. Out-of-Distribution Text
- Messages that are very different from training / pretraining data (slang, code snippets, emojis)  
- May lead to low confidence or fallback to "Unknown" intent.






