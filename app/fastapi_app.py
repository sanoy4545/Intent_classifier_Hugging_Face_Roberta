from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import logging
import json

from fastapi.responses import FileResponse

from app.core.model import classifier
from app.core.utils import setup_logging
from app.services.classifier_service import ClassifierService
from app.config import settings

# Initialize FastAPI app
app = FastAPI(
    title="Multi-turn Intent Classifier",
    description="API for classifying intents in multi-turn conversations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize classifier service , single service no need for container
classifier_service = ClassifierService()


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    try:
        classifier.load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


@app.post("/classify")
async def classify_intent(file: UploadFile = File(...)):
    """
    Accept a JSON file containing multiple conversations
    and return classified intents for each.
    """
    try:
        # Validate file
        if not file.filename.endswith(".json"):
            raise HTTPException(status_code=400, detail="Please upload a valid JSON file")

        # Read file
        content = await file.read()
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format")

        # Delegate all logic to the service layer
        result_path = classifier_service.classify_conversations(data)
        return FileResponse(result_path, media_type='application/zip', filename='classification_results.zip')

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error classifying conversations: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
