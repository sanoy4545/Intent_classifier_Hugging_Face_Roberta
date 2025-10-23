from typing import List
from pydantic import BaseModel, Field

#input Request schemas
class Message(BaseModel):
    sender: str
    text: str

class Conversation(BaseModel):
    conversation_id: str
    messages: List[Message]

class BatchClassificationRequest(BaseModel):
    conversations: List[Conversation]



#output Response schemas
class ClassificationResponse(BaseModel):
    conversation_id: str = Field(..., description="Unique identifier for the conversation")
    predicted_intent: str = Field(..., description="Predicted intent or category of the conversation")
    rationale: str = Field(..., description="Reasoning or explanation behind the predicted intent")

class BatchClassificationResponse(BaseModel):
    results: List[ClassificationResponse] = Field(..., description="List of classification results for each conversation")
