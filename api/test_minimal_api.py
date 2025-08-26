#!/usr/bin/env python3
"""
Minimal test API to debug hanging issue
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import logging

# Basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Test Spiritual API",
    version="1.0.0",
    description="Minimal API for debugging"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    model: str = "gpt-4.1"

class QueryResponse(BaseModel):
    status: str
    answer: str
    model: str = "gpt-4.1"
    processing_time: float = 0.0

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "title": "Test Spiritual API",
        "status": "running",
        "message": "API is working"
    }

@app.get("/models")
async def get_models():
    """Get available models"""
    return {
        "models": ["gpt-4.1", "gpt-5", "gpt-4o", "o3-mini", "claude-3-opus"],
        "default": "gpt-4.1"
    }

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Test endpoint that doesn't use AnswerGenerator"""
    start_time = time.time()
    
    try:
        # Simple mock response
        answer = f"This is a test response for your question: '{request.question}' using model {request.model}"
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            status="success",
            answer=answer,
            model=request.model,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        processing_time = time.time() - start_time
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=65200)
