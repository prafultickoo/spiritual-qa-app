"""
Main FastAPI backend for Spiritual Q&A System.
Provides endpoints for all required LLM models and spiritual document querying.
"""

import os
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.document_retriever import get_document_retriever, search_spiritual_documents
from utils.llm_integrations import generate_spiritual_answer

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Spiritual Q&A API",
    description="AI-powered spiritual guidance system with access to ancient wisdom texts",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class SpiritualQuery(BaseModel):
    """Request model for spiritual questions."""
    question: str = Field(..., description="The spiritual question to ask", min_length=1, max_length=500)
    model: str = Field(default="gpt-5", description="LLM model to use for response")
    max_context_docs: int = Field(default=5, description="Maximum number of context documents to retrieve")

class SpiritualResponse(BaseModel):
    """Response model for spiritual answers."""
    query: str
    response: str
    model_used: str
    sources: List[str]
    guidance_type: str
    suggested_practices: List[str]
    related_teachings: List[str]
    timestamp: str
    processing_time_ms: int

class DocumentSearchQuery(BaseModel):
    """Request model for document search."""
    query: str = Field(..., description="Search query for spiritual documents")
    num_docs: int = Field(default=5, description="Number of documents to return")
    source_filter: Optional[str] = Field(None, description="Filter by specific source")

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    version: str
    vector_store_status: str
    available_models: List[str]

# Available LLM models as per user requirements
AVAILABLE_MODELS = [
    "gpt-4o",
    "gpt-4.1", 
    "o3-mini",
    "grok-3-mini-beta",
    "deepseek-reasoner",
    "gemini-pro-2.5-flash",
    "claude-3.7-sonnet-thinking"
]

DEFAULT_MODEL = "gpt-4.1"

@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "🕉️ Welcome to the Spiritual Q&A API",
        "description": "AI-powered spiritual guidance system with access to ancient wisdom texts",
        "version": "1.0.0",
        "endpoints": {
            "ask": "/ask - Ask a spiritual question",
            "search": "/search - Search spiritual documents", 
            "health": "/health - Health check",
            "models": "/models - List available models",
            "docs": "/docs - API documentation"
        },
        "wisdom": "The divine wisdom is within you. Ask, and you shall receive guidance."
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check vector store status
        retriever = get_document_retriever()
        test_docs = search_spiritual_documents("test", num_docs=1)
        vector_store_status = "healthy" if test_docs else "warning"
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow().isoformat(),
            version="1.0.0",
            vector_store_status=vector_store_status,
            available_models=AVAILABLE_MODELS
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow().isoformat(),
            version="1.0.0",
            vector_store_status="error",
            available_models=[]
        )

@app.get("/models")
async def list_models():
    """List available LLM models."""
    return {
        "available_models": AVAILABLE_MODELS,
        "default_model": DEFAULT_MODEL,
        "model_descriptions": {
            "gpt-4o": "OpenAI GPT-4o - Advanced reasoning and spiritual wisdom",
            "gpt-4.1": "OpenAI GPT-4.1 - Default model with excellent spiritual guidance",
            "o3-mini": "OpenAI o3-mini - Compact yet powerful spiritual insights",
            "grok-3-mini-beta": "xAI Grok 3 Mini Beta - Alternative perspective",
            "deepseek-reasoner": "DeepSeek Reasoner - Deep analytical spiritual guidance",
            "gemini-pro-2.5-flash": "Google Gemini Pro 2.5 Flash - Fast spiritual responses",
            "claude-3.7-sonnet-thinking": "Anthropic Claude 3.7 Sonnet - Thoughtful spiritual wisdom"
        }
    }

@app.post("/ask", response_model=SpiritualResponse)
async def ask_spiritual_question(query: SpiritualQuery):
    """
    Ask a spiritual question and receive AI-powered guidance.
    
    This endpoint retrieves relevant spiritual texts and generates
    personalized guidance using the specified LLM model.
    """
    start_time = datetime.utcnow()
    
    try:
        # Validate model
        if query.model not in AVAILABLE_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{query.model}' not available. Choose from: {AVAILABLE_MODELS}"
            )
        
        logger.info(f"Processing spiritual question: {query.question[:50]}... using model: {query.model}")
        
        # Generate spiritual response
        response_data = await generate_spiritual_answer(query.question, query.model)
        
        # Calculate processing time
        end_time = datetime.utcnow()
        processing_time = int((end_time - start_time).total_seconds() * 1000)
        
        # Format response
        spiritual_response = SpiritualResponse(
            query=query.question,
            response=response_data["response"],
            model_used=query.model,
            sources=response_data["sources"],
            guidance_type=response_data["guidance_type"],
            suggested_practices=response_data["suggested_practices"],
            related_teachings=response_data["related_teachings"],
            timestamp=end_time.isoformat(),
            processing_time_ms=processing_time
        )
        
        logger.info(f"Successfully processed spiritual question in {processing_time}ms")
        return spiritual_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing spiritual question: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating spiritual guidance: {str(e)}"
        )

@app.post("/search")
async def search_documents(search: DocumentSearchQuery):
    """
    Search spiritual documents by query.
    
    Retrieves relevant spiritual texts based on the search query
    without generating AI responses.
    """
    try:
        logger.info(f"Searching documents for: {search.query}")
        
        # Search documents
        if search.source_filter:
            retriever = get_document_retriever()
            documents = retriever.search_by_source(search.query, search.source_filter, search.num_docs)
        else:
            documents = search_spiritual_documents(search.query, search.num_docs)
        
        return {
            "query": search.query,
            "documents_found": len(documents),
            "documents": documents,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error searching spiritual documents: {str(e)}"
        )

@app.get("/ask/{model}")
async def ask_with_model(
    model: str = Path(..., description="LLM model to use"),
    question: str = Query(..., description="Spiritual question to ask")
):
    """
    Quick spiritual question endpoint with model specified in URL.
    
    Convenient endpoint for testing different models quickly.
    """
    query = SpiritualQuery(question=question, model=model)
    return await ask_spiritual_question(query)

@app.get("/random-wisdom")
async def get_random_wisdom():
    """
    Get a random spiritual quote or wisdom passage.
    
    Perfect for daily inspiration or meditation.
    """
    try:
        retriever = get_document_retriever()
        quote = retriever.get_random_spiritual_quote()
        
        return {
            "wisdom": quote,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "May this wisdom illuminate your spiritual path 🕉️"
        }
        
    except Exception as e:
        logger.error(f"Error getting random wisdom: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving spiritual wisdom: {str(e)}"
        )

@app.get("/stats")
async def get_api_stats():
    """
    Get API usage statistics and system information.
    """
    try:
        retriever = get_document_retriever()
        
        # Get some basic stats
        test_search = search_spiritual_documents("spiritual", num_docs=1)
        
        return {
            "system_info": {
                "api_version": "1.0.0",
                "status": "operational",
                "timestamp": datetime.utcnow().isoformat()
            },
            "vector_store": {
                "status": "healthy" if test_search else "warning",
                "documents_available": "25,808 spiritual texts"
            },
            "available_models": len(AVAILABLE_MODELS),
            "default_model": DEFAULT_MODEL,
            "wisdom_note": "Every question is a step on the spiritual journey 🙏"
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return {
            "system_info": {
                "api_version": "1.0.0",
                "status": "degraded",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
        }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "message": "The path you seek does not exist. Perhaps try /docs for guidance?",
            "wisdom": "In seeking, we sometimes find what we did not expect to find."
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "A challenge arose in processing your request. Please try again.",
            "wisdom": "Obstacles are opportunities for growth and learning."
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    print("🕉️ Starting Spiritual Q&A API Server...")
    print(f"📍 Server will be available at: http://{host}:{port}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    print(f"🧘 Available Models: {', '.join(AVAILABLE_MODELS)}")
    print(f"⭐ Default Model: {DEFAULT_MODEL}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
