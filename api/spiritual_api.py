"""
FastAPI service for Spiritual Document QA system with multiple LLM endpoints.
"""
import os
import json
import time
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Depends, Request, Form, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Import answer generator and document retriever
import sys
import logging
import uuid
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from answer_generator import AnswerGenerator
from enhanced_document_retriever import EnhancedDocumentRetriever
from utils.state_manager import state_manager, validate_shared_state

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("spiritual_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Spiritual Document QA API",
    description="API for querying spiritual documents using various LLM models",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Shared configuration
# Use absolute path to the vector store containing the spiritual documents
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vector_store"))
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-4.1")

# Feature flags
ENABLE_DUAL_SOURCE = os.getenv("ENABLE_DUAL_SOURCE", "true").lower() == "true"
ENABLE_CONVERSATION_CONTEXT = os.getenv("ENABLE_CONVERSATION_CONTEXT", "false").lower() == "true"


# Define request and response models
class QueryRequest(BaseModel):
    query: str
    model: str = DEFAULT_LLM_MODEL
    k: int = 5
    query_type: Optional[str] = None
    use_mmr: bool = True


class QueryResponse(BaseModel):
    status: str
    answer: str
    chunks_used: Optional[int] = None
    model: str
    query_type: Optional[str] = None
    error: Optional[str] = None
    processing_time: float


class ConversationHistoryItem(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None
    verses: Optional[List[Dict[str, Any]]] = None


class AskRequest(BaseModel):
    question: str
    model: str = DEFAULT_LLM_MODEL
    max_context_docs: int = 5
    reading_style: str = 'balanced'
    use_mmr: bool = True
    diversity: float = 0.3
    k: int = 10
    conversation_history: Optional[List[ConversationHistoryItem]] = None
    conversation_id: Optional[str] = None
    reasoning_effort: Optional[str] = None  # For o3-mini: "low", "medium", "high"
    rag_technique: str = 'stuff'  # "stuff", "refine", "map_reduce", "map_rerank", "selective"


class QuestionResponse(BaseModel):
    status: str
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None
    verses: Optional[List[Dict[str, Any]]] = None
    model: str
    error: Optional[str] = None
    processing_time: float
    conversation_id: Optional[str] = None


class AdminConfigRequest(BaseModel):
    default_model: str
    temperature: float = 0.0
    max_tokens: int = 1024


# LLM Model configuration
LLM_MODELS = {
    "gpt-5": {
        "name": "GPT-5",
        "provider": "openai",
        "model_id": "gpt-5",
        "max_completion_tokens": 3000,
        "supports_temperature": False,
        "supports_streaming": True,
        "is_reasoning_model": False
    },
    "gpt-4.1": {
        "name": "GPT-4.1",
        "provider": "openai",
        "model_id": "gpt-4.1",
        "max_tokens": 8096,
        "temperature": 0.0
    },
    "o3-mini": {
        "name": "o3 mini (Reasoning)",
        "provider": "openai",
        "model_id": "o3-mini",
        "max_completion_tokens": 4096,
        "reasoning_efforts": ["low", "medium", "high"],
        "default_reasoning_effort": "medium",
        "supports_temperature": False,
        "supports_streaming": True,
        "is_reasoning_model": True
    },
    "gpt-4o": {
        "name": "GPT-4o",
        "provider": "openai",
        "model_id": "gpt-4o",
        "max_tokens": 4096,
        "temperature": 0.0,
        "supports_temperature": True,
        "supports_streaming": True,
        "is_reasoning_model": False
    },
    "grok-3-mini-beta": {
        "name": "Grok 3 mini beta",
        "provider": "anthropic",  # placeholder, would need proper client
        "model_id": "grok-3-mini-beta",
        "max_tokens": 4096,
        "temperature": 0.0
    },
    "deepseek-reasoner": {
        "name": "Deepseek reasoner",
        "provider": "deepseek",  # placeholder, would need proper client
        "model_id": "deepseek-reasoner",
        "max_tokens": 4096,
        "temperature": 0.0
    },
    "gemini-pro-2.5-flash": {
        "name": "Gemini pro 2.5 flash",
        "provider": "google",  # placeholder, would need proper client
        "model_id": "gemini-pro-2.5-flash",
        "max_tokens": 4096,
        "temperature": 0.0
    },
    "claude-3.7-sonnet-thinking": {
        "name": "Claude 3.7 Sonnet thinking",
        "provider": "anthropic",  # placeholder, would need proper client
        "model_id": "claude-3.7-sonnet-thinking",
        "max_tokens": 4096,
        "temperature": 0.0
    }
}

# Initialize answer generators for each model
answer_generators = {}


def get_answer_generator(model: str) -> AnswerGenerator:
    """
    Get or initialize an answer generator for the specified model.
    
    Args:
        model: LLM model ID
    
    Returns:
        AnswerGenerator instance
    """
    model_id = model.lower()
    
    # If model not in available models, use default
    if model_id not in LLM_MODELS:
        model_id = DEFAULT_LLM_MODEL
    
    # If generator not initialized, create it
    if model_id not in answer_generators:
        model_config = LLM_MODELS[model_id]
        
        # Initialize answer generator with model configuration
        # Note: In practice, we would need different client initializations for different providers
        
        # Handle different model parameter structures
        if model_config.get("is_reasoning_model", False):
            # o3-mini reasoning model
            answer_generators[model_id] = AnswerGenerator(
                vector_store_dir=VECTOR_STORE_DIR,
                llm_model=model_config["model_id"],
                temperature=0.0,  # Not used by o3-mini, but required by constructor
                max_tokens=model_config.get("max_completion_tokens", 4096),
                reasoning_effort=model_config.get("default_reasoning_effort", "medium"),
                enable_dual_source=ENABLE_DUAL_SOURCE
            )
        elif "max_completion_tokens" in model_config:
            # Models that use max_completion_tokens (like GPT-5)
            temp_value = model_config.get("temperature", 0.0) if model_config.get("supports_temperature", True) else 0.0
            answer_generators[model_id] = AnswerGenerator(
                vector_store_dir=VECTOR_STORE_DIR,
                llm_model=model_config["model_id"],
                temperature=temp_value,
                max_tokens=model_config["max_completion_tokens"],
                enable_dual_source=ENABLE_DUAL_SOURCE
            )
        else:
            # Standard GPT models with max_tokens
            answer_generators[model_id] = AnswerGenerator(
                vector_store_dir=VECTOR_STORE_DIR,
                llm_model=model_config["model_id"],
                temperature=model_config["temperature"],
                max_tokens=model_config["max_tokens"],
                enable_dual_source=ENABLE_DUAL_SOURCE
            )
    
    return answer_generators[model_id]


@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "title": "Spiritual Document QA API",
        "version": app.version,
        "description": "Welcome! This API provides access to a spiritual document Q&A system.",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }


@app.get("/health")
async def health_check():
    """Endpoint to check if the API is running."""
    return {"status": "ok"}


@app.get("/random-wisdom")
async def random_wisdom():
    """Endpoint to get a random piece of wisdom."""
    # In the future, this could be fetched from the vector store
    wisdom_list = [
        "The journey of a thousand miles begins with a single step.",
        "Patience is a virtue, and the key to inner peace.",
        "True wisdom is knowing you know nothing.",
        "The only way to do great work is to love what you do."
    ]
    import random
    return {"wisdom": random.choice(wisdom_list)}
@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Submit a query to the spiritual document QA system.
    
    Args:
        request: QueryRequest object with query and model preferences
    
    Returns:
        QueryResponse with generated answer
    """
    start_time = time.time()
    
    try:
        # Get answer generator for requested model
        generator = get_answer_generator(request.model)
        
        # Optimize context size based on model
        k_value = request.k
        if request.model.lower() == "gpt-5":
            # Reduce context for GPT-5 to improve performance
            k_value = min(request.k, 3)
        
        # Generate answer
        result = generator.generate_answer(
            query=request.query,
            k=k_value,
            use_mmr=request.use_mmr
        )
        
        # Log query
        logger.info(f"Query: '{request.query}' processed with model {request.model}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Return response
        if result.get("status") == "success":
            return QueryResponse(
                status="success",
                answer=result.get("answer"),
                chunks_used=result.get("chunks_used"),
                model=result.get("model"),
                query_type=result.get("prompt_type"),
                processing_time=processing_time
            )
        else:
            return QueryResponse(
                status="error",
                answer="Sorry, I couldn't generate an answer for your question.",
                error=result.get("error"),
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


@app.get("/models")
async def get_models():
    """
    Get available LLM models.
    
    Returns:
        Dict with available models and their configuration
    """
    return {
        "default_model": DEFAULT_LLM_MODEL,
        "models": LLM_MODELS
    }


@app.get("/admin/config")
async def get_admin_config():
    """
    Get admin configuration.
    
    Returns:
        Dict with current configuration
    """
    return {
        "default_model": DEFAULT_LLM_MODEL,
        "vector_store_dir": VECTOR_STORE_DIR,
        "model_settings": LLM_MODELS.get(DEFAULT_LLM_MODEL, {})
    }


@app.post("/admin/config")
async def update_admin_config(config: AdminConfigRequest):
    """
    Update admin configuration.
    
    Args:
        config: AdminConfigRequest with new configuration
    
    Returns:
        Dict with updated configuration
    """
    global DEFAULT_LLM_MODEL
    
    try:
        # Update default model if valid
        if config.default_model in LLM_MODELS:
            DEFAULT_LLM_MODEL = config.default_model
            
            # Update model configuration
            LLM_MODELS[DEFAULT_LLM_MODEL]["temperature"] = config.temperature
            LLM_MODELS[DEFAULT_LLM_MODEL]["max_tokens"] = config.max_tokens
            
            # Reinitialize answer generator for this model
            if DEFAULT_LLM_MODEL in answer_generators:
                del answer_generators[DEFAULT_LLM_MODEL]
            
            logger.info(f"Updated default model to {DEFAULT_LLM_MODEL}")
            
            return {
                "status": "success",
                "message": f"Default model updated to {DEFAULT_LLM_MODEL}",
                "config": {
                    "default_model": DEFAULT_LLM_MODEL,
                    "model_settings": LLM_MODELS.get(DEFAULT_LLM_MODEL, {})
                }
            }
        else:
            return {
                "status": "error",
                "message": f"Invalid model: {config.default_model}"
            }
            
    except Exception as e:
        logger.error(f"Error updating admin config: {str(e)}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Error updating admin config: {str(e)}"
        )


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: AskRequest):
    """
    Submit a question to the spiritual Q&A system with optional conversation history.
    This endpoint supports multi-turn conversations by retaining context.
    
    Args:
        request: AskRequest object with question, model preferences, and optional conversation history
    
    Returns:
        QuestionResponse with generated answer, sources, and conversation ID
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]  # Short unique ID for this request
    
    logger.info(f"Processing request {request_id} - Question: '{request.question[:50]}...'")
    
    try:
        # Use isolated state context to prevent corruption
        with state_manager.isolated_request_context(request_id) as request_state:
            # Check global state health before processing
            if not state_manager.check_global_health():
                logger.error(f"Request {request_id}: Global state is unhealthy, rejecting request")
                raise HTTPException(
                    status_code=503, 
                    detail="Service temporarily unavailable due to state issues"
                )
            
            # Get answer generator for requested model
            generator = get_answer_generator(request.model)
            logger.info(f"Request {request_id}: Generator retrieved: llm_model={generator.llm_model}")
            
            # Process reading style parameters
            use_mmr = request.use_mmr
            diversity = request.diversity
            k = request.k
            
            # Apply reading style overrides
            if request.reading_style == 'deep':
                use_mmr = False  # Use pure similarity search for Deep style
                logger.info(f"Request {request_id}: Using 'Deep' reading style: pure similarity search (use_mmr=False)")
            elif request.reading_style == 'balanced':
                use_mmr = True  # Use MMR for Balanced style
                diversity = 0.5  # 50/50 balance between relevance and diversity
                logger.info(f"Request {request_id}: Using 'Balanced' reading style: MMR with lambda=0.5 (use_mmr=True, diversity=0.5)")
            elif request.reading_style == 'practical':
                use_mmr = True  # Use MMR for Practical style
                diversity = 0.8  # Emphasize diversity for varied practical guidance
                logger.info(f"Request {request_id}: Using 'Practical' reading style: MMR with lambda=0.8 (use_mmr=True, diversity=0.8)")
            else:
                logger.info(f"Request {request_id}: Using '{request.reading_style}' reading style with use_mmr={use_mmr}, diversity={diversity}")
            
            # Process conversation history if available
            conversation_context = ""
            if request.conversation_history and len(request.conversation_history) > 0:
                # Format conversation history for context
                history_text = []
                # Use the last 5 exchanges at most to avoid context length issues
                for item in request.conversation_history[-10:]:
                    role_prefix = "User" if item.role == "user" else "Assistant"
                    history_text.append(f"{role_prefix}: {item.content}")
                
                # Create conversation context string
                conversation_context = "\n\n".join(history_text)
                logger.info(f"Request {request_id}: Using conversation context with {len(history_text)} previous messages")
            
            # Generate answer with conversation context if available
            if conversation_context:
                # Combine the current question with conversation history
                logger.info(f"Request {request_id}: Generating answer with conversation context")
                # Use IDENTICAL parameters as working test scripts
                result = generator.generate_answer_with_context(
                    query=request.question,
                    conversation_context=conversation_context,
                    k=k,
                    use_mmr=use_mmr,
                    reasoning_effort=request.reasoning_effort
                )
            else:
                # Just use the current question without context
                logger.info(f"Request {request_id}: Generating answer without conversation context")
                result = generator.generate_answer(
                    query=request.question,
                    k=k,
                    reasoning_effort=request.reasoning_effort  # Pass user-provided reasoning effort
                )
            
            # Log query completion
            logger.info(f"Request {request_id}: Question processed with model {request.model}")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Generate or use provided conversation ID
            conversation_id = request.conversation_id or f"conv_{int(time.time())}_{hash(request.question)%10000}"
            
            # Mark request as successful in state
            request_state['success'] = True
            logger.info(f"Request {request_id}: Completed successfully in {processing_time:.2f}s")
            
            # Return response
            if result.get("status") == "success":
                # Debug logging for model field issue
                logger.info(f"Request {request_id}: Result keys: {list(result.keys())}")
                logger.info(f"Request {request_id}: Model from result: {result.get('model')}")
                logger.info(f"Request {request_id}: Has conversation context: {bool(conversation_context)}")
                
                return QuestionResponse(
                    status="success",
                    answer=result.get("answer"),
                    sources=result.get("sources"),
                    verses=result.get("verses"),
                    model=result.get("model"),
                    processing_time=processing_time,
                    conversation_id=conversation_id
                )
            else:
                logger.error(f"Request {request_id}: Result status was not success: {result.get('status')}")
                return QuestionResponse(
                    status="error",
                    answer="I apologize, but I couldn't generate an answer for your spiritual question.",
                    model=request.model,
                    error=result.get("error"),
                    processing_time=processing_time,
                    conversation_id=conversation_id
                )
                
    except Exception as e:
        logger.error(f"Request {request_id}: Exception occurred: {str(e)}")
        processing_time = time.time() - start_time
        
        # State isolation will handle cleanup automatically via context manager
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )


@app.get("/admin/health")
async def check_health():
    """
    Check the health of the backend including shared state status.
    
    Returns:
        Health status with detailed diagnostics
    """
    try:
        health_status = validate_shared_state()
        
        # Add additional health checks
        health_status.update({
            "api_status": "healthy",
            "vector_store": "operational",
            "llm_models_available": len(LLM_MODELS),
            "active_generators": len(answer_generators)
        })
        
        logger.info(f"Health check performed: {health_status}")
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "api_status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }


@app.get("/admin/logs")
async def get_admin_logs(n: int = 100):
    """
    Get system logs.
    
    Args:
        n: Number of log entries to return
    
    Returns:
        Dict with log entries
    """
    try:
        with open("spiritual_api.log", "r") as f:
            logs = f.readlines()[-n:]
            
        return {
            "status": "success",
            "logs": logs
        }
    except Exception as e:
        logger.error(f"Error reading logs: {str(e)}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Error reading logs: {str(e)}"
        )


if __name__ == "__main__":
    # Run the FastAPI app
    uvicorn.run(
        "spiritual_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
