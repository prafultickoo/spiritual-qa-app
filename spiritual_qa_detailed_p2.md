# Spiritual Q&A System: Detailed Backend Flow (Part 2)

This document continues the granular breakdown of the backend flow for the Spiritual Q&A system. Part 2 covers API request processing, document retrieval, and the beginning of LLM answer generation.

## Table of Contents
1. [API Request Reception and Validation](#1-api-request-reception-and-validation)
2. [Answer Generator Initialization](#2-answer-generator-initialization)
3. [Document Retrieval Preparation](#3-document-retrieval-preparation)
4. [Query Embedding Generation](#4-query-embedding-generation)

## 1. API Request Reception and Validation

### Detailed Explanation
When the frontend's HTTP request reaches the backend server, FastAPI processes it through several steps:

1. The incoming HTTP request is received by the web server (uvicorn/FastAPI).

2. FastAPI's routing system directs the request to the appropriate endpoint based on the URL path (`/ask` or `/query`).

3. The JSON payload is automatically extracted from the request body.

4. The extracted JSON is automatically validated against the `QueryRequest` Pydantic model:
   - Required fields are checked for presence
   - Data types are validated (strings for query and model, integer for k, etc.)
   - Default values are applied where specified but not provided

5. If validation fails, FastAPI automatically returns a 422 Unprocessable Entity response with validation error details.

6. If validation passes, a `QueryRequest` instance is created with the validated data.

7. The request timestamp is recorded to measure processing time.

8. The system logs the incoming request with query text and model information.

### Implementing Code
```python
# Definition of the request model used for validation
class QueryRequest(BaseModel):
    query: str
    model: str = DEFAULT_LLM_MODEL
    k: int = 5
    query_type: Optional[str] = None
    use_mmr: bool = True
    diversity: float = 0.7
    max_context_docs: int = 5

# FastAPI endpoint definition
@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Submit a query to the spiritual document QA system.
    
    Args:
        request: QueryRequest object with query and model preferences
    
    Returns:
        QueryResponse with generated answer
    """
    # Record start time for performance measurement
    start_time = time.time()
    
    try:
        # Log the incoming request
        logger.info(f"Query: '{request.query}' processed with model {request.model}")
        
        # Continue processing... (next step)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        processing_time = time.time() - start_time
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )
```

## 2. Answer Generator Initialization

### Detailed Explanation
After validating the request, the system needs to initialize or retrieve a cached `AnswerGenerator` instance for the requested LLM model:

1. The system calls `get_answer_generator(request.model)` to obtain an `AnswerGenerator` instance.

2. Inside `get_answer_generator`, several sub-steps occur:
   - The function checks if an `AnswerGenerator` for the requested model already exists in the `GENERATORS` cache dictionary.
   - If found in the cache, that instance is returned immediately, avoiding redundant initialization.
   - If not found, the system verifies that the requested model exists in the `LLM_MODELS` configuration.
   - If the model doesn't exist, it falls back to the default LLM model to prevent failure.

3. When creating a new `AnswerGenerator` instance, the system:
   - Retrieves model-specific configuration (temperature, max_tokens) from `LLM_MODELS`.
   - Initializes the generator with the vector store directory, embedding model, and model-specific parameters.
   - Caches the new generator instance in the `GENERATORS` dictionary for future reuse.

4. Throughout this process, logging captures important actions:
   - When retrieving from cache: "Using cached generator for model..."
   - When creating new: "Initializing new generator for model..."
   - If model not found: "Unknown model, falling back to default..."

5. The final result is a properly configured `AnswerGenerator` instance ready to process the user query.

### Implementing Code
```python
# Global cache of AnswerGenerator instances
GENERATORS = {}

def get_answer_generator(model: str):
    """
    Get or initialize an answer generator for the specified model.
    
    Args:
        model: LLM model ID
    
    Returns:
        AnswerGenerator instance
    """
    global GENERATORS
    
    # Check if we already have a generator for this model
    if model in GENERATORS:
        logger.info(f"Using cached generator for model: {model}")
        return GENERATORS[model]
    
    # Check if model exists in configuration
    if model not in LLM_MODELS:
        logger.warning(f"Unknown model: {model}, falling back to {DEFAULT_LLM_MODEL}")
        model = DEFAULT_LLM_MODEL
    
    # Get model configuration
    model_config = LLM_MODELS[model]
    
    # Initialize new generator
    logger.info(f"Initializing new generator for model: {model}")
    generator = AnswerGenerator(
        vector_store_dir=VECTOR_STORE_DIR,
        embedding_model="openai",
        llm_model=model_config["model_id"],
        temperature=model_config.get("temperature", 0.0),
        max_tokens=model_config.get("max_tokens", 1024)
    )
    
    # Cache the generator for future use
    GENERATORS[model] = generator
    
    return generator
```

## 3. Document Retrieval Preparation

### Detailed Explanation
With an `AnswerGenerator` instance ready, the system begins the process of retrieving relevant document chunks:

1. The API endpoint calls `generator.generate_answer()` with parameters from the request:
   - query: The user's question text
   - k: The number of chunks to retrieve
   - query_type: Optional query type for prompt selection

2. Inside `generate_answer`, the system first prepares to retrieve relevant chunks:
   - It validates that the query is non-empty
   - It prepares retrieval parameters (k, use_mmr, diversity)
   - It logs the retrieval request with these parameters

3. For actual chunk retrieval, it calls `self.retrieve_relevant_chunks()`, which:
   - Forwards the call to the `DocumentRetriever` instance stored in `self.retriever`
   - Passes along the query and retrieval parameters (k, use_mmr)

4. The `DocumentRetriever` then begins preparing the retrieval process:
   - It validates that the vector store has been properly loaded
   - It accesses the query processor to handle embedding and retrieval operations
   - It logs that it's about to retrieve chunks for the given query

5. All these steps occur before actually embedding the query or searching the vector database.

### Implementing Code
```python
# In answer_generator.py
def generate_answer(self, 
                   query: str, 
                   k: int = 5,
                   query_type: str = None) -> Dict[str, Any]:
    """
    Generate an answer for a user query.
    
    Args:
        query: User query text
        k: Number of chunks to retrieve
        query_type: Type of query for prompt selection
        
    Returns:
        Dict with generated answer and metadata
    """
    try:
        # Validate query
        if not query or not query.strip():
            return {
                "status": "error",
                "error": "Query cannot be empty",
                "model": self.llm_model
            }
        
        # Log the retrieval request
        logger.info(f"Retrieving chunks for query: '{query}' with k={k}")
        
        # Retrieve relevant chunks
        retrieval_result = self.retrieve_relevant_chunks(query, k)
        
        # Continue with chunk processing... (in next section)
        
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "model": self.llm_model
        }

# Chunk retrieval method in AnswerGenerator
def retrieve_relevant_chunks(self, 
                            query: str, 
                            k: int = 5,
                            use_mmr: bool = True) -> Dict[str, Any]:
    """
    Retrieve relevant chunks for a query.
    
    Args:
        query: User query text
        k: Number of chunks to retrieve
        use_mmr: Whether to use MMR for diverse retrieval
        
    Returns:
        Dict containing retrieved chunks
    """
    # Forward to the DocumentRetriever instance
    return self.retriever.retrieve_chunks(
        query=query,
        k=k,
        use_mmr=use_mmr
    )
```

## 4. Query Embedding Generation

### Detailed Explanation
Before the system can search for relevant document chunks, it must convert the user's query into a vector embedding:

1. Inside `DocumentRetriever.retrieve_chunks()`, the system follows these detailed steps:
   - It delegates the embedding process to `self.query_processor.process_query()`
   - It prepares variables to handle the result of this embedding process

2. Inside `QueryProcessor.process_query()`:
   - The system validates that the query string is not empty
   - It checks that the vector store has successfully loaded
   - It prepares to create an embedding of the query text

3. The embedding creation process involves:
   - Using the configured embedding model (typically OpenAI's text-embedding-ada-002)
   - Converting the query string into a high-dimensional vector
   - This process performs API calls to the embedding provider (usually OpenAI)
   - The result is a floating-point vector typically with 1536 dimensions (for OpenAI)

4. Error handling at each step ensures:
   - If the query is empty, a meaningful error is returned
   - If the vector store failed to load, this is reported
   - If the embedding API call fails, the error is caught and logged

5. The successful result of `process_query` is a dictionary with:
   - status: "success" or "error"
   - embedding: The vector representation of the query (if successful)
   - error: Error message (if failed)

### Implementing Code
```python
# Inside DocumentRetriever
def retrieve_chunks(self, 
                   query: str, 
                   k: int = 5,
                   use_mmr: bool = True,
                   diversity: float = 0.7) -> Dict[str, Any]:
    """
    Retrieve relevant chunks for a user query.
    
    Args:
        query: User query text
        k: Number of chunks to retrieve
        use_mmr: Whether to use Maximum Marginal Relevance for diverse retrieval
        diversity: MMR diversity parameter (0 = maximal diversity, 1 = maximal relevance)
        
    Returns:
        Dict with retrieved chunks and metadata
    """
    try:
        # Process the query (generate embedding)
        query_result = self.query_processor.process_query(query)
        
        if query_result.get("status") != "success":
            logger.error(f"Failed to process query: {query_result.get('error', 'Unknown error')}")
            return {
                "status": "error",
                "error": query_result.get("error", "Failed to process query"),
                "chunks": []
            }
        
        # Continue with retrieval using the embedding... (next section)
    
    except Exception as e:
        logger.error(f"Error retrieving chunks: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "chunks": []
        }

# Inside QueryProcessor
def process_query(self, query_text: str) -> Dict[str, Any]:
    """
    Process a query by creating embedding.
    
    Args:
        query_text: User query text
        
    Returns:
        Dict with embedding or error
    """
    # Validate query
    if not query_text or not query_text.strip():
        return {
            "status": "error",
            "error": "Query text cannot be empty"
        }
    
    # Check if vector store is loaded
    if self.vector_store is None:
        return {
            "status": "error",
            "error": "Vector store not loaded"
        }
    
    try:
        # Generate embedding for the query using the embedding model
        # This is a call to the embedding provider's API (e.g., OpenAI)
        query_embedding = self.embedding_model.embed_query(query_text)
        
        return {
            "status": "success",
            "embedding": query_embedding
        }
    except Exception as e:
        logger.error(f"Error creating query embedding: {str(e)}")
        return {
            "status": "error",
            "error": f"Error creating query embedding: {str(e)}"
        }
```

---

**[Continue to Part 3 for Vector Similarity Search, Context Assembly, and LLM Prompt Construction]**
