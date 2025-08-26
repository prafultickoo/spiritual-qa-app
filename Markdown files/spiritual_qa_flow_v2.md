# Spiritual Q&A System – Core Runtime Flow (English ➜ Code)

This document walks through every **runtime** step that happens **while a user waits for an answer**.  Each block has:

1. **English explanation** of the functionality
2. The **exact code** that implements it

---

## 1. Capture & Send the Question (Frontend)

### English explanation
When the user presses **“Seek Wisdom”**, the browser collects the typed question, the selected LLM model and reading-style radio button, converts the reading style to concrete retrieval parameters, and POSTs a JSON body to the backend’s `/ask` endpoint.

### Code (frontend/app.js)
```javascript
// listens to <form id="question-form"> submit
async handleQuestionSubmit(e) {
  e.preventDefault();
  if (this.isLoading) return;

  const question     = document.getElementById('spiritual-question').value.trim();
  const model        = document.getElementById('llm-model').value;
  const readingStyle = document.querySelector('input[name="reading-style"]:checked').value;
  if (!question) { this.showNotification('Please enter a spiritual question','warning'); return; }

  this.isLoading = true;
  try {
    const response = await this.askSpiritualQuestion(question, model, readingStyle);
    this.displayResponse(response);      // hand off to renderer
  } finally {
    this.isLoading = false;
  }
}

// builds JSON payload and POSTs to backend
async askSpiritualQuestion(question, model = 'gpt-4.1', readingStyle = 'balanced') {
  const retrieval = this.getRetrievalParamsFromStyle(readingStyle); // {use_mmr, diversity, k}
  const resp = await fetch(`${this.apiBase}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      model,
      reading_style: readingStyle,
      use_mmr: retrieval.use_mmr,
      diversity: retrieval.diversity,
      k: retrieval.k
    })
  });
  return resp.json();
}
```

---

## 2. Receive & Validate Request (FastAPI)

### English explanation
FastAPI automatically validates the incoming JSON against the `QueryRequest` schema.  A cached `AnswerGenerator` for the selected LLM model is fetched.  The endpoint then calls `generate_answer`, times the operation, and packages the result into a `QueryResponse` object that is returned as JSON.

### Code (api/spiritual_api.py)
```python
@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    start_time = time.time()
    generator = get_answer_generator(request.model)  # cached per-model

    result = generator.generate_answer(
        query=request.query,
        k=request.k,
        query_type=request.query_type,
    )

    processing_time = time.time() - start_time
    return QueryResponse(
        status="success",
        answer=result["answer"],
        chunks_used=result["chunks_used"],
        model=result["model"],
        query_type=result["prompt_type"],
        processing_time=processing_time,
    )
```

---

## 3. Retrieve Relevant Chunks (Vector Store)

### English explanation
`DocumentRetriever` converts the user query into an embedding vector using `OpenAIEmbeddings`.  It then searches the ChromaDB collection `spiritual_texts` using either **similarity search** or **MMR** depending on `use_mmr`.  The top-k (or MMR-reranked) chunks are returned alongside metadata.

### Code (document_retriever.py)
```python
def retrieve_chunks(self, query: str, k: int = 5, use_mmr: bool = True, diversity: float = 0.7):
    # 1️⃣ embed the query
    query_result = self.query_processor.process_query(query)

    # 2️⃣ choose retrieval strategy
    if use_mmr:
        result = self.query_processor.mmr_retrieval(
            query_text=query,
            k=k,
            diversity=diversity,
        )
    else:
        result = self.query_processor.similarity_retrieval(
            query_text=query,
            k=k,
        )

    return {"status": "success", "chunks": result}
```

---

## 4. Build Prompt & Invoke LLM

### English explanation
`AnswerGenerator` formats the retrieved chunks into a single context block, selects the correct prompt template based on `query_type`, and calls the chosen LLM (OpenAI, Anthropic, etc.).  The LLM’s response is returned along with metadata.

### Code (answer_generator.py)
```python
def generate_answer(self, query: str, k: int = 5, query_type: str = None):
    # 1️⃣ Retrieve chunks
    retrieval = self.retrieve_relevant_chunks(query, k)

    # 2️⃣ Build context string
    context = format_context_from_chunks(retrieval["chunks"])

    # 3️⃣ Select prompt template and fill placeholders
    template = select_prompt_template(query_type)
    prompt   = template.format(context=context, question=query)

    # 4️⃣ Call the LLM
    response = client.chat.completions.create(
        model=self.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=self.temperature,
        max_tokens=self.max_tokens,
    )

    return {
        "status": "success",
        "answer": response.choices[0].message.content,
        "chunks_used": len(retrieval["chunks"]),
        "model": self.llm_model,
        "prompt_type": query_type or "default",
    }
```

---

## 5. Package & Return the Answer (FastAPI → Frontend)

### English explanation
`QueryResponse` is automatically serialised to JSON.  Fields include the final answer text, the model that produced it, how long processing took, and how many document chunks were injected into the prompt.

### Code (api/spiritual_api.py – same endpoint)
```python
return QueryResponse(
    status="success",
    answer=result["answer"],
    chunks_used=result["chunks_used"],
    model=result["model"],
    query_type=result["prompt_type"],
    processing_time=processing_time,
)
```

---

## 6. Render Answer in Browser

### English explanation
The frontend hides the loading overlay, reveals the response panel, inserts the answer text, and lists the cited source documents so users can verify authenticity.

### Code (frontend/app.js)
```javascript
// very small extract
displayResponse(responseData) {
  document.getElementById('guidance-response').textContent = responseData.response;
  const list = document.getElementById('sources-list');
  list.innerHTML = '';
  responseData.sources.forEach(src => {
    const li = document.createElement('li');
    li.textContent = src;
    list.appendChild(li);
  });
  document.getElementById('response-area').style.display = 'block';
}
```

---

### Core Flow Recap
User input → **POST /ask** → FastAPI validation → **DocumentRetriever** finds chunks → **AnswerGenerator** calls LLM → FastAPI packages answer → frontend renders guidance.  These six steps constitute the entire synchronous path the user experiences.
