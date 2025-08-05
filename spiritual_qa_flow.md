# Spiritual Q&A System: End-to-End Flow Documentation

This document explains the complete journey of a user question through the Spiritual Q&A system, from frontend input to response display.

## Table of Contents
1. [User Input & Frontend Processing](#user-input--frontend-processing)
2. [API Request Handling](#api-request-handling)
3. [Document Retrieval & Processing](#document-retrieval--processing)
4. [Answer Generation](#answer-generation)
5. [Response Delivery & Display](#response-delivery--display)

## User Input & Frontend Processing

### User Interface Components
The journey begins when a user interacts with the frontend interface to submit a spiritual question.

---

## 1. User Input & Frontend Processing

### 1.1 Key Code Snippet (frontend/app.js)
```javascript
// handle form submission and fire API call
async handleQuestionSubmit(e) {
  e.preventDefault();
  if (this.isLoading) return;
  const question      = document.getElementById('spiritual-question').value.trim();
  const model         = document.getElementById('llm-model').value;
  const readingStyle  = document.querySelector('input[name="reading-style"]:checked').value;
  if (!question) { this.showNotification('Please enter a spiritual question','warning'); return; }
  this.isLoading = true;
  try {
    const response = await this.askSpiritualQuestion(question, model, readingStyle);
    this.displayResponse(response);
  } finally {
    this.isLoading = false;
  }
}

async askSpiritualQuestion(question, model = 'gpt-4.1', readingStyle = 'balanced') {
  const retrieval = this.getRetrievalParamsFromStyle(readingStyle);
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
**Explanation**
1. The `handleQuestionSubmit` listener captures the question and UI-selected parameters, validates them, then forwards the data to `askSpiritualQuestion`.
2. `askSpiritualQuestion` converts the reading-style preset into concrete retrieval parameters (MMR, diversity, k) and issues a JSON POST to `/ask`.

---

## 2. API Request Handling

### 2.1 Key Code Snippet (api/spiritual_api.py)
```python
@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    start_time = time.time()
    generator = get_answer_generator(request.model)
    result = generator.generate_answer(
        query=request.query,
        k=request.k,
        query_type=request.query_type
    )
    processing_time = time.time() - start_time
    return QueryResponse(
        status="success",
        answer=result["answer"],
        chunks_used=result["chunks_used"],
        model=result["model"],
        query_type=result["prompt_type"],
        processing_time=processing_time
    )
```
**Explanation**
1. FastAPI validates the incoming JSON against `QueryRequest`.
2. `get_answer_generator` provides a cached `AnswerGenerator` instance for the chosen model.
3. `generate_answer` encapsulates retrieval + LLM call; the endpoint simply relays its structured result back to the caller.

---

## 3. Document Retrieval & Processing

### 3.1 Key Code Snippet (document_retriever.py)
```python
def retrieve_chunks(self, query, k=5, use_mmr=True, diversity=0.7):
    query_result = self.query_processor.process_query(query)
    if use_mmr:
        result = self.query_processor.mmr_retrieval(
            query_text=query, k=k, diversity=diversity
        )
    else:
        result = self.query_processor.similarity_retrieval(
            query_text=query, k=k
        )
    return {"status": "success", "chunks": result}
```
**Explanation**
1. The user query is embedded and compared against document vectors in ChromaDB.
2. Retrieval strategy depends on `use_mmr`:
   • **Similarity search** – top-k most relevant
   • **MMR** – balances relevance with diversity using `diversity` lambda
3. Returns a list of chunk dictionaries (text + metadata).

---

## 4. Answer Generation

### 4.1 Key Code Snippet (answer_generator.py)
```python
def generate_answer(self, query, k=5, query_type=None):
    retrieval   = self.retrieve_relevant_chunks(query, k)
    context     = format_context_from_chunks(retrieval["chunks"])
    prompt_tmpl = select_prompt_template(query_type)
    prompt      = prompt_tmpl.format(context=context, question=query)
    response    = client.chat.completions.create(
        model=self.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=self.temperature,
        max_tokens=self.max_tokens
    )
    return {
        "status": "success",
        "answer": response.choices[0].message.content,
        "chunks_used": len(retrieval["chunks"]),
        "model": self.llm_model,
        "prompt_type": query_type or "default"
    }
```
**Explanation**
1. Relevant chunks are folded into a single **context block** via `format_context_from_chunks`.
2. Prompt template selection is dynamic (`query_type`).
3. OpenAI (or other provider) returns the final answer text, which is bundled with metadata for the API response.

---

## 5. Response Delivery & Display

### 5.1 Key Code Snippet (frontend/app.js)
```javascript
// Render answer & metadata
displayResponse(responseData) {
  document.getElementById('question-display').textContent   = responseData.query;
  document.getElementById('guidance-response').textContent  = responseData.response;
  const sourcesList = document.getElementById('sources-list');
  sourcesList.innerHTML = '';
  responseData.sources.forEach(src => {
    const li = document.createElement('li');
    li.textContent = src;
    sourcesList.appendChild(li);
  });
}
```
**Explanation**
1. The JSON payload from `/ask` is mapped back onto the UI, separating answer text from citations and metadata.
2. Dynamic DOM updates ensure a seamless user experience without page reloads.

---

> **Core Flow Recap**: The user’s question moves synchronously through five critical stages—Frontend → API → Retrieval → LLM Generation → Frontend Render—before the answer appears on screen.
