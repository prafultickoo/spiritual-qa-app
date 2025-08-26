# Spiritual Q&A System: Detailed Backend Flow (Part 3)

This part focuses on the **core retrieval and generation loop** that converts the user’s embedded question into an answer string.

Sections:
1. Vector Similarity / MMR Search
2. Context Assembly
3. Prompt Template Selection & Filling
4. Large-Language-Model Invocation
5. Post-processing & Packaging

---

## 1. Vector Similarity / MMR Search

### Detailed Explanation
After obtaining the query embedding (see Part 2), the system must fetch the most relevant document chunks.

1. `DocumentRetriever.retrieve_chunks()` chooses between two strategies:
   - **Pure Similarity** (`use_mmr=False`) ➜ return top-k most similar vectors.
   - **MMR (Max-Marginal-Relevance)** (`use_mmr=True`) ➜ first fetch a larger candidate pool (`fetch_k`) then rerank to balance diversity vs relevance via `diversity` (LangChain’s `lambda_mult`).
2. The `QueryProcessor` helper performs the actual call to Chroma via `vector_store.similarity_search()` or `max_marginal_relevance_search()`.
3. The resulting `Document` objects are converted to plain dictionaries (`{"page_content": ..., "metadata": ...}`) for downstream use.
4. Detailed logging captures strategy, k, diversity, and timing.

### Code (extracts from `document_retriever.py` & `query_utils.py`)
```python
# document_retriever.py
if use_mmr:
    fetch_k = max(k * 3, 15)  # wider pool for diversity
    result = self.query_processor.mmr_retrieval(
        query_text=query,
        k=k,
        fetch_k=fetch_k,
        diversity=diversity,
        filter_metadata=filter_metadata
    )
else:
    result = self.query_processor.retrieve_relevant_chunks(
        query_text=query,
        k=k,
        filter_metadata=filter_metadata
    )
return result
```
```python
# query_utils.py – similarity version
 def retrieve_relevant_chunks(self, query_text: str, k: int = 5, **kw):
     docs = self.vector_store.similarity_search(query_text, k=k)
     return {
         "status": "success",
         "chunks": [d.dict() for d in docs]
     }

 # MMR version
 def mmr_retrieval(self, query_text: str, k: int, fetch_k: int, diversity: float, **kw):
     docs = self.vector_store.max_marginal_relevance_search(query_text,
                                                            k=k,
                                                            fetch_k=fetch_k,
                                                            lambda_mult=diversity)
     return {
         "status": "success",
         "chunks": [d.dict() for d in docs]
     }
```

---

## 2. Context Assembly

### Detailed Explanation
The raw chunks must be concatenated into a single **context block** to feed the LLM.  Requirements:
- Preserve original ordering (by relevance score returned).
- Include metadata such as source filename & page so answers can cite sources.
- Limit total tokens so prompt fits model’s context window.

The helper `format_context_from_chunks()` (in `prompts/answer_prompts.py`) handles these duties:
1. Iterates through chunks up to `max_context_docs` (default = 5) trimming whitespace.
2. Formats each chunk as:
   ```
   [Source: filename – page]
   «chunk text»
   ```
3. Joins blocks with two line breaks.

### Code
```python
# prompts/answer_prompts.py
MAX_DOCS = 5

def format_context_from_chunks(chunks: list) -> str:
    formatted = []
    for idx, ch in enumerate(chunks[:MAX_DOCS]):
        meta = ch.get("metadata", {})
        src  = f"{meta.get('source', 'unknown')} – {meta.get('page', '?') }"
        text = ch.get("page_content", "").strip()
        formatted.append(f"[Source: {src}]\n{text}")
    return "\n\n".join(formatted)
```

---

## 3. Prompt Template Selection & Filling

### Detailed Explanation
Different query types (general, verse-focused, comparative, practical) require slightly different instructions.  `select_prompt_template()` decides which of four predefined strings to use, falling back to `ANSWER_BASE_PROMPT`.

Once selected, the generator fills `{context}` and `{question}` placeholders.

### Code
```python
# prompts/answer_prompts.py
TEMPLATES = {
   None: ANSWER_BASE_PROMPT,
   "verse_focused": VERSE_FOCUSED_PROMPT,
   "comparative":   COMPARATIVE_ANALYSIS_PROMPT,
   "practical":     PRACTICAL_APPLICATION_PROMPT,
}

def select_prompt_template(query_type: str = None):
    return TEMPLATES.get(query_type, ANSWER_BASE_PROMPT)
```
```python
# answer_generator.py (within generate_answer)
context     = format_context_from_chunks(retrieval["chunks"])
prompt_tmpl = select_prompt_template(query_type)
prompt      = prompt_tmpl.format(context=context, question=query)
```

---

## 4. Large-Language-Model Invocation

### Detailed Explanation
1. The assembled `prompt` is sent to the chosen model via the vendor SDK (OpenAI, Anthropic, etc.).
2. Parameters (`temperature`, `max_tokens`) originate from the model config when the generator was instantiated.
3. The call is synchronous – user waits for completion; retries are handled upstream in higher-level fault-tolerance wrappers (omitted here).

### Code (OpenAI client example)
```python
response = client.chat.completions.create(
    model=self.llm_model,
    messages=[{"role": "user", "content": prompt}],
    temperature=self.temperature,
    max_tokens=self.max_tokens,
)
answer_text = response.choices[0].message.content
```

---

## 5. Post-processing & Packaging

### Detailed Explanation
1. The raw LLM response is wrapped with metadata:
   - `chunks_used`: how many context docs were supplied
   - `model`: id of the LLM
   - `prompt_type`: template chosen
2. API endpoint converts this dict into a `QueryResponse` Pydantic model ➜ FastAPI serializes to JSON.
3. Frontend (see Part 1) renders `response`, `sources`, etc.

### Code
```python
# answer_generator.py – still inside generate_answer
return {
    "status": "success",
    "answer": answer_text,
    "chunks_used": len(retrieval["chunks"]),
    "model": self.llm_model,
    "prompt_type": query_type or "default",
}
```
```python
# spiritual_api.py – endpoint response
return QueryResponse(
    status="success",
    answer=result["answer"],
    chunks_used=result["chunks_used"],
    model=result["model"],
    query_type=result["prompt_type"],
    processing_time=time.time() - start_time,
)
```

---

**Up Next (Part 4)** – Answer reception by the frontend, DOM rendering, clipboard share, and admin panel logging.
