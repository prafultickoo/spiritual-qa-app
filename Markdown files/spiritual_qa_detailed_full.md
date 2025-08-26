# Spiritual Q&A System – **Full Granular Runtime Flow**

Welcome!  
This guide explains **how a user question travels through the Spiritual Q&A application** — from button-click to answer display — in plain language first, then code. If you were *not* on the dev team, don’t worry; each step begins with an English explanation before any snippet appears.

A high-level flowchart (`Flowchart.png`) sits in the project root. Open it alongside this document if you prefer a visual map while reading.


---

## Master Table of Contents
1. Frontend Capture & Validation  
   1.1 Event Listener Registration  
   1.2 `handleQuestionSubmit()` Logic  
   1.3 Loading-State Management
2. Parameter Processing  
   2.1 Reading-Style → Retrieval-Params Map
3. API Request Construction & Transmission  
4. Backend API Reception & Validation  
5. Answer-Generator Preparation  
6. Document Retrieval Pipeline  
   6.1 Query Embedding Creation  
   6.2 Similarity vs. MMR Search  
7. Context Assembly  
8. Prompt Template Selection & Filling  
9. LLM Invocation  
10. Post-processing & API Response  
11. Frontend Response Handling  
    11.1 Promise Resolution & Error Guard  
    11.2 DOM Rendering Sequence  
12. Auxiliary UI Actions (copy / share / ask-another)  
13. Admin Panel Logging

---

## How to Read This Document (Non-Developers)

1. **Start with the English explanation** at the top of every section; skip the fenced code block if you only want the concept.
2. The *bold* term inside each explanation is the step name that appears in the Table of Contents.
3. If a word feels unfamiliar, consult the Glossary at the end — no prior ML or backend knowledge assumed.
4. Follow the arrows in `Flowchart.png` while you read; each numbered box maps to a section here.

---

## 1. Frontend Capture & Validation

**English Explanation:** This phase begins entirely on the browser side. As soon as the page finishes loading, we wire up JavaScript event listeners so that a seeker’s click on the **“Seek Wisdom”** button is intercepted by our code instead of triggering the default page refresh. The listener then:
1. Verifies that no other question is already being processed (avoiding double-submits).
2. Extracts the question text, the LLM model the user selected, and the desired *reading style* (Deep / Balanced / Broad).
3. Activates loading indicators to give instant UI feedback.
4. Hands control to the asynchronous function that will talk to the backend.


### 1.1 Event Listener Registration (frontend/app.js)

In plain English, we attach a `submit` listener to the form that wraps the question input box. This guarantees that every time the user presses the button (or hits Enter) our JavaScript handler runs instead of the browser’s default form submission behaviour.

```javascript
const questionForm = document.getElementById('question-form');
if (questionForm) {
    questionForm.addEventListener('submit', (e) => this.handleQuestionSubmit(e));
}
```

### 1.2 `handleQuestionSubmit()` Core Logic

The `handleQuestionSubmit` function is the heart of the frontend workflow. It prevents a full-page reload, validates the question, toggles loading spinners, calls the backend, and finally restores the UI state—displaying either the spiritual guidance or an error toast.

```javascript
async handleQuestionSubmit(e) {
  e.preventDefault();                  // stop full-page reload
  if (this.isLoading) return;          // duplicate-submit guard

  const question     = document.getElementById('spiritual-question').value.trim();
  const model        = document.getElementById('llm-model').value;
  const readingStyle = document.querySelector('input[name="reading-style"]:checked').value;

  if (!question) {
      this.showNotification('Please enter a spiritual question', 'warning');
      return;
  }

  this.isLoading = true;
  this.setLoadingState(true);
  try {
      const response = await this.askSpiritualQuestion(question, model, readingStyle);
      this.displayResponse(response);
      this.showNotification('Guidance received! 🙏', 'success');
  } catch (err) {
      console.error(err);
      this.showNotification('Failed to receive guidance', 'error');
  } finally {
      this.isLoading = false;
      this.setLoadingState(false);
  }
}
```

---

## 2. Parameter Processing – Reading-Style Mapping

**English Explanation:** Different seekers prefer different depths of answer. We expose this as *reading styles* on the UI and translate them to concrete retrieval parameters. For instance, *Deep* means the system should focus purely on relevance (no diversity), while *Broad* favours a wider thematic spread using MMR with a low λ (diversity) value.

```javascript
getRetrievalParamsFromStyle(style) {
  if (this.adminControls && document.querySelector('.admin-card.active')) {
      return this.adminControls.getCurrentSettings();
  }
  switch (style) {
    case 'broad':    return { use_mmr: true,  diversity: 0.2, k: 7 };
    case 'deep':     return { use_mmr: false, diversity: 0.0, k: 5 };
    case 'balanced':
    default:         return { use_mmr: true,  diversity: 0.6, k: 6 };
  }
}
```

---

## 3. API Request Construction & Transmission

**English Explanation:** With all parameters gathered, the browser crafts a JSON body describing the seeker’s question, chosen LLM model, and retrieval preferences. It then performs a `fetch()` POST to the `/ask` (alias `/query`) endpoint. This step serialises data, adds appropriate headers, and includes client-side error handling so that any non-2xx HTTP status surfaces immediately to the user.
```javascript
async askSpiritualQuestion(q, model, style) {
  const r = this.getRetrievalParamsFromStyle(style);
  const res = await fetch(`${this.apiBase}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
          question: q,
          model,
          reading_style: style,
          max_context_docs: 5,
          use_mmr: r.use_mmr,
          diversity: r.diversity,
          k: r.k
      })
  });
  if (!res.ok) throw new Error((await res.json()).detail || 'API error');
  return res.json();
}
```

---

## 4. Backend API Reception & Validation (api/spiritual_api.py)

**English Explanation:** FastAPI receives the HTTP request and—before our code even runs—validates the payload against the `QueryRequest` Pydantic model. Missing or incorrectly typed fields trigger an automatic 422 response. For valid input, we timestamp the start, fetch an `AnswerGenerator`, and pass control deeper into the pipeline.
```python
class QueryRequest(BaseModel):
    query: str
    model: str = DEFAULT_LLM_MODEL
    k: int = 5
    query_type: Optional[str] = None
    use_mmr: bool = True
    diversity: float = 0.7
    max_context_docs: int = 5

@app.post('/query', response_model=QueryResponse)
async def query(request: QueryRequest):
    start_time = time.time()
    generator = get_answer_generator(request.model)
    result    = generator.generate_answer(
                   query=request.query,
                   k=request.k,
                   query_type=request.query_type)
    return QueryResponse(
        status='success',
        answer=result['answer'],
        chunks_used=result['chunks_used'],
        model=result['model'],
        query_type=result['prompt_type'],
        processing_time=time.time() - start_time)
```

---

## 5. Answer-Generator Preparation

**English Explanation:** Generators encapsulate model config, vector-store handles, and helper objects. Creating them is expensive, so we store them in a global `GENERATORS` dict keyed by model id. The first request for a given model constructs and caches the generator; subsequent requests reuse it, giving near-instant startup.
```python
GENERATORS = {}

def get_answer_generator(model:str):
    if model in GENERATORS:
        return GENERATORS[model]
    if model not in LLM_MODELS:
        model = DEFAULT_LLM_MODEL
    cfg = LLM_MODELS[model]
    gen = AnswerGenerator(
            vector_store_dir=VECTOR_STORE_DIR,
            embedding_model='openai',
            llm_model=cfg['model_id'],
            temperature=cfg.get('temperature',0.0),
            max_tokens=cfg.get('max_tokens',1024))
    GENERATORS[model] = gen
    return gen
```

---

## 6. Document Retrieval Pipeline

**English Explanation:** The generator now needs supporting context. It converts the user’s natural-language question into an embedding and queries ChromaDB for relevant chunks. Two retrieval modes exist:
• *Similarity* – return the top-k most relevant chunks.  
• *MMR* – retrieve a wider candidate pool then greedily select chunks that maximise diversity vs relevance according to the `diversity` λ parameter.
### 6.1 Query Embedding Creation (`QueryProcessor.process_query`)
```python
if not query_text.strip():
    return { 'status':'error','error':'empty' }
embedding = self.embedding_model.embed_query(query_text)
```

### 6.2 Similarity vs. MMR Search
```python
# document_retriever.py
if use_mmr:
    docs = self.query_processor.mmr_retrieval(query, k, fetch_k=max(k*3,15), diversity=diversity)
else:
    docs = self.query_processor.retrieve_relevant_chunks(query, k)
```

---

## 7. Context Assembly (`format_context_from_chunks`)

**English Explanation:** Raw chunks are combined into a single context block. Each chunk is prefixed with a source citation so the eventual answer can reference its origins. Only the first `max_context_docs` chunks are used, keeping the prompt within the LLM’s context window.
```python
blocks = []
for ch in chunks[:MAX_DOCS]:
    meta = ch['metadata']
    blocks.append(f"[Source: {meta.get('source','?')} – {meta.get('page','?')}]\n{ch['page_content'].strip()}")
context = "\n\n".join(blocks)
```

---

## 8. Prompt Template Selection & Filling

**English Explanation:** We maintain multiple prompt templates tailored to specific query types (e.g., verse-focused, comparative). The system selects one via `select_prompt_template`, then substitutes `{context}` and `{question}` placeholders to form the final prompt string.
```python
template = select_prompt_template(query_type)
prompt   = template.format(context=context, question=query)
```

---

## 9. LLM Invocation

**English Explanation:** The filled prompt is sent to the target LLM’s chat/completions endpoint via its official SDK (OpenAI, Anthropic, etc.). Runtime parameters like `temperature` and `max_tokens` come from the model’s config, ensuring deterministic behaviour for GPT-4.1 (temperature 0). The call blocks until we receive the model’s textual answer.
```python
response = client.chat.completions.create(
             model=self.llm_model,
             messages=[{'role':'user','content':prompt}],
             temperature=self.temperature,
             max_tokens=self.max_tokens)
answer_text = response.choices[0].message.content
```

---

## 10. Post-processing & API Response

**English Explanation:** The backend wraps the LLM’s answer together with metadata—how many chunks were used, which model responded, and which prompt variant was chosen—then returns the data as a JSON `QueryResponse`. Errors propagate with meaningful messages so the frontend can display user-friendly toasts.
```python
return {
  'status':'success',
  'answer': answer_text,
  'chunks_used': len(chunks),
  'model': self.llm_model,
  'prompt_type': query_type or 'default'}
```

---

## 11. Frontend Response Handling

**English Explanation:** On the browser, the pending `fetch()` promise resolves. Success flow calls `displayResponse`, which fills the UI with the answer, metadata, and citations, and hides loading spinners. Failure flow shows an error toast and keeps the question box intact so the user can tweak and retry.
### 11.1 Promise Resolution & Guard
```javascript
const res = await this.askSpiritualQuestion(...);
this.displayResponse(res);
```

### 11.2 DOM Rendering Sequence (`displayResponse`)
```javascript
responseArea.style.display = 'block';
answerBox.textContent   = res.response || res.answer;
modelUsed.textContent   = `Model: ${res.model}`;
responseTime.textContent= `${res.processing_time}ms`;
sourcesList.innerHTML   = '';
(res.sources||[]).forEach(s=>{ const li=document.createElement('li');li.textContent=s;sourcesList.appendChild(li); });
```

---

## 12. Auxiliary UI Actions

**English Explanation:** Beyond displaying answers, the UI provides convenience buttons: copy the text to clipboard, share via the native Web Share API, or reset the form for another question. Each action has a small helper method and visual feedback toast.
```javascript
copyResponse() { navigator.clipboard.writeText(answerBox.innerText)... }
shareResponse() { if(navigator.share) navigator.share({text:answerBox.innerText}); }
```

---

## 13. Admin Panel Logging

**English Explanation:** The Admin tab caters to power users. It periodically fetches current backend settings and the latest log entries, displaying them in an on-page console. This lets administrators adjust parameters in real-time and observe system activity without SSH’ing into the server.
```javascript
async loadLogs(){
  const logs = await (await fetch(`${this.apiBase}/logs?limit=100`)).json();
  document.getElementById('api-log').textContent = logs.entries.join('\n');
}
```

---

### Flow Complete
The combined flow above replaces the four separate part-files and can now serve as a single source of truth for developers and maintainers.

---

# Additional Runtime Pipelines

While the previous sections covered the primary Q&A request-response loop, the application also contains two auxiliary pipelines that keep the experience robust and user-friendly:

14. PDF Rotation & OCR Correction Pipeline  
15. One-Click Launcher & Deployment Flow

These are documented below in the same **Explanation → Code** format.

---

## 14. PDF Rotation & OCR Correction Pipeline

**English Explanation:** Many scanned scriptures arrive with pages rotated or stored as images without extractable text. The *PDF Orientation* utilities automatically analyse each file, decide whether simple rotation or full OCR is needed, and write a corrected copy. This ensures verses are readable before chunking/vectorisation.

The pipeline proceeds as follows:
1. `analyze_pdf_orientation()` — Quickly samples pages to detect skewed text or missing extractable text.
2. If `needs_correction` is false, we’re done. Otherwise it chooses a **correction method**:  
   • `rotation` — text exists but is rotated.  
   • `ocr` — no reliable text; we must rasterise pages and run Tesseract.
3. `correct_pdf_orientation()` calls either `apply_rotation_correction` or `apply_ocr_with_orientation_correction` accordingly.
4. The corrected PDF is saved under `corrected_<original>.pdf`, and downstream loaders pick that version.

```python
from utils.pdf_orientation_tools import analyze_pdf_orientation, correct_pdf_orientation

analysis = analyze_pdf_orientation('Documents/Bhagavad_Gita.pdf')
if analysis['needs_correction']:
    result = correct_pdf_orientation('Documents/Bhagavad_Gita.pdf')
    print(result['message'], '➡', result['output_path'])
```

Key internals (excerpt):
```python
# utils/pdf_orientation_tools.py – simplified
if analysis['correction_method'] == 'ocr':
    return apply_ocr_with_orientation_correction(pdf_path, output_path)
else:
    return apply_rotation_correction(pdf_path, output_path, analysis['pages_with_issues'])
```

---

## 15. One-Click Launcher & Deployment Flow

**English Explanation:** To give non-technical seekers a friction-free start-up, the script `start_app.py` boots **both** the FastAPI backend and a simple static web server for the frontend, opens the browser automatically, colour-codes log output, and cleans up on Ctrl+C.

Steps:
1. Display an ASCII-art banner so users know something is happening.
2. Spawn the backend with `subprocess.Popen`, capturing stdout for live logging.
3. Spawn the frontend using Python’s built-in `http.server` on port 8080.
4. Launch the default web browser to `http://localhost:8080` after a 3-second delay.
5. Monitor SIGINT; on Ctrl+C it terminates both child processes gracefully.

```python
# run from project root
author$ python start_app.py
```

Important snippets:
```python
# start_app.py – start_backend()
return subprocess.Popen([
    'python', os.path.join(BACKEND_DIR, 'main.py')
], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
```
```python
# Colourful logging thread
for line in iter(process.stdout.readline, ''):
    print(f"Backend: {line.strip()}")
```

### Production Deployment
For cloud hosting the same philosophy applies — a single container image (or Netlify site) starts gunicorn/Uvicorn for FastAPI and serves the React/Vanilla JS bundle via Nginx. The local launcher mirrors this behaviour so bugs are caught early.

---

---

## 16. RAG Component Function Reference (Every Function in the Pipeline)

This section provides an exhaustive breakdown of every function used in our Retrieval-Augmented Generation pipeline, explaining exactly how each piece works and showing the actual code that implements it. Understanding these building blocks will give you complete insight into how the Q&A system transforms user questions into contextual answers.

### 16.1 Document Chunking Functions – `utils/chunking_utils.py`

#### 16.1.1 `split_text_into_chunks(text, chunk_size, chunk_overlap)`

**Detailed Explanation:**

This function serves as the entry point for document chunking, breaking a large document into smaller, manageable pieces that can be vectorized and stored in our embedding database. It handles the critical task of balancing chunk size (for efficient processing) against semantic coherence (preserving the meaning of the content).

The function works through the following steps:

1. First, it converts the requested token-based chunk size and overlap into approximate character counts (using a 4:1 character-to-token ratio). This gives us workable character limits.

2. It identifies paragraphs in the text by splitting on double newlines, giving us natural semantic units to work with.

3. For each paragraph, it checks if adding it to the current chunk would exceed our target chunk size. If not, it adds the paragraph; if yes, it finalizes the current chunk and starts a new one.

4. When starting a new chunk, it intelligently carries over a portion of the previous chunk (the "overlap") to maintain context continuity between chunks.

5. After initial chunking, it passes the chunks to `ensure_verse_integrity()` to preserve Sanskrit verses that might have been split between chunks.

This approach prioritizes verse integrity over strict adherence to chunk size limits, ensuring that spiritual verses remain intact for accurate search and retrieval later.

**Code Implementation:**

```python
def split_text_into_chunks(text: str, chunk_size: int = 1500, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into chunks with specified size and overlap.
    Prioritizes verse integrity over strict chunk size.
    
    Args:
        text (str): Text to split into chunks
        chunk_size (int, optional): Target size of each chunk in tokens (approx. 4 chars per token)
        chunk_overlap (int, optional): Overlap between consecutive chunks in tokens
        
    Returns:
        List[str]: List of text chunks
    """
    logger.info(f"Splitting text into chunks (target size: {chunk_size}, overlap: {chunk_overlap})")
    
    # Approximate conversion from tokens to characters (4 chars ~= 1 token)
    char_size = chunk_size * 4
    char_overlap = chunk_overlap * 4
    
    # First, try to identify verse patterns
    # Simple verse identification pattern - can be expanded based on actual document structure
    verse_pattern = r'(\d+\.\d+\s*[\|\॥].+?[\|\॥])|(\d+\s*[\|\॥].+?[\|\॥])'
    
    # Identify paragraphs
    paragraphs = re.split(r'\n\n+', text)
    
    chunks = []
    current_chunk = ""
    current_size = 0
    
    for paragraph in paragraphs:
        # Check if adding this paragraph would exceed the chunk size
        if len(paragraph) + current_size <= char_size or current_size == 0:
            # If it fits or it's the first paragraph, add it to current chunk
            current_chunk += paragraph + "\n\n"
            current_size += len(paragraph) + 2
        else:
            # If it doesn't fit, start a new chunk with some overlap
            chunks.append(current_chunk)
            
            # Get the tail of the last chunk for overlap
            words = current_chunk.split()
            overlap_word_count = min(len(words), int(char_overlap / 5))  # Approx. 5 chars per word
            overlap_text = ' '.join(words[-overlap_word_count:]) if overlap_word_count > 0 else ""
            
            # Start new chunk with overlap and add current paragraph
            current_chunk = overlap_text + "\n\n" + paragraph + "\n\n"
            current_size = len(current_chunk)
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    # Post-processing: Ensure verse integrity
    chunks = ensure_verse_integrity(chunks, verse_pattern)
    
    logger.info(f"Created {len(chunks)} chunks")
    return chunks
```

#### 16.1.2 `ensure_verse_integrity(chunks, verse_pattern)`

**Detailed Explanation:**

This function addresses a critical issue in spiritual text processing: ensuring that Sanskrit verses (shlokas) remain intact and are not split between different chunks. This preservation is essential because Sanskrit verses often contain precise spiritual meaning that could be lost or distorted if the verse were fragmented.

The function works as follows:

1. It examines each chunk, looking for verses that match our verse pattern (typically formatted with specific Sanskrit punctuation markers like '|' or '॥').

2. When it finds a verse at the end of a chunk, it checks if the next chunk begins with another verse pattern.

3. If this happens, it's likely that a verse was split between chunks, so it merges the current chunk with the next one to keep the verse together.

4. It builds a new list of chunks where verses remain intact, potentially sacrificing the strict chunk size limit in favor of semantic integrity.

This post-processing step is crucial for maintaining the meaning and searchability of spiritual texts, where verse structure carries significant theological importance. By preserving verse boundaries, we ensure that later semantic searches will find complete verses rather than fragments.

**Code Implementation:**

```python
def ensure_verse_integrity(chunks: List[str], verse_pattern: str) -> List[str]:
    """
    Ensure that verses aren't split across chunks.
    
    Args:
        chunks (List[str]): Initial text chunks
        verse_pattern (str): Regex pattern to identify verses
        
    Returns:
        List[str]: Chunks with preserved verse integrity
    """
    logger.info("Ensuring verse integrity across chunks")
    result_chunks = []
    
    for i, chunk in enumerate(chunks):
        # Look for verses that might be split at the end of this chunk
        matches = re.finditer(verse_pattern, chunk)
        last_match_end = 0
        last_match = None
        
        for match in matches:
            last_match = match
            last_match_end = match.end()
        
        # If we have a match and it's near the end of the chunk
        if last_match and chunk[last_match_end:].strip() == "" and i < len(chunks) - 1:
            # The verse is the last thing in this chunk, check if it continues in the next chunk
            next_chunk = chunks[i+1]
            if re.match(verse_pattern, next_chunk.strip()):
                # It likely continues, so merge with next chunk
                chunks[i+1] = chunk + next_chunk
                continue
        
        result_chunks.append(chunk)
    
    return result_chunks
```

#### 16.1.3 `extract_verses(text)`

**Detailed Explanation:**

This specialized function identifies and extracts Sanskrit verses from text chunks. It's designed to recognize multiple verse formatting styles commonly found in spiritual texts. Extracting verses serves several purposes in our RAG pipeline:

1. It allows us to store verse information as metadata alongside each chunk, making verses searchable even when they're not explicitly mentioned in a query.

2. It enables our LLM to directly reference specific verses when answering questions, enhancing the authority and accuracy of responses.

3. It provides a way to quantify the spiritual content density of each chunk (by counting verses), which can be used as a ranking signal.

The function works by applying multiple regex patterns that match different verse formatting conventions, including:

- Numbered verses with specific punctuation (e.g., "1.1 | verse text |") 
- Sanskrit shlokas with traditional markers (e.g., "॥ verse text ॥")
- Other formatting variations found in our spiritual corpus

This comprehensive approach ensures we don't miss verses due to formatting inconsistencies across different source texts.

**Code Implementation:**

```python
def extract_verses(text: str) -> List[str]:
    """
    Extract verses from text based on common verse formatting.
    
    Args:
        text (str): Input text
        
    Returns:
        List[str]: Extracted verses
    """
    # Various verse patterns to look for
    patterns = [
        r'\d+\.\d+\s*[\|\॥].+?[\|\॥]',  # Format: 1.1 | verse text |
        r'\d+\s*[\|\॥].+?[\|\॥]',  # Format: 1 | verse text |
        r'श्लोक[^\n]+\n.+?\n\n',  # Sanskrit shloka format
        r'॥.+?॥'  # Traditional Sanskrit verse markers
    ]
    
    verses = []
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)
        for match in matches:
            verses.append(match.group(0))
    
    return verses
```

#### 16.1.4 `format_chunk_for_embedding(chunk, metadata)`

**Detailed Explanation:**

This function transforms a raw text chunk and its associated metadata into a structured format ready for embedding. It serves as the bridge between the chunking process and the vectorization process, enriching each chunk with critical metadata that will enhance retrieval quality later.

Here's how the function works and why it's important:

1. **Token Counting**: First, it estimates the token count of the chunk using our simple heuristic (4 characters ≈ 1 token). This is vital for:
   - Monitoring API usage with OpenAI's embedding service
   - Ensuring we don't exceed token limits during embedding
   - Providing metadata for potential token-based filtering/ranking

2. **Verse Extraction**: It calls the `extract_verses()` function we just examined to identify all verses in the chunk. This creates a valuable semantic index of spiritual content that can be used during retrieval and answer generation.

3. **Metadata Enrichment**: It combines document-level metadata (filename, title, author) with chunk-specific metadata (token count, verses, verse count) to create a rich, searchable record.

4. **Structured Output**: The final dictionary contains both the chunk's text content and its metadata, ready for embedding and storage in our vector database.

This enrichment process makes our RAG system more powerful than simple semantic search. By storing verse information alongside each chunk, we enable:

- Verse-specific queries ("What does verse 2.41 say about karma?")
- Source filtering ("What does the Bhagavad Gita say about dharma?")
- Content density ranking (chunks with more verses might be more authoritative)

Additionally, by tracking token counts, we maintain visibility into our API usage and can implement optimizations if needed.

**Code Implementation:**

```python
def format_chunk_for_embedding(chunk: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a chunk with its metadata for embedding.
    
    Args:
        chunk (str): Text chunk
        metadata (Dict[str, Any]): Document metadata
        
    Returns:
        Dict[str, Any]: Formatted chunk with metadata
    """
    # Estimate token count
    token_count = estimate_token_count(chunk)
    
    # Extract verses if present
    verses = extract_verses(chunk)
    
    return {
        "text": chunk,
        "token_count": token_count,
        "verses": verses,
        "source": metadata.get("filename", ""),
        "title": metadata.get("title", ""),
        "author": metadata.get("author", ""),
        "verse_count": len(verses)
    }
```

#### 16.1.5 Supporting Function: `estimate_token_count(text)`

**Detailed Explanation:**

This utility function provides a quick and efficient way to estimate the number of tokens in a piece of text without calling an expensive tokenizer API. Token counts are crucial throughout our RAG pipeline for:

1. Ensuring we don't exceed token limits when sending text to embedding or LLM APIs
2. Splitting documents into appropriately-sized chunks
3. Monitoring API usage and costs
4. Planning batching strategies for vectorization

Rather than using a full tokenizer (which would add dependencies and slow down processing), this function uses a simple heuristic: each token is approximately 4 characters in length. This estimate works reasonably well for English and transliterated Sanskrit text.

While not perfectly accurate (actual tokenization depends on the specific model), this approximation provides a good balance between speed and usefulness, especially during preprocessing when exact token counts aren't critical.

**Code Implementation:**

```python
def estimate_token_count(text: str) -> int:
    """
    Estimate token count based on text length.
    
    Args:
        text (str): Input text
        
    Returns:
        int: Estimated token count
    """
    # Rough estimation: 1 token ≈ 4 characters
    return len(text) // 4
```

### 16.2 Embedding & Vector Store – `utils/vectorization_utils.py`

The vectorization module is responsible for converting our processed text chunks into numerical vector embeddings and storing them in a searchable database. This is the critical step that enables semantic search by transforming text into a mathematical space where similarity can be computed.

#### 16.2.1 `DocumentVectorizer` Class

**Detailed Explanation:**

This class encapsulates all the functionality needed to convert document chunks into embeddings and store them in a vector database. It's designed to be flexible, supporting different embedding models (OpenAI or HuggingFace) and handling database persistence.

The class maintains the state of the embedding process, including:
- The chosen embedding model and its configuration
- The vector database connection (ChromaDB)
- The collection name where vectors will be stored
- Persistence directory for database files

By providing this abstraction, the rest of our application can work with document vectorization without needing to understand the details of embedding APIs or vector database implementation.

**Initialization Code:**

```python
class DocumentVectorizer:
    """Class for vectorizing document chunks and storing in a vector database."""
    
    def __init__(self, 
                 persist_directory: str, 
                 embedding_model: str = "openai",
                 embedding_model_kwargs: Optional[Dict[str, Any]] = None,
                 collection_name: str = "spiritual_texts"):
        """
        Initialize the document vectorizer.
        
        Args:
            persist_directory: Directory to persist vector database
            embedding_model: Name of embedding model to use ('openai' or 'huggingface')
            embedding_model_kwargs: Additional keyword arguments for embedding model
            collection_name: Name of the ChromaDB collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize embedding model
        if embedding_model not in EMBEDDING_MODELS:
            logger.warning(f"Unknown embedding model: {embedding_model}. Defaulting to OpenAI.")
            self.embedding_model_name = "openai"
            
        # Default kwargs for embedding models
        default_kwargs = {
            "openai": {"model": "text-embedding-ada-002"},
            "huggingface": {"model_name": "sentence-transformers/all-MiniLM-L6-v2"}
        }
        
        # Use provided kwargs or defaults
        model_kwargs = embedding_model_kwargs or default_kwargs.get(self.embedding_model_name, {})
        
        # Initialize the embedding model
        try:
            EmbeddingClass = EMBEDDING_MODELS[self.embedding_model_name]
            self.embedding_model = EmbeddingClass(**model_kwargs)
            logger.info(f"Initialized {self.embedding_model_name} embedding model")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            raise
        
        # Initialize vector store
        self.vector_store = None
```

#### 16.2.2 `_convert_chunks_to_documents(chunks)` Method

**Detailed Explanation:**

This private helper method performs a crucial transformation: it converts our internal `DocumentChunk` objects into LangChain's `Document` objects, which the vector database can process. This conversion involves several important steps:

1. **Metadata Extraction**: It extracts any existing metadata from the chunk.

2. **Unique ID Generation**: It assigns a unique UUID to each chunk, enabling tracking and later retrieval.

3. **Metadata Sanitization**: This is critical - ChromaDB requires all metadata values to be primitive types (string, int, float, bool). This method converts any complex types (like lists or dicts) into string representations.

4. **Verse List Serialization**: Special handling for the 'verses' field, which is originally a list but must be converted to a string for ChromaDB compatibility. The verses are joined with a separator that preserves their distinctiveness.

This conversion process addresses a subtle but critical challenge we faced during development: ChromaDB's strict metadata type requirements. Without this careful conversion, our earlier attempts at vectorization failed with type errors.

**Code Implementation:**

```python
def _convert_chunks_to_documents(self, chunks: List[DocumentChunk]) -> List[Document]:
    """
    Convert DocumentChunk objects to Langchain Document objects.
    
    Args:
        chunks: List of DocumentChunk objects
        
    Returns:
        List of Langchain Document objects
    """
    documents = []
    for chunk in chunks:
        # Get base metadata
        metadata = dict(chunk.metadata) if hasattr(chunk, 'metadata') else {}
        
        # Add chunk-specific metadata
        metadata['chunk_id'] = str(uuid.uuid4())
        metadata['has_verses'] = chunk.has_verses if hasattr(chunk, 'has_verses') else False
        
        # If chunk has verses, convert them to string format for ChromaDB compatibility
        if hasattr(chunk, 'verses') and chunk.verses:
            # Convert list of verses to a single string
            metadata['verses'] = ' | '.join(str(verse) for verse in chunk.verses)
            metadata['verse_count'] = len(chunk.verses)
        
        # Convert any remaining complex metadata to strings
        for key, value in metadata.items():
            if isinstance(value, (list, dict, tuple)):
                metadata[key] = str(value)
            elif not isinstance(value, (str, int, float, bool, type(None))):
                metadata[key] = str(value)
        
        # Create Document object
        doc = Document(
            page_content=chunk.content if hasattr(chunk, 'content') else str(chunk),
            metadata=metadata
        )
        documents.append(doc)
    
    return documents
```

#### 16.2.3 `vectorize_chunks(chunks, batch_size)` Method

**Detailed Explanation:**

This is the workhorse method that actually performs the embedding and storage of our document chunks. It handles the core functionality of transforming text into vector embeddings and saving them to our ChromaDB vector database.

The method implements several critical aspects of the embedding process:

1. **Batch Processing**: To prevent overwhelming the embedding API and to stay within token limits, this method processes chunks in controlled batches (default: 100 chunks per batch). This addresses a key challenge we encountered during development where large batches would trigger OpenAI API's token limits.

2. **Database Management**: The method intelligently handles both scenarios - creating a new vector database or appending to an existing one. It checks if a database already exists at the specified location and adapts its behavior accordingly.

3. **Incremental Persistence**: After each batch is processed, the method calls `persist()` on the vector store. This ensures that we don't lose progress if the process is interrupted, and provides checkpoint-style safety for long-running vectorization jobs.

4. **Error Handling**: The method wraps all operations in a try-except block, providing detailed error information if something goes wrong. This was crucial during development to diagnose issues with token limits, metadata formatting, and API connectivity.

This careful, batch-wise approach solved a critical challenge in our early development: attempting to vectorize all 25,000+ chunks at once would fail due to API rate limits and token constraints. By processing chunks incrementally with persistence after each batch, we achieved reliable, restartable embedding.

**Code Implementation:**

```python
def vectorize_chunks(self, chunks: List[DocumentChunk], batch_size: int = 100) -> Dict[str, Any]:
    """
    Vectorize document chunks and store them in the vector database.
    
    Args:
        chunks: List of DocumentChunk objects
        batch_size: Batch size for vectorization (reduced to avoid token limits)
        
    Returns:
        Dict with vectorization results
    """
    logger.info(f"Vectorizing {len(chunks)} document chunks in batches of {batch_size}")
    
    # Convert chunks to Langchain Document objects
    documents = self._convert_chunks_to_documents(chunks)
    
    try:
        # Initialize or load existing Chroma DB
        if os.path.exists(os.path.join(self.persist_directory, "chroma.sqlite3")):
            logger.info(f"Loading existing Chroma DB from {self.persist_directory}")
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model,
                collection_name=self.collection_name
            )
            # Process documents in batches to avoid token limits
            total_processed = 0
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
                self.vector_store.add_documents(batch)
                total_processed += len(batch)
                
                # Persist after each batch to avoid data loss
                self.vector_store.persist()
                
        else:
            # Create new DB with first batch
            logger.info(f"Creating new Chroma DB at {self.persist_directory}")
            first_batch = documents[:batch_size]
            self.vector_store = Chroma.from_documents(
                documents=first_batch,
                embedding=self.embedding_model,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
            self.vector_store.persist()
            total_processed = len(first_batch)
            logger.info(f"Created DB with first batch of {len(first_batch)} documents")
            
            # Process remaining documents in batches
            remaining_documents = documents[batch_size:]
            for i in range(0, len(remaining_documents), batch_size):
                batch = remaining_documents[i:i + batch_size]
                batch_num = (i // batch_size) + 2  # +2 because we already processed batch 1
                total_batches = ((len(documents) + batch_size - 1) // batch_size)
                logger.info(f"Processing batch {batch_num}/{total_batches}")
                
                self.vector_store.add_documents(batch)
                total_processed += len(batch)
                
                # Persist after each batch
                self.vector_store.persist()
        
        # Final persist
        self.vector_store.persist()
        logger.info(f"Successfully vectorized and stored {total_processed} documents")
        
        return {
            "status": "success",
            "chunks_processed": total_processed,
            "persist_directory": self.persist_directory,
            "collection_name": self.collection_name
        }
        
    except Exception as e:
        logger.error(f"Error vectorizing chunks: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "chunks_processed": 0
        }
```

#### 16.2.4 `similarity_search(query, k)` Method

**Detailed Explanation:**

This method provides a straightforward way to search our vector database for documents similar to a given query. It's a convenience wrapper that handles the common case of finding the top-k most similar chunks to a query string.

The method handles several important details:

1. **Database Loading**: It checks if the vector store is already loaded, and if not, loads it from disk. This makes the method safe to use even if the `DocumentVectorizer` instance was just created.

2. **Error Handling**: It gracefully handles potential errors during search operations, returning an empty list rather than crashing if something goes wrong.

3. **Result Count Control**: The `k` parameter allows the caller to specify exactly how many similar documents to return, supporting different retrieval needs.

This method serves as a simple entry point for basic semantic search operations, abstracting away the details of vector database initialization and query processing.

**Code Implementation:**

```python
def similarity_search(self, query: str, k: int = 5) -> List[Document]:
    """
    Perform similarity search on the vector store.
    
    Args:
        query: Query text
        k: Number of results to return
        
    Returns:
        List of Documents most similar to query
    """
    if not self.vector_store:
        if not self.load_vector_store():
            logger.error("Cannot perform search: Vector store not loaded")
            return []
    
    try:
        results = self.vector_store.similarity_search(query, k=k)
        return results
    except Exception as e:
        logger.error(f"Error performing similarity search: {str(e)}")
        return []
```

#### 16.2.5 Factory Function: `create_vectorizer`

**Detailed Explanation:**

This factory function provides a simple way to create a `DocumentVectorizer` with the appropriate embedding model. It serves as a convenient entry point for other parts of the application to work with document vectorization without needing to deal with the details of the `DocumentVectorizer` class.

By centralizing vectorizer creation in this function, we gain several benefits:

1. **Consistency**: All parts of the application create vectorizers with the same settings.

2. **Configuration**: The function provides sensible defaults while allowing customization of key parameters.

3. **Dependency Injection**: It makes it easier to swap out embedding models or vector stores in the future.

This pattern of using factory functions for complex object creation is a best practice that makes the code more maintainable and testable.

**Code Implementation:**

```python
# Factory function to create vectorizer with appropriate model
def create_vectorizer(persist_directory: str, 
                     model_name: str = "openai",
                     collection_name: str = "spiritual_texts",
                     **kwargs) -> DocumentVectorizer:
    """
    Create a document vectorizer with the specified embedding model.
    
    Args:
        persist_directory: Directory to persist vector database
        model_name: Name of embedding model ('openai', 'huggingface')
        collection_name: Name of the ChromaDB collection
        **kwargs: Additional keyword arguments for the embedding model
        
    Returns:
        DocumentVectorizer instance
    """
    return DocumentVectorizer(
        persist_directory=persist_directory,
        embedding_model=model_name,
        embedding_model_kwargs=kwargs,
        collection_name=collection_name
    )
```

### 16.3 Query Processing – `utils/query_utils.py`

The Query Processing module is responsible for handling user questions, converting them to embeddings, and retrieving relevant documents from our vector store. This is where the "Retrieval" part of our RAG pipeline happens.

#### 16.3.1 `QueryProcessor` Class

**Detailed Explanation:**

The `QueryProcessor` class encapsulates the functionality for processing user queries and retrieving relevant chunks from our vector database. It serves as an abstraction layer between the application and the vector database, providing methods for both standard similarity search and more advanced retrieval techniques.

This class is designed to handle the complexities of:

1. Converting user queries into vector embeddings
2. Searching the vector database efficiently 
3. Supporting different retrieval strategies like MMR (Maximum Marginal Relevance)
4. Formatting retrieved chunks for use in LLM prompts

By centralizing this functionality, we ensure consistent query processing across the application and make it easier to experiment with different retrieval strategies.

**Initialization Code:**

```python
class QueryProcessor:
    """Class for processing queries and retrieving relevant chunks."""
    
    def __init__(self, 
                 vector_store_dir: str,
                 embedding_model: str = "openai",
                 collection_name: str = "spiritual_texts"):
        """
        Initialize query processor.
        
        Args:
            vector_store_dir: Directory containing vector store
            embedding_model: Name of embedding model ('openai' or 'huggingface')
            collection_name: Name of the ChromaDB collection
        """
        self.vector_store_dir = vector_store_dir
        self.embedding_model_name = embedding_model
        self.collection_name = collection_name
        
        # Initialize embedding model
        if embedding_model not in EMBEDDING_MODELS:
            logger.warning(f"Unknown embedding model: {embedding_model}. Defaulting to OpenAI.")
            self.embedding_model_name = "openai"
            
        # Default kwargs for embedding models
        default_kwargs = {
            "openai": {"model": "text-embedding-ada-002"},
            "huggingface": {"model_name": "sentence-transformers/all-MiniLM-L6-v2"}
        }
        
        # Initialize the embedding model
        try:
            EmbeddingClass = EMBEDDING_MODELS[self.embedding_model_name]
            self.embedding_model = EmbeddingClass(**default_kwargs.get(self.embedding_model_name, {}))
            logger.info(f"Initialized {self.embedding_model_name} embedding model for query processing")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            raise
        
        # Initialize vector store
        self.vector_store = self._initialize_vector_store()
```

#### 16.3.2 `process_query(query_text)` Method

**Detailed Explanation:**

This method is responsible for converting a user's natural language question into a vector embedding that can be used for similarity search. It's a crucial step in the RAG pipeline because the quality of the embedding directly impacts the relevance of the retrieved documents.

The method works by:

1. Taking a raw text query from the user interface
2. Passing it to the embedding model (OpenAI's `text-embedding-ada-002` by default)
3. Receiving back a high-dimensional vector representation of the query's meaning

While the implementation is simple, this step is where the "magic" of semantic search happens - the embedding model encodes the conceptual meaning of the question into a mathematical form that can be compared to our document embeddings.

A key point to understand is that this embedding process is completely independent of the document embeddings - it happens at query time. The same embedding model must be used for both documents and queries to ensure compatibility in the vector space.

**Code Implementation:**

```python
def process_query(self, query_text: str) -> List[float]:
    """
    Process a query into an embedding vector.
    
    Args:
        query_text: Query text
        
    Returns:
        List[float]: Embedding vector
    """
    try:
        # Get embedding for the query text
        query_embedding = self.embedding_model.embed_query(query_text)
        logger.info(f"Processed query: '{query_text[:50]}...'")
        return query_embedding
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return []
```

#### 16.3.3 `retrieve_relevant_chunks(query_text, k)` Method

**Detailed Explanation:**

This method implements standard similarity search to find the most relevant chunks for a given query. It forms the foundation of our retrieval strategy and is used when we want the most semantically similar documents without considering diversity.

The method performs these steps:

1. Ensures the vector store is properly initialized
2. Performs a similarity search using the query text
3. Formats the results into a consistent structure that includes both content and metadata

The similarity search works by:
- Converting the query text into an embedding vector (via the embedding model)
- Computing cosine similarity between this query vector and all document vectors in the database
- Returning the top-k documents with the highest similarity scores

This is the most direct retrieval approach, favoring documents that are semantically closest to the query without considering the diversity of the result set. It's most effective when the user wants focused, highly relevant answers from a limited number of sources.

**Code Implementation:**

```python
def retrieve_relevant_chunks(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve chunks relevant to a query using similarity search.
    
    Args:
        query_text: Query text
        k: Number of chunks to retrieve
        
    Returns:
        List of dictionaries containing chunk content and metadata
    """
    if not self.vector_store:
        logger.error("Cannot retrieve chunks: Vector store not initialized")
        return []
        
    try:
        # Perform similarity search
        results = self.vector_store.similarity_search(query_text, k=k)
        logger.info(f"Retrieved {len(results)} chunks using similarity search")
        
        # Format results
        formatted_chunks = []
        for doc in results:
            formatted_chunks.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
            
        return formatted_chunks
    
    except Exception as e:
        logger.error(f"Error retrieving chunks: {str(e)}")
        return []
```

#### 16.3.4 `mmr_retrieval(query_text, k, fetch_k, diversity)` Method

**Detailed Explanation:**

This method implements Maximum Marginal Relevance (MMR) retrieval, which balances relevance with diversity in the result set. It's a more sophisticated retrieval strategy that addresses a common problem with standard similarity search: returning redundant documents that contain similar information.

MMR works through these steps:

1. First retrieves a larger pool of candidate documents (`fetch_k` > `k`)
2. Then selects a subset of `k` documents that maximize both:
   - Relevance to the query
   - Diversity compared to already-selected documents

The `diversity` parameter (lambda_mult) controls the trade-off between relevance and diversity:
- Value close to 0: Emphasizes diversity over relevance
- Value close to 1: Emphasizes relevance over diversity

This retrieval method is particularly valuable in our spiritual Q&A system because it can present the user with different perspectives or interpretations of a concept, rather than multiple chunks saying essentially the same thing. For example, when asking about "karma", MMR might return passages from different texts rather than multiple passages from the same text.

**Code Implementation:**

```python
def mmr_retrieval(self, query_text: str, k: int = 5, fetch_k: int = 15, diversity: float = 0.7) -> List[Dict[str, Any]]:
    """
    Retrieve chunks using Maximum Marginal Relevance (MMR) for diversity.
    
    Args:
        query_text: Query text
        k: Number of chunks to retrieve
        fetch_k: Number of candidates to consider before filtering
        diversity: Controls trade-off between relevance and diversity (0-1)
            - 0: Maximum diversity
            - 1: Maximum relevance
        
    Returns:
        List of dictionaries containing chunk content and metadata
    """
    if not self.vector_store:
        logger.error("Cannot retrieve chunks: Vector store not initialized")
        return []
        
    try:
        # Perform MMR search
        results = self.vector_store.max_marginal_relevance_search(
            query_text, 
            k=k, 
            fetch_k=fetch_k, 
            lambda_mult=diversity
        )
        logger.info(f"Retrieved {len(results)} chunks using MMR (diversity={diversity})")
        
        # Format results
        formatted_chunks = []
        for doc in results:
            formatted_chunks.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
            
        return formatted_chunks
    
    except Exception as e:
        logger.error(f"Error retrieving chunks with MMR: {str(e)}")
        return []
```

#### 16.3.5 Factory Function: `create_query_processor`

**Detailed Explanation:**

Similar to our vectorizer factory, this function provides a convenient way to create a `QueryProcessor` with appropriate settings. It follows the factory pattern to abstract away the details of object creation and provide a simple interface for the rest of the application.

This factory function maintains consistency across the application and makes it easier to experiment with different embedding models or collection names in the future.

**Code Implementation:**

```python
def create_query_processor(vector_store_dir: str,
                           embedding_model: str = "openai",
                           collection_name: str = "spiritual_texts") -> QueryProcessor:
    """
    Create a query processor for the specified vector store.
    
    Args:
        vector_store_dir: Directory containing vector store
        embedding_model: Name of embedding model ('openai' or 'huggingface')
        collection_name: Name of the ChromaDB collection
        
    Returns:
        QueryProcessor instance
    """
    return QueryProcessor(
        vector_store_dir=vector_store_dir,
        embedding_model=embedding_model,
        collection_name=collection_name
    )
```

### 16.4 Agent Tools Layer – `utils/query_agent_tools.py`

The Agent Tools Layer integrates our retrieval functionality with the CrewAI and LangChain frameworks, allowing AI agents to interact with our document database through well-defined tool interfaces. This abstraction enables our agents to reason about spiritual texts and perform complex tasks without needing to understand the underlying vector database or embedding mechanics.

#### 16.4.1 `SpiritualDocumentRetriever` Class

**Detailed Explanation:**

The `SpiritualDocumentRetriever` class serves as the primary interface between our agents and the vector database. It wraps the `QueryProcessor` functionality we just covered, providing a high-level abstraction specifically designed for retrieving spiritual texts.

This class is critically important in our architecture because:

1. **Unified Retrieval Interface**: It provides a consistent way for agents to access document chunks, regardless of the underlying retrieval method (similarity search or MMR).

2. **Reading Style Management**: It translates the user's selected reading style ("Deep", "Balanced", "Broad") into appropriate technical parameters for the retrieval process.

3. **Result Formatting**: It transforms raw chunks from the database into a format that's optimized for LLM comprehension, ensuring that the context provided to the LLM is well-structured.

4. **Resource Management**: It handles the initialization and lifecycle of the vector database connection, ensuring efficient resource usage.

Under the hood, this class maintains a `QueryProcessor` instance and exposes methods that translate high-level retrieval requests into specific database queries. It's designed to be instantiated once and reused across multiple queries, maintaining the database connection for efficiency.

**Initialization Code:**

```python
class SpiritualDocumentRetriever:
    """Retriever for spiritual documents from vector store."""
    
    def __init__(self, vector_store_dir: str):
        """
        Initialize the spiritual document retriever.
        
        Args:
            vector_store_dir: Directory containing the vector store
        """
        self.vector_store_dir = vector_store_dir
        self.query_processor = create_query_processor(vector_store_dir)
        
        # Default retrieval parameters
        self.default_params = {
            "deep": {"k": 5, "fetch_k": 15, "diversity": 0.9},  # High relevance, low diversity
            "balanced": {"k": 5, "fetch_k": 15, "diversity": 0.7},  # Balanced
            "broad": {"k": 5, "fetch_k": 15, "diversity": 0.3},   # High diversity, lower relevance
        }
        
        logger.info(f"Initialized SpiritualDocumentRetriever with vector store at {vector_store_dir}")
```

#### 16.4.2 `get_relevant_documents` Method

**Detailed Explanation:**

This is the core retrieval method that our agents use to find relevant spiritual texts for a given query. It's designed to be flexible, supporting different reading styles and retrieval approaches while maintaining a simple interface.

The method works through these steps:

1. **Reading Style Interpretation**: It translates the user's reading style preference ("Deep", "Balanced", "Broad") into concrete retrieval parameters.

2. **Retrieval Strategy Selection**: Based on the reading style and diversity parameter, it decides whether to use standard similarity search or MMR retrieval.

3. **Query Execution**: It calls the appropriate `QueryProcessor` method to actually perform the retrieval operation.

4. **Result Formatting**: It standardizes the returned documents into a consistent format, including content, metadata, and verse information.

The `diversity` parameter is especially important as it controls the trade-off between relevance and diversity in the retrieved documents. For "Deep" reading, we use high values (close to 1.0) to prioritize relevance; for "Broad" reading, we use lower values to encourage more diverse results.

This method embodies our philosophy of presenting spiritual knowledge in a way that respects the user's preferred learning style - whether they want focused, in-depth information or a broader overview of different perspectives.

**Code Implementation:**

```python
def get_relevant_documents(self, query: str, reading_style: str = "balanced", diversity: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Retrieve relevant documents for a query based on reading style.
    
    Args:
        query: The query to search for
        reading_style: Reading style ('deep', 'balanced', 'broad')
        diversity: Optional override for diversity parameter
        
    Returns:
        List of documents with content and metadata
    """
    # Normalize reading style
    reading_style = reading_style.lower()
    if reading_style not in self.default_params:
        logger.warning(f"Unknown reading style: {reading_style}. Using balanced.")
        reading_style = "balanced"
    
    # Get parameters for the reading style
    params = self.default_params[reading_style].copy()
    
    # Override diversity if provided
    if diversity is not None:
        params["diversity"] = float(diversity)
    
    # For deep reading style or when diversity is very high (>0.95), use pure similarity search
    # This gives the most relevant results without considering diversity
    if reading_style == "deep" or (diversity is not None and diversity > 0.95):
        logger.info(f"Using similarity search for query: '{query[:50]}...'")
        return self.query_processor.retrieve_relevant_chunks(query, k=params["k"])
    
    # For other reading styles, use MMR with appropriate diversity setting
    else:
        logger.info(f"Using MMR search for query with diversity={params['diversity']}: '{query[:50]}...'")
        return self.query_processor.mmr_retrieval(
            query,
            k=params["k"],
            fetch_k=params["fetch_k"],
            diversity=params["diversity"]
        )
```

#### 16.4.3 `SpiritualDocumentSearch` Tool Class

**Detailed Explanation:**

This class implements the LangChain Tool interface, making our document retrieval functionality available as a tool that AI agents can use in their reasoning process. It's a crucial bridge between our retrieval system and the agentic framework.

When an agent needs to find relevant spiritual information, it can invoke this tool by name ("search_for_relevant_info"), providing a query and optionally specifying a reading style. The tool handles the details of retrieval and returns the results in a format that's optimized for the agent to understand and use.

The implementation follows the LangChain BaseTool pattern, which requires:
- A descriptive name for the tool
- A detailed description of what the tool does (for the agent to understand when to use it)
- An implementation of the `_run` method that executes the tool's functionality

This abstraction allows our agents to think in terms of high-level actions like "search for information about karma" rather than the low-level details of vector database queries and embedding calculations.

**Code Implementation:**

```python
class SpiritualDocumentSearch(BaseTool):
    """Tool for searching spiritual documents."""
    
    name = "search_for_relevant_info"
    description = "Searches for relevant information in spiritual texts based on a query. Use this tool when you need to find specific information to answer questions about spiritual topics."
    
    retriever: SpiritualDocumentRetriever
    
    def _run(self, query: str, reading_style: str = "balanced") -> str:
        """
        Run the tool to search for relevant documents.
        
        Args:
            query: The query to search for
            reading_style: Reading style ('deep', 'balanced', 'broad')
            
        Returns:
            Formatted string with relevant document chunks
        """
        documents = self.retriever.get_relevant_documents(query, reading_style)
        
        if not documents:
            return "No relevant information found."
        
        # Format results for the agent
        result_str = f"Found {len(documents)} relevant chunks:\n\n"
        
        for i, doc in enumerate(documents):
            # Get metadata fields
            source = doc.get("metadata", {}).get("source", "Unknown source")
            title = doc.get("metadata", {}).get("title", "Unknown title")
            
            # Add chunk info
            result_str += f"Chunk {i+1}:\n"
            result_str += f"Source: {source}\n"
            result_str += f"Title: {title}\n"
            result_str += f"Content:\n{doc.get('content', '')}\n\n"
            
            # Add verse info if available
            verses = doc.get("metadata", {}).get("verses", "")
            if verses:
                result_str += f"Verses: {verses}\n\n"
            
        return result_str
```

#### 16.4.4 `ExtractVersesFromChunksTool` Class

**Detailed Explanation:**

This specialized tool helps our agents extract and work with specific verses from spiritual texts. When answering questions about particular verses, agents need to be able to isolate those verses from the broader context in which they appear.

The tool works by:

1. Taking a chunk of text (which may contain multiple verses and explanations)
2. Using regex patterns to identify and extract the specific verse numbers and text mentioned
3. Returning a structured representation of just the verses

This functionality is particularly valuable when users ask questions like "What does verse 2.47 of the Bhagavad Gita mean?" or "Can you explain Gita verse 4.7-8?". In these cases, the agent needs to first extract the exact verses before providing an interpretation.

By having this functionality available as a tool, the agent can reason about verses as specific entities rather than just handling undifferentiated text. This leads to more precise and informative answers about spiritual texts.

**Code Implementation:**

```python
class ExtractVersesFromChunksTool(BaseTool):
    """Tool for extracting verses from document chunks."""
    
    name = "extract_verses"
    description = "Extracts specific verses from spiritual text chunks. Use this when you need to isolate and present specific verses mentioned in the retrieved documents."
    
    def _run(self, text: str) -> str:
        """
        Extract verses from text.
        
        Args:
            text: Text containing verses
            
        Returns:
            Extracted verses as formatted string
        """
        verses = extract_verses(text)
        
        if not verses:
            return "No verses found in the provided text."
        
        result = f"Extracted {len(verses)} verses:\n\n"
        for i, verse in enumerate(verses):
            result += f"Verse {i+1}: {verse.strip()}\n\n"
        
        return result
```

### 16.5 High-Level Retriever – `document_retriever.py`

The High-Level Retriever module serves as the main integration point between our vector database and the API layer. It provides a clean, simplified interface for retrieving contextually relevant information from our spiritual texts.

#### 16.5.1 `DocumentRetriever` Class

**Detailed Explanation:**

The `DocumentRetriever` class is a façade that shields the API endpoints from the complexities of vector store operations. This architectural pattern is crucial for maintaining clean separation of concerns in our application.

This class has several important responsibilities:

1. **Abstraction**: It hides the implementation details of vector databases, embeddings, and retrieval algorithms behind a simple interface.

2. **Configuration Management**: It handles the loading and application of retrieval settings based on user preferences.

3. **Reading Style Translation**: It maps user-friendly reading style concepts ("Deep", "Balanced", "Broad") to technical retrieval parameters.

4. **Result Formatting**: It transforms raw vector search results into structured context blocks that can be directly inserted into LLM prompts.

The `DocumentRetriever` serves as the highest level of abstraction in our RAG pipeline, providing exactly what the API needs without exposing unnecessary complexity.

**Initialization Code:**

```python
class DocumentRetriever:
    """High-level retriever for getting contextual information for questions."""
    
    def __init__(self, vector_store_dir: str = "./chroma_db"):
        """
        Initialize the document retriever.
        
        Args:
            vector_store_dir: Directory containing vector store
        """
        self.vector_store_dir = vector_store_dir
        
        # Initialize query processor with appropriate settings
        self.query_processor = create_query_processor(
            vector_store_dir=vector_store_dir,
            embedding_model="openai",
            collection_name="spiritual_texts"
        )
        
        # Define reading styles and their parameters
        self.reading_styles = {
            "deep": {"use_mmr": False, "k": 5},            # Pure similarity search
            "balanced": {"use_mmr": True, "k": 5, "fetch_k": 15, "diversity": 0.7},  # Balanced
            "broad": {"use_mmr": True, "k": 5, "fetch_k": 15, "diversity": 0.3}     # High diversity
        }
        
        logger.info(f"Initialized DocumentRetriever with vector store at {vector_store_dir}")
```

#### 16.5.2 `get_context_for_question(question, style)` Method

**Detailed Explanation:**

This is the primary method that the API calls to retrieve relevant context for a user's question. It's designed to be flexible yet simple, handling the complexities of retrieval while providing results in a format ready for prompt construction.

The method performs several key steps:

1. **Reading Style Interpretation**: It normalizes and validates the user's reading style choice, defaulting to "Balanced" if an invalid style is provided.

2. **Retrieval Strategy Selection**: Based on the reading style, it chooses between standard similarity search (for "Deep" reading) or MMR retrieval (for "Balanced" or "Broad" reading).

3. **Chunk Retrieval**: It calls the appropriate query processor method to retrieve relevant document chunks.

4. **Context Formatting**: It transforms the retrieved chunks into a formatted string that can be directly injected into the LLM prompt template. This includes metadata like source and verse information.

This method embodies our philosophy of making spiritual knowledge accessible according to the user's preferred learning style - whether they want focused, in-depth information from similar sources (Deep), or a broader overview of different perspectives (Broad).

**Code Implementation:**

```python
def get_context_for_question(self, question: str, style: str = "Balanced") -> str:
    """
    Get formatted context for a question based on reading style.
    
    Args:
        question: The user's question
        style: Reading style ('Deep', 'Balanced', 'Broad')
        
    Returns:
        Formatted context string ready for prompt insertion
    """
    # Normalize and validate reading style
    style = style.lower()
    if style not in self.reading_styles:
        logger.warning(f"Unknown reading style: {style}. Using 'balanced'.")
        style = "balanced"
    
    # Get retrieval parameters for the style
    params = self.reading_styles[style]
    
    # Retrieve chunks based on style
    logger.info(f"Retrieving context for question with style: {style}")
    
    if params.get("use_mmr", False):
        # Use MMR for balanced or broad styles
        chunks = self.query_processor.mmr_retrieval(
            query_text=question,
            k=params.get("k", 5),
            fetch_k=params.get("fetch_k", 15),
            diversity=params.get("diversity", 0.7)
        )
    else:
        # Use similarity search for deep style
        chunks = self.query_processor.retrieve_relevant_chunks(
            query_text=question,
            k=params.get("k", 5)
        )
    
    # Format the retrieved chunks into context
    if not chunks:
        logger.warning("No relevant chunks found for question.")
        return "No relevant information found."
    
    # Build formatted context string
    context_blocks = []
    for i, chunk in enumerate(chunks):
        # Get metadata
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", "Unknown")
        title = metadata.get("title", "Unknown")
        
        # Format chunk with metadata
        block = f"[Source: {source}]"
        if title and title != "Unknown":
            block += f" [Title: {title}]"
        
        # Add content
        block += f"\n{chunk.get('content', '').strip()}"
        
        # Add verses if available
        verses = metadata.get("verses", "")
        if verses and verses != "[]":
            block += f"\n[Verses: {verses}]"
        
        context_blocks.append(block)
    
    # Join all blocks with separators
    formatted_context = "\n\n---\n\n".join(context_blocks)
    
    logger.info(f"Retrieved {len(chunks)} chunks for context")
    return formatted_context
```

#### 16.5.3 Example Usage in Application

**Detailed Explanation:**

This section demonstrates how the `DocumentRetriever` is used in practice within our application. It shows the typical patterns for creating a retriever and using it to get context for user questions.

The examples illustrate:

1. **Basic Retriever Creation**: How to instantiate a retriever with default settings.

2. **Different Reading Styles**: How to retrieve context with different reading styles to meet different user needs.

3. **Context Integration**: How the retrieved context flows into prompt construction and ultimately to answer generation.

These examples serve as a reference for developers to understand how the retriever fits into the application's workflow.

**Code Examples:**

```python
# Example 1: Basic retriever creation and usage
retriever = DocumentRetriever(vector_store_dir="./chroma_db")
context = retriever.get_context_for_question(
    question="What is the concept of karma in Hinduism?",
    style="Balanced"
)

# Example 2: Using Deep reading style for focused results
deep_context = retriever.get_context_for_question(
    question="What does Bhagavad Gita verse 2.47 teach about karma yoga?",
    style="Deep"
)

# Example 3: Using Broad reading style for diverse perspectives
broad_context = retriever.get_context_for_question(
    question="What are the different interpretations of karma?",
    style="Broad"
)

# Example 4: Integration with prompt construction and LLM
context = retriever.get_context_for_question(user_question, user_style)
prompt = build_prompt(context, user_question)  # Construct prompt with context
answer = llm_client.generate(prompt)           # Generate answer using LLM
```

### 16.6 Putting It Together in FastAPI – `api/spiritual_api.py`

The FastAPI module is where all our components come together to form a coherent web service. This is the entry point for user requests and the orchestration layer that coordinates the RAG pipeline components to generate responses to spiritual questions.

#### 16.6.1 API Setup and Configuration

**Detailed Explanation:**

The setup portion of our FastAPI module establishes the web service, configures Cross-Origin Resource Sharing (CORS), and initializes the necessary components for handling requests. This foundational code is crucial for creating a secure, accessible API that can be called from our frontend application.

The setup performs several key tasks:

1. **API Creation**: Instantiates the FastAPI application with appropriate metadata and settings.

2. **CORS Configuration**: Sets up CORS policies to allow requests from our frontend origin, ensuring security while enabling functionality.

3. **Middleware Setup**: Adds middleware for request logging, error handling, and other cross-cutting concerns.

4. **Environment Loading**: Loads necessary environment variables for API keys and service configuration.

This setup code runs once when the API service starts, establishing the foundation for handling all incoming requests.

**Code Implementation:**

```python
# Import necessary modules
import os
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import application components
from document_retriever import DocumentRetriever
from utils.llm_client import LLMClient
from utils.prompt_templates import build_prompt
from utils.config import get_config

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Spiritual Q&A API",
    description="API for answering questions about spiritual texts",
    version="1.0.0"
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the actual origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get configuration
config = get_config()

# Global variables for caching
_retriever_cache = {}
_llm_client_cache = {}
```

#### 16.6.2 Request and Response Models

**Detailed Explanation:**

These Pydantic models define the structure of incoming requests and outgoing responses for our API. Using strong typing with Pydantic provides several benefits:

1. **Automatic Validation**: Incoming requests are automatically validated against the schema, ensuring they contain the required fields in the correct format.

2. **Documentation Generation**: FastAPI uses these models to automatically generate OpenAPI documentation, making it easier for frontend developers to understand the API.

3. **Type Safety**: The explicit typing helps catch errors at development time rather than runtime.

4. **Response Consistency**: By defining response models, we ensure that our API returns consistent, predictable data structures.

These models act as a contract between the frontend and backend, ensuring clear communication between the two systems.

**Code Implementation:**

```python
# Request and response models
class QueryRequest(BaseModel):
    """Model for incoming query requests."""
    query: str
    style: str = "Balanced"  # Default to balanced reading style
    model: str = "gpt-4"     # Default LLM model

class QueryResponse(BaseModel):
    """Model for query responses."""
    answer: str
    context_used: Optional[str] = None  # Optional for debugging
    metadata: Dict[str, Any] = {}       # Additional metadata about the response
```

#### 16.6.3 Helper Functions

**Detailed Explanation:**

These helper functions support our API endpoints by managing resources efficiently and providing reusable functionality. They include:

1. **Resource Caching**: The `_get_retriever` and `_get_llm_client` functions implement a caching pattern to avoid recreating expensive resources for each request.

2. **Dependency Injection**: These functions are used with FastAPI's dependency injection system, making them available to endpoints without repeating code.

3. **Configuration Management**: They handle the details of creating properly configured components based on environment settings.

4. **Error Handling**: They include appropriate error handling and logging to ensure robust operation.

By centralizing these concerns in helper functions, we keep our endpoint handlers clean and focused on their primary responsibility.

**Code Implementation:**

```python
def _get_retriever(model: str = "gpt-4") -> DocumentRetriever:
    """
    Get or create a DocumentRetriever instance with caching.
    
    Args:
        model: LLM model identifier (used as cache key)
        
    Returns:
        DocumentRetriever instance
    """
    global _retriever_cache
    
    if model not in _retriever_cache:
        logger.info(f"Creating new DocumentRetriever for model: {model}")
        vector_store_dir = config.get("VECTOR_DB_PATH", "./chroma_db")
        _retriever_cache[model] = DocumentRetriever(vector_store_dir=vector_store_dir)
    
    return _retriever_cache[model]

def _get_llm_client(model: str = "gpt-4") -> LLMClient:
    """
    Get or create an LLMClient instance with caching.
    
    Args:
        model: LLM model identifier
        
    Returns:
        LLMClient instance
    """
    global _llm_client_cache
    
    if model not in _llm_client_cache:
        logger.info(f"Creating new LLMClient for model: {model}")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
            
        _llm_client_cache[model] = LLMClient(
            api_key=api_key,
            model=model
        )
    
    return _llm_client_cache[model]
```

#### 16.6.4 `/query` Endpoint

**Detailed Explanation:**

The `/query` endpoint is the heart of our API, bringing together all the components of our RAG pipeline to answer user questions. This is where the entire flow we've been documenting comes together:

1. **Request Processing**: The endpoint receives and validates the user's question and preferences.

2. **Resource Acquisition**: It gets the necessary components (retriever, LLM client) using our helper functions.

3. **Context Retrieval**: It retrieves relevant context from our vector database using the appropriate reading style.

4. **Prompt Construction**: It builds a prompt that combines the user's question with the retrieved context.

5. **LLM Generation**: It sends the prompt to the language model to generate an answer.

6. **Response Formatting**: It formats the response according to our API contract and returns it to the user.

7. **Logging and Monitoring**: It logs key information for monitoring and debugging.

This endpoint represents the culmination of our entire RAG pipeline, connecting the user's question to our document retrieval system and language model to provide meaningful, contextually informed answers.

**Code Implementation:**

```python
@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """
    Process a query about spiritual texts and return an answer.
    
    Args:
        req: QueryRequest containing the query, style, and model
        
    Returns:
        QueryResponse with the answer and metadata
    """
    logger.info(f"Received query: '{req.query[:50]}...' with style: {req.style}")
    
    try:
        # Get retriever and LLM client
        retriever = _get_retriever(req.model)
        llm_client = _get_llm_client(req.model)
        
        # Retrieve relevant context
        context = retriever.get_context_for_question(req.query, style=req.style)
        logger.info(f"Retrieved context ({len(context)} chars)")
        
        # Build prompt with context and query
        prompt = build_prompt(context, req.query)
        logger.info(f"Built prompt ({len(prompt)} chars)")
        
        # Generate answer using LLM
        answer = llm_client.chat(prompt)
        logger.info(f"Generated answer ({len(answer)} chars)")
        
        # Return response with metadata
        return QueryResponse(
            answer=answer,
            context_used=context if config.get("DEBUG_MODE", False) else None,
            metadata={
                "model": req.model,
                "style": req.style,
                "context_length": len(context),
                "prompt_length": len(prompt)
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

#### 16.6.5 `/health` Endpoint

**Detailed Explanation:**

The health endpoint provides a simple way to verify that our API is running and accessible. While seemingly simple, this endpoint serves several important purposes:

1. **Load Balancer Checks**: Enables load balancers to determine if the service is healthy.

2. **Deployment Verification**: Provides a quick way to verify that a new deployment is working.

3. **Monitoring Integration**: Can be used by monitoring systems to track service availability.

4. **Connection Testing**: Allows frontend developers to verify connectivity to the backend.

By including basic system information in the response, the endpoint also provides useful diagnostic information without exposing sensitive details.

**Code Implementation:**

```python
@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    
    Returns:
        Dictionary with status information
    """
    return {
        "status": "healthy",
        "api_version": app.version,
        "vector_db_path": config.get("VECTOR_DB_PATH", "./chroma_db"),
        "default_model": config.get("DEFAULT_MODEL", "gpt-4")
    }
```

#### 16.6.6 Running the API Server

**Detailed Explanation:**

This code section starts the FastAPI server using Uvicorn when the script is run directly. It's crucial for both development and production deployments, setting up the server with appropriate host, port, and worker configurations.

The conditional execution (`if __name__ == "__main__":`) ensures that the server only starts when the script is run directly, not when it's imported as a module. This pattern allows the same file to be used both as an application entry point and as a module imported by other scripts.

**Code Implementation:**

```python
if __name__ == "__main__":
    import uvicorn
    
    # Get host and port from environment or use defaults
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    # Start server
    logger.info(f"Starting Spiritual Q&A API server at http://{host}:{port}")
    uvicorn.run(
        "spiritual_api:app",
        host=host,
        port=port,
        reload=config.get("DEBUG_MODE", False)  # Enable hot reload in debug mode
    )
    logger.info("Server stopped")
```

---

## Glossary of Key Terms

| Term | Plain-English Meaning |
|------|----------------------|
| LLM | Large Language Model — the AI you ask the question to (e.g., GPT-4.1). |
| Embedding | A numeric vector representation of text that lets us measure similarity. |
| Vector Store / ChromaDB | A specialised database that stores embeddings for fast similarity search. |
| MMR (Max-Marginal-Relevance) | A retrieval algorithm that mixes relevance with diversity so answers don’t repeat themselves. |
| FastAPI | A Python library we use to expose backend endpoints (`/query`). |
| CrewAI | An *agentic* orchestration framework coordinating task-specific agents. |
| OCR | Optical Character Recognition — turning images of text into real text. |
| Tesseract | The open-source engine performing OCR during PDF correction. |
| Uvicorn | An ASGI server that actually runs our FastAPI app in production. |

---

With these auxiliary pipelines documented, the **Spiritual Q&A System** end-to-end documentation is complete. Feel free to request more depth on any module or new features.
