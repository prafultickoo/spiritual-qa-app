# Spiritual Q&A System: Detailed Backend Flow (Part 1)

This document provides an extremely granular breakdown of the backend flow for the Spiritual Q&A system, from the moment a user submits a question to when they receive an answer. Each step and sub-step is explained in detail, followed by the exact code that implements it.

## Table of Contents
1. [Frontend Event Capture](#1-frontend-event-capture)
2. [Frontend Parameter Processing](#2-frontend-parameter-processing)
3. [API Request Construction](#3-api-request-construction)
4. [API Request Transmission](#4-api-request-transmission)

## 1. Frontend Event Capture

### Detailed Explanation
When the user clicks the "Seek Wisdom" button on the frontend interface, the following process occurs:

1. The browser's JavaScript event system detects the form submission event (`submit`).
2. The event is passed to the event listener that was registered during app initialization.
3. The default form submission behavior (page reload) is immediately prevented with `e.preventDefault()`.
4. The system checks if a query is already in progress by examining the `isLoading` flag.
5. If another query is in progress (isLoading is true), the function exits early to prevent duplicate submissions.
6. The system then extracts three critical pieces of information:
   - The question text from the `spiritual-question` input field, with whitespace trimmed from both ends
   - The selected LLM model from the `llm-model` dropdown
   - The reading style preference from the active radio button in the `reading-style` group

7. The system performs validation on the question:
   - If the question is empty after trimming, it displays a warning notification
   - If empty, the function exits early with a return statement

8. Once validation passes, the system sets the `isLoading` flag to true, which:
   - Prevents duplicate submissions
   - Will be used to display a loading indicator to the user
   - Ensures the interface accurately reflects the processing state

9. A try/finally block is established to ensure the loading state is always reset when the operation completes (whether successful or failed).

### Implementing Code
```javascript
// This event listener is registered during app initialization in setupEventListeners()
const questionForm = document.getElementById('question-form');
if (questionForm) {
    questionForm.addEventListener('submit', (e) => this.handleQuestionSubmit(e));
}

// The actual event handler that processes the form submission
async handleQuestionSubmit(e) {
    // 1. Prevent the default form submission (page reload)
    e.preventDefault();
    
    // 2. Check if another query is already in progress
    if (this.isLoading) return;
    
    // 3. Extract user inputs from the form elements
    const question = document.getElementById('spiritual-question').value.trim();
    const model = document.getElementById('llm-model').value;
    const readingStyle = document.querySelector('input[name="reading-style"]:checked').value;
    
    // 4. Validate that a question was entered
    if (!question) {
        this.showNotification('Please enter a spiritual question', 'warning');
        return;
    }
    
    // 5. Set loading state to true (activates spinner, disables form)
    this.isLoading = true;
    this.setLoadingState(true);  // This updates UI elements to reflect loading state
    
    // 6. Wrap in try/finally to ensure loading state is reset properly
    try {
        // (Processing continues in next step...)
        const response = await this.askSpiritualQuestion(question, model, readingStyle);
        this.displayResponse(response);
        this.showNotification('Guidance received! 🙏', 'success');
    } catch (error) {
        console.error('Error asking question:', error);
        this.showNotification('Failed to receive guidance. Please try again.', 'error');
    } finally {
        // 7. Always reset loading state when completed
        this.isLoading = false;
        this.setLoadingState(false);
    }
}
```

## 2. Frontend Parameter Processing

### Detailed Explanation
Before sending the request to the backend, the frontend must translate user-friendly UI concepts into technical parameters that the backend can use directly. Most importantly, it needs to convert the reading style selection (broad/balanced/deep) into specific retrieval parameters:

1. The `askSpiritualQuestion` method is called from the form submission handler, receiving:
   - question: The trimmed question text
   - model: The selected LLM model ID (e.g., "gpt-4.1")
   - readingStyle: The selected reading style ("broad", "balanced", or "deep")

2. The first operation is to convert the user-friendly reading style into technical retrieval parameters by calling `getRetrievalParamsFromStyle()`:
   - This checks if the admin settings panel is active and, if so, uses those custom values instead
   - Otherwise, it maps each reading style to specific technical parameters:
     - "broad": Uses MMR with high diversity (lambda_mult=0.2) and more results (k=7)
     - "balanced": Uses MMR with moderate diversity (lambda_mult=0.6) and standard results (k=6)
     - "deep": Uses pure similarity search (no MMR) for maximum relevance focus and fewer results (k=5)

3. The returned object contains three key technical parameters:
   - use_mmr: Boolean flag for whether to use Maximum Marginal Relevance (true) or pure similarity search (false)
   - diversity: A float value between 0-1 where lower values prioritize diversity and higher values prioritize relevance
   - k: Integer number of chunks to retrieve from the vector database

4. These parameters, along with the original question, model selection, and reading style, will be included in the API request.

### Implementing Code
```javascript
// Convert user-friendly reading style into technical retrieval parameters
getRetrievalParamsFromStyle(selectedStyle) {
    // Check if we should use custom admin settings (when in admin mode)
    if (this.adminControls && document.querySelector('.admin-card.active')) {
        return this.adminControls.getCurrentSettings();
    }
    
    // Otherwise use preset values based on reading style
    switch(selectedStyle) {
        case 'broad':
            return {
                use_mmr: true,
                diversity: 0.2,  // LOW value = HIGH diversity
                k: 7             // More results for broader coverage
            };
        case 'deep':
            return {
                use_mmr: false,  // Pure similarity search for focused results
                diversity: 0.0,  // Not used with similarity search
                k: 5             // Fewer results for deeper focus
            };
        case 'balanced':
        default:
            return {
                use_mmr: true,
                diversity: 0.6,  // Middle value - moderate diversity
                k: 6             // Standard number of results
            };
    }
}

// Function called from the form submission handler
async askSpiritualQuestion(question, model = 'gpt-4.1', readingStyle = 'balanced') {
    // 1. Convert reading style to concrete technical parameters
    const retrieval = this.getRetrievalParamsFromStyle(readingStyle);
    
    // (Request construction continues in next step...)
}
```

## 3. API Request Construction

### Detailed Explanation
With all parameters processed, the frontend now constructs a proper HTTP request to the backend API:

1. The system builds a formal HTTP request using the Fetch API:
   - The endpoint URL is constructed by combining the base API URL with the '/ask' path
   - The HTTP method is set to 'POST' to submit data to the server
   - Headers specify the content type as JSON
   - The body is constructed as a JSON object with all required parameters

2. The request body includes several key pieces of information:
   - question: The original user question text
   - model: The LLM model identifier to use for answering
   - reading_style: The user-selected reading style (for logging/tracking)
   - use_mmr: Boolean parameter controlling retrieval strategy
   - diversity: Float parameter controlling MMR diversity vs. relevance balance
   - k: Integer parameter controlling how many chunks to retrieve
   - max_context_docs: Integer limiting the maximum number of documents to include in context

3. The JSON body is stringified to convert the JavaScript object into a JSON string for transmission
4. The fetch() call initiates an asynchronous HTTP request to the backend server
5. The frontend code awaits the completion of this request before proceeding

### Implementing Code
```javascript
async askSpiritualQuestion(question, model = 'gpt-4.1', readingStyle = 'balanced') {
    // 1. Convert reading style to technical parameters (from previous step)
    const retrieval = this.getRetrievalParamsFromStyle(readingStyle);
    
    // 2. Construct and send the API request
    const response = await fetch(`${this.apiBase}/ask`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            question: question,
            model: model,
            max_context_docs: 5,
            reading_style: readingStyle,
            use_mmr: retrieval.use_mmr,
            diversity: retrieval.diversity,
            k: retrieval.k
        })
    });
    
    // (Response handling continues in next step...)
}
```

## 4. API Request Transmission

### Detailed Explanation
After constructing the request, the frontend transmits it to the backend API and handles the response:

1. The browser sends the HTTP POST request to the backend API endpoint.
2. The frontend code awaits the server's response.
3. Once a response is received, the frontend checks if it was successful:
   - If not successful (HTTP error code), it parses the JSON error response
   - The error object is extracted from the response body
   - An Error is thrown with the error detail message or a generic message

4. If the response is successful:
   - The JSON response body is parsed into a JavaScript object
   - This object is returned to the calling function (handleQuestionSubmit)
   - The calling function will then display the response in the UI

5. Throughout this process, proper error handling ensures that:
   - Network failures are caught
   - API errors are properly extracted and presented to the user
   - The loading state is always reset to false in the finally block

### Implementing Code
```javascript
async askSpiritualQuestion(question, model = 'gpt-4.1', readingStyle = 'balanced') {
    // Previous steps: parameter processing and request construction
    const retrieval = this.getRetrievalParamsFromStyle(readingStyle);
    
    const response = await fetch(`${this.apiBase}/ask`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            question: question,
            model: model,
            max_context_docs: 5,
            reading_style: readingStyle,
            use_mmr: retrieval.use_mmr,
            diversity: retrieval.diversity,
            k: retrieval.k
        })
    });
    
    // Error handling for unsuccessful responses
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to get spiritual guidance');
    }
    
    // Parse and return the successful response
    return response.json();
}
```

---

**[Continue to Part 2 for API Request Processing, Document Retrieval, and LLM Answer Generation]**
