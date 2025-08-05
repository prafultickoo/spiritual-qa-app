# Spiritual Q&A System: Detailed Backend Flow (Part 4)

This part covers the **final leg** of the journey: how the JSON response from the backend is consumed by the browser, rendered in the UI, and optionally copied or shared by the user.

Sections:
1. Fetch-Promise Resolution
2. Response Validation & Error Handling
3. DOM Update Workflow
4. Auxiliary Actions (copy, share, ask another)
5. Admin Panel Logging & Parameter Echo

---

## 1. Fetch-Promise Resolution

### Detailed Explanation
After the backend sends its JSON payload, the `fetch()` promise in `askSpiritualQuestion()` resolves.  Control returns to `handleQuestionSubmit()` with the parsed object.

Key data fields expected:
- `answer` – string with formatted response
- `model` – LLM id
- `chunks_used` – integer count
- `processing_time` – float ms
- (optional) `sources`, `suggested_practices`, `related_teachings`

### Code
```javascript
// handleQuestionSubmit (excerpt after await)
try {
    const response = await this.askSpiritualQuestion(question, model, readingStyle);
    this.displayResponse(response);         // ➜ Section 3
    this.showNotification('Guidance received! 🙏', 'success');
} catch (error) {
    console.error('Error asking question:', error);
    this.showNotification('Failed to receive guidance. Please try again.', 'error');
}
```

---

## 2. Response Validation & Error Handling

### Detailed Explanation
Inside `askSpiritualQuestion()` earlier, we already handled HTTP-level errors (`!response.ok`).  Now, in `displayResponse()` we trust that the payload is structurally correct, but we still guard against missing keys to avoid runtime exceptions.

### Code Snippet
```javascript
// Defensive defaults
const { query = '', response = '', sources = [], model_used, processing_time_ms } = responseData || {};
```

---

## 3. DOM Update Workflow

### Detailed Explanation
`displayResponse()` updates multiple regions of the page.

1. **Reveal response panel**
   - `#response-area` previously hidden via CSS.
2. **Insert main answer text**
   - `#guidance-response` innerText = `responseData.response`.
3. **Show metadata**
   - `.model-used` = `Model: ${responseData.model}`.
   - `.response-time` = `${responseData.processing_time}ms`.
4. **Populate citations** (if `sources` list given)
   - Clear `<ul id="sources-list">` then append `<li>` for each.
5. **Populate practices / teachings** similarly.
6. **Smooth scroll** to bring answer into view.

### Code (excerpts)
```javascript
const responseArea = document.getElementById('response-area');
responseArea.style.display = 'block';

// 2. main answer
document.getElementById('guidance-response').textContent = responseData.answer || responseData.response;

// 3. metadata
queryDisplay.textContent      = responseData.query;
modelUsed.textContent         = `Model: ${responseData.model}`;
responseTime.textContent      = `${responseData.processing_time}ms`;

// 4. sources
sourcesList.innerHTML = '';
(responseData.sources || []).forEach(src => {
   const li = document.createElement('li');
   li.textContent = src;
   sourcesList.appendChild(li);
});

responseArea.scrollIntoView({ behavior: 'smooth', block: 'start' });
```

---

## 4. Auxiliary Actions (copy, share, ask another)

### Detailed Explanation
Global click listener in `setupEventListeners()` delegates to helper methods:
- `copyResponse()` – copies plain text answer to clipboard, shows toast.
- `shareResponse()` – uses Web Share API if available.
- `askAnother()` – resets form & hides `#response-area`.

### Code
```javascript
copyResponse() {
   navigator.clipboard.writeText(document.getElementById('guidance-response').innerText)
       .then(() => this.showNotification('Copied to clipboard', 'success'))
       .catch(() => this.showNotification('Copy failed', 'error'));
}

shareResponse() {
   if (navigator.share) {
       navigator.share({ text: document.getElementById('guidance-response').innerText });
   } else {
       this.showNotification('Web Share API not available', 'warning');
   }
}
```

---

## 5. Admin Panel Logging & Parameter Echo

### Detailed Explanation
If the **Admin** tab is open, the frontend echoes back key parameters so power-users can observe impact in real-time.

1. `loadAPIStatus()` (invoked when switching to settings page) fetches `/get_admin_config` endpoint.
2. Values (`default_model`, `temperature`, etc.) populate form controls.
3. A small log area (`#api-log`) polls `/logs?limit=100` every 10 s and streams latest entries.

### Code
```javascript
async loadAPIStatus() {
   const res = await fetch(`${this.apiBase}/get_admin_config`);
   const cfg = await res.json();
   document.getElementById('default-model-setting').value = cfg.default_model;
   // … additional fields
   this.loadLogs();
}

async loadLogs() {
   const res = await fetch(`${this.apiBase}/logs?limit=100`);
   document.getElementById('api-log').textContent = (await res.json()).entries.join('\n');
}
```

---

### Flow Complete
At this point the user sees the answer, can copy/share it, and the application is ready for another question without page reload.

**End of Part 4.**  Further parts (if needed) could document offline batch jobs, PDF rotation/OCR pipelines, or deployment scripts.
