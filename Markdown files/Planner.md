## Spiritual Guide app planner 
[] Read and load the documents 
[] Vectorize the documents into embeddings 
[] Store the chunks in vector database 
[] Vector database should be persistent 
[] Convert user question into embeddings
[] Create document retriever to retrieve relevant chunks basis user query
[] Write a detailed prompt, point by point which instructs LLMs to answer user query 
[] Create a detailed answer from the chunks retrieved that answers users questions
[] Create endpoints for the following LLMs:
    1. GPT 4o 
    2. GPT 4.1 
    3. o3 mini 
    4. Grok 3 mini beta 
    5. Deepseek reasoner 
    6. Gemini pro 2.5 flash 
    7. Claude 3.7 Sonnet thinking 

    Default LLM to use is GPT 4.1

[] Create a sleek, modern looking front end which has a spiritual feeling to it
[] The user should be able to ask question via a front end. 
[] The answer should be displayed back on the front end 
[] Create API for the backend Q&A which front can invoke, and which can be invoked from any other application (eg Postman)
[] Write test scripts in a separate folder to test each functionality 

## Enhancements 
[] Adapt user personality style and answer accoringly 
[] Store answers in a database 
[] Have a separate "admin" tab on the front end which displays model parameters and logs
[] Admin tab should have the capability where user can change the default model from GPT4.1 to any model of his / her choice 
