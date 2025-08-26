/**
 * Spiritual Wisdom Chat App - Simplified Interface
 * Clean chat interface with WhatsApp-style design
 */

class SpiritualChatApp {
    constructor() {
        this.apiBase = window.APP_CONFIG.API_BASE_URL;
        this.isLoading = false;
        this.conversationHistory = [];
        this.conversationId = this.generateConversationId();
        this.currentReadingStyle = 'balanced';  // Options: 'deep', 'balanced', 'practical'
        this.settings = {
            model: 'gpt-4.1',
            use_mmr: true,
            k: 10,
            diversity: 0.3,
            reasoning_effort: 'medium',  // For o3-mini reasoning model
            rag_technique: 'stuff'       // For user-selectable RAG techniques
        };
        
        this.init();
    }
    
    async init() {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.setupApp());
        } else {
            this.setupApp();
        }
    }
    
    async setupApp() {
        console.log('🕉️ Initializing Spiritual Chat App...');
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Load initial data
        await this.loadInitialData();
        
        // Hide loading screen
        this.hideLoadingScreen();
        
        console.log('✅ Chat app initialized successfully');
    }
    
    setupEventListeners() {
        // Chat form submission
        const form = document.getElementById('wisdom-form');
        if (form) {
            form.addEventListener('submit', (e) => this.handleChatSubmission(e));
        }
        
        // Auto-resize textarea
        const textarea = document.getElementById('wisdom-question');
        console.log('🔍 Textarea element found:', textarea);
        if (textarea) {
            console.log('✅ Setting up textarea event listeners');
            textarea.addEventListener('input', () => this.autoResizeTextarea(textarea));
            
            // Submit on Enter key (without Shift)
            textarea.addEventListener('keydown', (e) => {
                console.log('🔑 Key pressed:', e.key, 'Shift held:', e.shiftKey);
                if (e.key === 'Enter' && !e.shiftKey) {
                    console.log('🚀 Enter key detected - submitting form');
                    e.preventDefault();
                    this.handleChatSubmission(e);
                }
            });
        } else {
            console.error('❌ Textarea element not found!');
        }
        
        // Reading style selector (now in admin settings)
        const readingStyleSelect = document.getElementById('reading-style-select');
        if (readingStyleSelect) {
            readingStyleSelect.addEventListener('change', (e) => {
                this.currentReadingStyle = e.target.value;
                console.log('📚 Reading style changed to:', this.currentReadingStyle);
            });
        }
        
        // Settings panel - Modern Zen structure
        const settingsToggle = document.getElementById('settings-toggle');
        const settingsPanel = document.getElementById('settings-panel');
        const closeSettings = document.getElementById('close-settings');
        
        if (settingsToggle && settingsPanel) {
            settingsToggle.addEventListener('click', (e) => {
                e.preventDefault();
                console.log('⚙️ Settings toggle clicked');
                settingsPanel.style.display = 'block';
                settingsPanel.classList.add('active');
            });
        } else {
            console.error('❌ Settings toggle or panel not found!');
            console.log('settingsToggle:', settingsToggle);
            console.log('settingsPanel:', settingsPanel);
        }
        
        if (closeSettings && settingsPanel) {
            closeSettings.addEventListener('click', (e) => {
                e.preventDefault();
                console.log('✕ Settings close clicked');
                settingsPanel.classList.remove('active');
                setTimeout(() => {
                    settingsPanel.style.display = 'none';
                }, 300);
            });
        }
        
        // Settings controls
        this.setupSettingsControls();
    }
    
    setupSettingsControls() {
        // Model selection
        const modelSelect = document.getElementById('model-select');
        const reasoningEffortGroup = document.getElementById('reasoning-effort-group');
        const reasoningEffortSelect = document.getElementById('reasoning-effort-select');
        
        if (modelSelect) {
            modelSelect.addEventListener('change', (e) => {
                this.settings.model = e.target.value;
                
                // Show/hide reasoning effort control based on model selection
                if (e.target.value === 'o3-mini') {
                    console.log('🧠 o3-mini selected - showing reasoning effort control');
                    reasoningEffortGroup.style.display = 'block';
                } else {
                    console.log('🤖 Standard model selected - hiding reasoning effort control');
                    reasoningEffortGroup.style.display = 'none';
                }
            });
        }
        
        // Reasoning effort selection (for o3-mini)
        if (reasoningEffortSelect) {
            reasoningEffortSelect.addEventListener('change', (e) => {
                this.settings.reasoning_effort = e.target.value;
                console.log(`🧠 Reasoning effort set to: ${e.target.value}`);
            });
        }
        
        // RAG technique selection
        const ragTechniqueSelect = document.getElementById('rag-technique-select');
        if (ragTechniqueSelect) {
            ragTechniqueSelect.addEventListener('change', (e) => {
                this.settings.rag_technique = e.target.value;
                console.log(`📊 RAG technique set to: ${e.target.value}`);
            });
        }
        
        // MMR checkbox
        const mmrCheckbox = document.getElementById('use-mmr');
        if (mmrCheckbox) {
            mmrCheckbox.addEventListener('change', (e) => {
                this.settings.use_mmr = e.target.checked;
            });
        }
        
        // K parameter slider
        const kSlider = document.getElementById('k-parameter');
        const kValue = document.getElementById('k-value');
        if (kSlider && kValue) {
            kSlider.addEventListener('input', (e) => {
                this.settings.k = parseInt(e.target.value);
                kValue.textContent = e.target.value;
            });
        }
        
        // Diversity slider
        const diversitySlider = document.getElementById('diversity-parameter');
        const diversityValue = document.getElementById('diversity-value');
        if (diversitySlider && diversityValue) {
            diversitySlider.addEventListener('input', (e) => {
                this.settings.diversity = parseFloat(e.target.value);
                diversityValue.textContent = e.target.value;
            });
        }
        
        // Info icon tooltips
        this.setupInfoTooltips();
    }
    
    setupInfoTooltips() {
        console.log('🔧 Setting up info tooltips...');
        
        // Use event delegation since settings panel might not be in DOM yet
        document.addEventListener('click', (e) => {
            // Handle processing info icon
            if (e.target && e.target.id === 'processing-info') {
                e.preventDefault();
                e.stopPropagation();
                const tooltip = document.getElementById('processing-tooltip');
                if (tooltip) {
                    const isVisible = tooltip.style.display === 'block';
                    tooltip.style.display = isVisible ? 'none' : 'block';
                    console.log('📋 Processing tooltip toggled:', !isVisible ? 'shown' : 'hidden');
                } else {
                    console.log('❌ Processing tooltip element not found');
                }
            }
            
            // Handle reading style info icon
            if (e.target && e.target.id === 'reading-style-info') {
                e.preventDefault();
                e.stopPropagation();
                const tooltip = document.getElementById('reading-style-tooltip');
                if (tooltip) {
                    const isVisible = tooltip.style.display === 'block';
                    tooltip.style.display = isVisible ? 'none' : 'block';
                    console.log('📚 Reading style tooltip toggled:', !isVisible ? 'shown' : 'hidden');
                } else {
                    console.log('❌ Reading style tooltip element not found');
                }
            }
            
            // Close all tooltips when clicking elsewhere (but not on info icons)
            if (e.target && !e.target.classList.contains('info-icon') && !e.target.closest('.info-tooltip')) {
                const tooltips = document.querySelectorAll('.info-tooltip');
                tooltips.forEach(tooltip => {
                    tooltip.style.display = 'none';
                });
            }
        });
        
        console.log('✅ Info tooltip event delegation setup complete');
    }
    
    async loadInitialData() {
        try {
            // Test API connection
            const response = await fetch(`${this.apiBase}/health`);
            if (!response.ok) {
                throw new Error('API connection failed');
            }
            console.log('✅ API connection successful');
        } catch (error) {
            console.error('❌ API connection failed:', error);
            this.showNotification('Unable to connect to the spiritual wisdom service. Please try again later.', 'error');
        }
    }
    
    hideLoadingScreen() {
        const loadingScreen = document.getElementById('loading');
        if (loadingScreen) {
            loadingScreen.style.opacity = '0';
            setTimeout(() => {
                loadingScreen.style.display = 'none';
            }, 500);
        } else {
            console.error('❌ Loading screen element not found!');
        }
    }
    
    async handleChatSubmission(e) {
        e.preventDefault();
        
        if (this.isLoading) return;
        
        const textarea = document.getElementById('wisdom-question');
        const question = textarea.value.trim();
        
        if (!question) {
            this.showNotification('Please enter a question before submitting.', 'warning');
            return;
        }
        
        // Add user message to chat
        this.addMessage(question, 'user');
        
        // Clear input
        textarea.value = '';
        this.autoResizeTextarea(textarea);
        
        // Set loading state
        this.setLoadingState(true);
        
        try {
            // Make API call
            const response = await this.callSpiritualAPI(question);
            
            // Add bot response to chat
            this.addMessage(response.answer, 'bot');
            
            // Update conversation history
            this.conversationHistory.push({
                question: question,
                answer: response.answer,
                timestamp: new Date().toISOString()
            });
            
        } catch (error) {
            console.error('Error getting spiritual guidance:', error);
            this.addMessage('I apologize, but I encountered an issue while seeking wisdom. Please try again.', 'bot');
            this.showNotification('Failed to get spiritual guidance. Please try again.', 'error');
        } finally {
            this.setLoadingState(false);
        }
    }
    
    async callSpiritualAPI(question) {
        const payload = {
            question: question,
            model: this.settings.model,
            conversation_id: this.conversationId,
            reading_style: this.currentReadingStyle,
            k: this.settings.k,
            use_mmr: this.settings.use_mmr,
            diversity: this.settings.diversity
        };
        
        // Include conversation history for context-aware responses
        if (this.conversationHistory && this.conversationHistory.length > 0) {
            payload.conversation_history = [];
            // Convert conversation history to the format expected by the backend
            this.conversationHistory.forEach(item => {
                // Add user question
                payload.conversation_history.push({
                    role: 'user',
                    content: item.question,
                    timestamp: item.timestamp
                });
                // Add assistant answer
                payload.conversation_history.push({
                    role: 'assistant',
                    content: item.answer,
                    timestamp: item.timestamp
                });
            });
            console.log(`📝 Including ${payload.conversation_history.length} conversation history items`);
        }
        
        // Add reasoning_effort parameter for o3-mini model
        if (this.settings.model === 'o3-mini') {
            payload.reasoning_effort = this.settings.reasoning_effort;
            console.log(`🧠 Adding reasoning_effort: ${this.settings.reasoning_effort} for o3-mini`);
        }
        
        // Add RAG technique parameter (always included)
        payload.rag_technique = this.settings.rag_technique;
        console.log(`📊 Using RAG technique: ${this.settings.rag_technique}`);
        
        
        console.log('📤 Sending request:', payload);
        
        const response = await fetch(`${this.apiBase}/ask`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) {
            throw new Error(`API request failed: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('📥 Received response:', data);
        
        return data;
    }
    
    addMessage(content, type) {
        const messagesContainer = document.getElementById('chat-messages');
        if (!messagesContainer) return;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
        
        if (type === 'user') {
            messageDiv.textContent = content;
        } else {
            // Bot message with formatting
            messageDiv.innerHTML = this.formatBotMessage(content);
        }
        
        messagesContainer.appendChild(messageDiv);
        
        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    formatBotMessage(content) {
        // Basic formatting for bot messages
        return content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>');
    }
    
    setLoadingState(loading) {
        this.isLoading = loading;
        const sendButton = document.getElementById('send-button');
        const textarea = document.getElementById('wisdom-question');
        
        if (sendButton) {
            sendButton.disabled = loading;
            sendButton.textContent = loading ? '⏳' : '➤';
        }
        
        if (textarea) {
            textarea.disabled = loading;
        }
        
        if (loading) {
            // Add typing indicator
            this.addTypingIndicator();
        } else {
            // Remove typing indicator
            this.removeTypingIndicator();
        }
    }
    
    addTypingIndicator() {
        const messagesContainer = document.getElementById('chat-messages');
        if (!messagesContainer) return;
        
        const typingDiv = document.createElement('div');
        typingDiv.className = 'bot-message typing-indicator';
        typingDiv.id = 'typing-indicator';
        typingDiv.innerHTML = '<span class="typing-dots">🤔 Seeking wisdom...</span>';
        
        messagesContainer.appendChild(typingDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    removeTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
    
    autoResizeTextarea(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    }
    
    generateConversationId() {
        return `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }
    
    showNotification(message, type = 'info') {
        const container = document.getElementById('notification-container');
        if (!container) return;
        
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        container.appendChild(notification);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            notification.remove();
        }, 5000);
    }
}

// Initialize the app when the page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Starting Spiritual Chat App...');
    new SpiritualChatApp();
});

// Global error handler
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
});

// Global unhandled promise rejection handler
window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
});
