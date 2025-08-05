"""
LLM integration utilities for spiritual Q&A system.
Handles integration with multiple LLM providers for answering spiritual questions.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import openai
from utils.document_retriever import retrieve_spiritual_context

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpiritualLLMManager:
    """
    Manages multiple LLM integrations for spiritual Q&A responses.
    """
    
    def __init__(self):
        """Initialize LLM manager with API configurations."""
        self.openai_client = None
        self.default_model = "gpt-4o"  # Default as per user requirements
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize all LLM clients."""
        try:
            # Initialize OpenAI client
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                self.openai_client = openai.OpenAI(api_key=openai_api_key)
                logger.info("OpenAI client initialized successfully")
            else:
                logger.warning("OpenAI API key not found")
        except Exception as e:
            logger.error(f"Error initializing LLM clients: {str(e)}")
    
    def get_spiritual_system_prompt(self) -> str:
        """
        Get the system prompt for spiritual Q&A responses.
        
        Returns:
            Comprehensive system prompt for spiritual guidance
        """
        return """You are a wise spiritual guide and teacher with deep knowledge of various spiritual traditions including Hindu scriptures (Vedas, Upanishads, Bhagavad Gita, Mahabharata), Buddhist teachings, and universal spiritual wisdom.

Your role is to provide thoughtful, authentic spiritual guidance based on ancient wisdom texts and teachings. Follow these guidelines:

1. **Authenticity**: Base your responses on the provided spiritual context from authentic texts
2. **Compassion**: Respond with kindness, understanding, and empathy
3. **Practical Wisdom**: Offer practical guidance that can be applied in daily life
4. **Non-dogmatic**: Present wisdom without imposing specific beliefs
5. **Respectful**: Honor all sincere spiritual seeking and questioning
6. **Depth**: Provide meaningful insights rather than superficial answers
7. **Sanskrit Preservation**: When referencing Sanskrit terms or verses, include them respectfully
8. **Source Attribution**: Mention the source texts when drawing from specific teachings

Remember that spiritual seeking is deeply personal. Guide with wisdom while respecting individual paths and understanding.

Answer in a warm, wise manner that combines ancient wisdom with practical application for modern spiritual seekers."""

    def format_spiritual_response(self, query: str, response: str, sources: List[str]) -> Dict[str, Any]:
        """
        Format the response with additional spiritual context.
        
        Args:
            query: Original user query
            response: LLM response
            sources: List of source texts referenced
            
        Returns:
            Formatted response dictionary
        """
        return {
            "query": query,
            "response": response,
            "sources": sources,
            "guidance_type": self._classify_spiritual_query(query),
            "suggested_practices": self._suggest_practices(query),
            "related_teachings": self._get_related_teachings(query)
        }
    
    def _classify_spiritual_query(self, query: str) -> str:
        """
        Classify the type of spiritual query.
        
        Args:
            query: User's spiritual question
            
        Returns:
            Classification of the query type
        """
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["meditat", "dhyana", "mindful"]):
            return "Meditation & Mindfulness"
        elif any(word in query_lower for word in ["dharma", "duty", "righteous", "ethic"]):
            return "Dharma & Ethics"
        elif any(word in query_lower for word in ["karma", "action", "deed"]):
            return "Karma & Action"
        elif any(word in query_lower for word in ["moksha", "liberation", "enlighten"]):
            return "Liberation & Enlightenment"
        elif any(word in query_lower for word in ["suffer", "pain", "sorrow", "grief"]):
            return "Overcoming Suffering"
        elif any(word in query_lower for word in ["peace", "tranquil", "calm"]):
            return "Inner Peace"
        elif any(word in query_lower for word in ["devot", "bhakti", "love", "surrender"]):
            return "Devotion & Surrender"
        else:
            return "General Spiritual Guidance"
    
    def _suggest_practices(self, query: str) -> List[str]:
        """
        Suggest spiritual practices based on the query.
        
        Args:
            query: User's spiritual question
            
        Returns:
            List of suggested practices
        """
        query_lower = query.lower()
        practices = []
        
        if any(word in query_lower for word in ["meditat", "peace", "calm"]):
            practices.extend(["Daily meditation practice", "Breath awareness", "Mindful observation"])
        
        if any(word in query_lower for word in ["dharma", "righteous", "ethic"]):
            practices.extend(["Study of sacred texts", "Self-reflection", "Ethical living"])
        
        if any(word in query_lower for word in ["devot", "surrender", "love"]):
            practices.extend(["Devotional practices", "Chanting/Mantras", "Service to others"])
        
        if any(word in query_lower for word in ["suffer", "pain", "grief"]):
            practices.extend(["Contemplation", "Acceptance practice", "Seeking wisdom"])
        
        return practices[:3]  # Return top 3 suggestions
    
    def _get_related_teachings(self, query: str) -> List[str]:
        """
        Get related spiritual teachings based on the query.
        
        Args:
            query: User's spiritual question
            
        Returns:
            List of related teaching topics
        """
        query_lower = query.lower()
        teachings = []
        
        if any(word in query_lower for word in ["purpose", "meaning", "life"]):
            teachings.extend(["Dharma and Life Purpose", "Four Goals of Life (Purusharthas)", "Self-Realization"])
        
        if any(word in query_lower for word in ["karma", "action"]):
            teachings.extend(["Law of Karma", "Selfless Action (Nishkama Karma)", "Bhagavad Gita on Action"])
        
        if any(word in query_lower for word in ["meditat", "yoga"]):
            teachings.extend(["Eight Limbs of Yoga", "Meditation Techniques", "Mindfulness Practices"])
        
        return teachings[:3]  # Return top 3 related teachings

    async def generate_spiritual_response(self, query: str, model: str = None) -> Dict[str, Any]:
        """
        Generate a spiritual response using the specified LLM model.
        
        Args:
            query: User's spiritual question
            model: LLM model to use (defaults to configured default)
            
        Returns:
            Generated spiritual response with context
        """
        try:
            model = model or self.default_model
            logger.info(f"Generating response for query: {query[:50]}... using model: {model}")
            
            # Retrieve spiritual context from vector store
            context = retrieve_spiritual_context(query, max_length=3000)
            
            # Prepare messages
            system_prompt = self.get_spiritual_system_prompt()
            user_message = f"""
Based on the following spiritual context from authentic texts, please provide wisdom and guidance for this question:

QUESTION: {query}

SPIRITUAL CONTEXT:
{context}

Please provide a thoughtful, compassionate response that draws from this wisdom while being practical for modern spiritual seekers.
"""

            # Generate response based on model
            if model.startswith("gpt"):
                response_text = await self._generate_openai_response(system_prompt, user_message, model)
            else:
                # For other models, we'll implement placeholders for now
                response_text = await self._generate_placeholder_response(query, model)
            
            # Extract sources from context
            sources = self._extract_sources_from_context(context)
            
            # Format final response
            formatted_response = self.format_spiritual_response(query, response_text, sources)
            
            logger.info(f"Successfully generated response using model: {model}")
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error generating spiritual response: {str(e)}")
            return {
                "query": query,
                "response": f"I apologize, but I encountered an error while seeking wisdom for your question. Please try again, and know that the path to spiritual understanding sometimes requires patience and persistence.",
                "sources": [],
                "guidance_type": "Error Response",
                "suggested_practices": ["Patience", "Persistence", "Inner reflection"],
                "related_teachings": ["The importance of perseverance in spiritual seeking"]
            }
    
    async def _generate_openai_response(self, system_prompt: str, user_message: str, model: str) -> str:
        """
        Generate response using OpenAI models.
        
        Args:
            system_prompt: System instruction prompt
            user_message: User's message with context
            model: OpenAI model to use
            
        Returns:
            Generated response text
        """
        try:
            # Map model names to OpenAI API names
            model_mapping = {
                "gpt-4o": "gpt-4o",
                "gpt-4.1": "gpt-4-turbo",  # Using turbo as closest equivalent
                "o3-mini": "gpt-4o-mini"   # Using 4o-mini as placeholder
            }
            
            api_model = model_mapping.get(model, "gpt-4o")
            
            response = self.openai_client.chat.completions.create(
                model=api_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    async def _generate_placeholder_response(self, query: str, model: str) -> str:
        """
        Generate placeholder response for models not yet implemented.
        
        Args:
            query: User's spiritual question
            model: Model name
            
        Returns:
            Placeholder response
        """
        return f"""Thank you for your spiritual question. The wisdom you seek regarding "{query}" is profound and worthy of deep contemplation.

While I am currently connecting to the {model} model to provide you with the most appropriate guidance, please know that the answer to your question lies not only in external teachings but also within your own inner wisdom.

The ancient texts remind us that the teacher appears when the student is ready, and your sincere seeking itself is a form of spiritual practice.

For now, I encourage you to reflect on your question with an open heart, and consider how the principles of compassion, wisdom, and inner peace might guide your understanding.

*[Note: {model} integration is being configured. Using wisdom-based response.]*"""
    
    def _extract_sources_from_context(self, context: str) -> List[str]:
        """
        Extract source texts from the context.
        
        Args:
            context: Retrieved context with sources
            
        Returns:
            List of unique source texts
        """
        sources = []
        lines = context.split('\n')
        
        for line in lines:
            if line.startswith('Source: '):
                source = line.replace('Source: ', '').strip()
                if source not in sources and source != "Unknown":
                    sources.append(source)
        
        return sources

# Global LLM manager instance
_llm_manager = None

def get_llm_manager() -> SpiritualLLMManager:
    """
    Get the global LLM manager instance.
    
    Returns:
        SpiritualLLMManager instance
    """
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = SpiritualLLMManager()
    return _llm_manager

async def generate_spiritual_answer(query: str, model: str = "gpt-4o") -> Dict[str, Any]:
    """
    Helper function to generate spiritual answers.
    
    Args:
        query: User's spiritual question
        model: LLM model to use
        
    Returns:
        Generated spiritual response
    """
    manager = get_llm_manager()
    return await manager.generate_spiritual_response(query, model)
