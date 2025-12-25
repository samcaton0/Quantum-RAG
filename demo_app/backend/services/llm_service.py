"""
LLM Service for generating responses based on retrieved chunks.
"""

import os
from typing import List, Dict, Any
import google.generativeai as genai


class LLMService:
    """Service for generating LLM responses from retrieved chunks."""

    def __init__(self):
        """Initialize the LLM service."""
        # Configure Gemini API
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-3-pro-preview')
            self.available = True
        else:
            self.model = None
            self.available = False
            print("WARNING: GEMINI_API_KEY not set. LLM responses will not be generated.")

    def generate_response(self, query: str, retrieved_chunks: List[Dict[str, Any]], method: str) -> str:
        """
        Generate a response based on the query and retrieved chunks.

        Args:
            query: The user's query
            retrieved_chunks: List of retrieved chunk dictionaries with 'text', 'source', etc.
            method: The retrieval method used (topk, mmr, qubo)

        Returns:
            Generated response string
        """
        if not self.available:
            return "LLM service unavailable (API key not configured)"

        # Build the context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            source = chunk.get('source', 'unknown')
            text = chunk.get('text', '')
            context_parts.append(f"[Document {i} - {source}]\n{text}")

        context = "\n\n".join(context_parts)

        # Create the system prompt
        system_prompt = """You are an assistant that answers questions using ONLY the provided documents.

CRITICAL RULES:
1. Use ONLY information explicitly stated in the documents - no external knowledge
2. If documents lack necessary information, state this clearly
3. Be concise - write roughly one short paragraph per distinct topic in the documents
4. Synthesize information from multiple documents covering the same topic
5. Keep total response to 2-4 sentences per document provided

Your response must directly address the query using only the retrieved documents."""

        # Create the prompt
        prompt = f"""{system_prompt}

USER QUERY:
{query}

RETRIEVED DOCUMENTS:
{context}

Based ONLY on the information in the documents above, provide a response to the user's query. If the documents don't contain the necessary information, explicitly state that."""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"


# Global service instance
_llm_service_instance = None


def get_llm_service() -> LLMService:
    """Get or create the global LLM service instance."""
    global _llm_service_instance
    if _llm_service_instance is None:
        _llm_service_instance = LLMService()
    return _llm_service_instance
