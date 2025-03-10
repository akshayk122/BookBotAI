"""
Streamlined AI system for Project Gutenberg books using direct URL analysis with Gemini AI.
"""
import os
from typing import Dict, List, Optional, Any, Tuple

from crewai import Crew, Agent, Task

# Import the URL agent for all functionality
from src.agents.url_agent import URLAgent


class AgentSystem:
    """Streamlined AI system for Project Gutenberg books using direct URL analysis."""
    
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        """
        Initialize the AgentSystem.
        
        Args:
            model_name: Name of the Gemini model to use
        """
        self.model_name = model_name
        
        # Initialize URL agent for all functionality
        self.url_agent = URLAgent(model_name=model_name)
        
        # Store the current URL and analysis results
        self.current_url = None
        self.current_analysis = None
        
        # Initialize crew
        self.crew = self._create_crew()
    
    def _create_crew(self) -> Crew:
        """
        Create a CrewAI crew with the URL agent.
        
        Returns:
            CrewAI crew
        """
        return Crew(
            agents=[
                self.url_agent.get_agent()
            ],
            tasks=[],  # Tasks will be added dynamically
            verbose=True
        )
    
    def analyze_url(self, url: str) -> Dict[str, Any]:
        """
        Analyze a URL directly using Gemini AI.
        
        Args:
            url: URL to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # Store the current URL and analysis
        self.current_url = url
        self.current_analysis = self.url_agent.analyze_url(url)
        return self.current_analysis
    
    def get_book_summary(self, url: str) -> str:
        """
        Get a summary of a book from its URL.
        
        Args:
            url: URL of the book
            
        Returns:
            Summary of the book
        """
        # If this is the current URL, use cached analysis
        if url == self.current_url and self.current_analysis:
            return self.current_analysis.get("summary", "Unable to generate summary")
        
        # Otherwise, get a new analysis
        return self.url_agent.get_book_summary(url)
    
    def classify_genre(self, url: str) -> str:
        """
        Classify the genre of a book from its URL.
        
        Args:
            url: URL of the book
            
        Returns:
            Genre classification
        """
        # If this is the current URL, use cached analysis
        if url == self.current_url and self.current_analysis:
            return self.current_analysis.get("genre", "Unable to classify genre")
        
        # Otherwise, get a new analysis
        return self.url_agent.classify_genre(url)
    
    def chat_with_url(self, url: str, query: str) -> str:
        """
        Chat about the content of a URL.
        
        Args:
            url: URL to chat about
            query: User query
            
        Returns:
            Response from the chat
        """
        return self.url_agent.chat_with_url(url, query)
    
    def _extract_url_reference(self, query: str, current_url: Optional[str] = None) -> str:
        """
        Extract URL reference from a query, handling references to "the book" or "this book".
        
        Args:
            query: User query
            current_url: URL of the current book (if available)
            
        Returns:
            URL to use for the query
        """
        query_lower = query.lower()
        
        # Check for references to "the book" or "this book"
        book_references = ["the book", "this book", "current book"]
        uses_generic_reference = any(ref in query_lower for ref in book_references)
        
        # If query uses a generic book reference and we have a current URL, use that
        if uses_generic_reference and current_url:
            return current_url
            
        # If no current URL is provided but we have one stored, use that
        if not current_url and self.current_url:
            return self.current_url
            
        # Return the provided URL or None
        return current_url
    
    def process_user_query(self, query: str, current_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user query and route it to the appropriate agent.
        
        Args:
            query: User query
            current_url: URL of the current book (if available)
            
        Returns:
            Response from the agent
        """
        # Determine query type and route to appropriate agent
        query_lower = query.lower()
        
        # Extract URL reference, handling "the book" or "this book" references
        url_to_use = self._extract_url_reference(query, current_url)
        
        # If we don't have a URL, return an error
        if not url_to_use:
            return {"type": "error", "content": "Please provide a Project Gutenberg URL first."}
        
        # Check for summarization request
        if "summarize" in query_lower or "summary" in query_lower:
            # Handle cases where "summarize" is followed by empty text or just articles
            if "summarize the" in query_lower or "summarize this" in query_lower or query_lower == "summarize":
                summary = self.get_book_summary(url_to_use)
                # Get title from current analysis if available
                title = self.current_analysis.get("title", "Unknown Title") if self.current_analysis else "Unknown Title"
                return {"type": "summary", "title": title, "content": summary}
            else:
                summary = self.get_book_summary(url_to_use)
                # Get title from current analysis if available
                title = self.current_analysis.get("title", "Unknown Title") if self.current_analysis else "Unknown Title"
                return {"type": "summary", "title": title, "content": summary}
        
        # Check for genre classification request
        elif "genre" in query_lower or "classify" in query_lower:
            genre = self.classify_genre(url_to_use)
            # Get title from current analysis if available
            title = self.current_analysis.get("title", "Unknown Title") if self.current_analysis else "Unknown Title"
            return {"type": "genre", "title": title, "content": genre}
        
        # Default to chat
        else:
            response = self.chat_with_url(url_to_use, query)
            # Get title from current analysis if available
            title = self.current_analysis.get("title", "Unknown Title") if self.current_analysis else "Unknown Title"
            return {"type": "chat", "title": title, "query": query, "content": response}
