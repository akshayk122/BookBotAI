"""
URL Agent for directly analyzing web content using Gemini AI.
"""
from typing import Dict, List, Optional, Any

from crewai import Agent

from src.analyzer.url_analyzer import URLAnalyzer
from src.config.llm_wrapper import get_gemini_llm


class URLAgent:
    """Agent for analyzing URLs directly with Gemini AI."""
    
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        """
        Initialize the URLAgent.
        
        Args:
            model_name: Name of the Gemini model to use
        """
        self.model_name = model_name
        self.analyzer = URLAnalyzer(model_name=model_name)
    
    def get_agent(self) -> Agent:
        """
        Get the CrewAI agent.
        
        Returns:
            CrewAI agent
        """
        return Agent(
            role="URL Analysis Agent",
            goal="Analyze web content directly using Gemini AI",
            backstory="I am an expert at extracting and analyzing information from web content. I specialize in using Gemini 2.0 to understand and summarize online text without storing unnecessary data.",
            verbose=True,
            allow_delegation=False,
            llm=self._get_llm()
        )
    
    def _get_llm(self):
        """
        Get the language model for the agent.
        
        Returns:
            Language model
        """
        # Use the centralized LLM wrapper
        return get_gemini_llm(model_name=self.model_name)
    
    def analyze_url(self, url: str) -> Dict[str, Any]:
        """
        Analyze a URL directly using Gemini AI.
        
        Args:
            url: URL to analyze
            
        Returns:
            Dictionary with analysis results
        """
        return self.analyzer.analyze_url(url)
    
    def get_book_summary(self, url: str) -> str:
        """
        Get a summary of a book from its URL.
        
        Args:
            url: URL of the book
            
        Returns:
            Summary of the book
        """
        analysis = self.analyze_url(url)
        return analysis.get("summary", "Unable to generate summary")
    
    def classify_genre(self, url: str) -> str:
        """
        Classify the genre of a book from its URL.
        
        Args:
            url: URL of the book
            
        Returns:
            Genre classification
        """
        analysis = self.analyze_url(url)
        return analysis.get("genre", "Unable to classify genre")
    
    def chat_with_url(self, url: str, query: str) -> str:
        """
        Chat about the content of a URL.
        
        Args:
            url: URL to chat about
            query: User query
            
        Returns:
            Response from the chat
        """
        # First get the analysis to have content available
        analysis = self.analyze_url(url)
        
        if "error" in analysis:
            return f"Error: {analysis['error']}"
        
        # Extract metadata and content
        title = analysis.get("title", "Unknown Title")
        author = analysis.get("author", "Unknown")
        content = analysis.get("content", "")
        
        if not content:
            return f"No content available for URL: {url}"
        
        # Create prompt for chat
        prompt = f"""
        You are an expert on the book "{title}" by {author}. A user has asked the following question about the book:
        
        "{query}"
        
        Please provide a detailed and accurate response based on the book's content. If the question cannot be answered based on the book content, explain why and provide general information that might be helpful.
        
        Here are relevant sections from the book to reference:
        {content}
        
        Provide a thoughtful, well-structured response that directly addresses the user's question with specific references to the book content where possible.
        """
        
        try:
            # Get response from LLM
            llm = self._get_llm()
            response = llm.invoke(prompt)
            return response
        except Exception as e:
            # Fallback with even smaller content if there's an error
            if len(content) > 8000:  # Increased for Gemini 2.0's larger context window
                smaller_content = content[:8000] + "..."
                fallback_prompt = f"Based on this excerpt from \"{title}\" by {author}, please answer: \"{query}\". Excerpt: {smaller_content}"
                try:
                    return llm.invoke(fallback_prompt)
                except:
                    return f"I'm sorry, but I couldn't process your query about '{title}' due to content length limitations. Please try asking a more specific question."
            else:
                return f"Error processing your query about '{title}': {str(e)}"
