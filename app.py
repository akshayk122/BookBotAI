"""
Streamlit app for Project Gutenberg AI Analyzer using direct URL analysis with Gemini 2.0 Flash.
"""
import os
import time
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Gemini AI
from src.config.litellm_config import configure_gemini
configure_gemini()

import streamlit as st

from src.agents.agent_system import AgentSystem


# Set page config
st.set_page_config(
    page_title="Project Gutenberg AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for sidebar
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "agent_system" not in st.session_state:
    st.session_state.agent_system = AgentSystem(model_name="gemini-2.0-flash")

if "current_url" not in st.session_state:
    st.session_state.current_url = None
    
if "current_analysis" not in st.session_state:
    st.session_state.current_analysis = None


def analyze_url(url: str):
    """
    Analyze a URL using Gemini AI.
    
    Args:
        url: URL to analyze
    """
    if not url.startswith("https://www.gutenberg.org"):
        st.error("Please enter a valid Project Gutenberg book URL (starting with https://www.gutenberg.org)")
        return
        
    with st.spinner(f"Analyzing content from {url}..."):
        # Use the agent system to analyze the URL
        analysis = st.session_state.agent_system.analyze_url(url)
        
        if "error" in analysis:
            st.error(f"Failed to analyze the URL: {analysis['error']}")
            return
            
        # Update session state
        st.session_state.current_url = url
        st.session_state.current_analysis = analysis
        
        st.success(f"Successfully analyzed '{analysis.get('title', 'Unknown')}' by {analysis.get('author', 'Unknown')}!")


def display_analysis_info(analysis: Dict[str, Any]):
    """
    Display information about a URL analysis.
    
    Args:
        analysis: Analysis data dictionary
    """
    st.subheader(analysis.get("title", "Unknown Title"))
    st.write(f"**Author:** {analysis.get('author', 'Unknown')}")
    st.write(f"**Genre:** {analysis.get('genre', 'Unknown')}")
    st.write(f"**Language:** {analysis.get('language', 'Unknown')}")
    st.write(f"**Year:** {analysis.get('year', 'Unknown')}")
    st.write(f"**URL:** {analysis.get('url', 'Unknown')}")
    
    # Display summary
    if "summary" in analysis and analysis["summary"]:
        with st.expander("Summary"):
            st.write(analysis["summary"])
    
    # Display content preview
    if "content" in analysis and analysis["content"]:
        with st.expander("Content Preview"):
            st.write(analysis["content"][:1000] + "..." if len(analysis["content"]) > 1000 else analysis["content"])


def process_user_query(query: str, current_url: str = None):
    """
    Process a user query and display the result.
    
    Args:
        query: User query
        current_url: URL of the current book (if available)
    """
    with st.spinner("Processing your query..."):
        try:
            result = st.session_state.agent_system.process_user_query(query, current_url)
            
            # Check if result is None
            if result is None:
                st.error("Sorry, I couldn't process your query. Please try a different question.")
                return
            
            # Use dict.get() with default values for all dictionary accesses
            result_type = result.get("type", "unknown")
            result_content = result.get("content", "No content available")
            result_title = result.get("title", "Unknown")
            
            if result_type == "error":
                st.error(result_content)
                return
            
            if result_type == "summary":
                st.subheader(f"Summary of '{result_title}'") 
                st.write(result_content)
            
            elif result_type == "genre":
                st.subheader(f"Genre Classification for '{result_title}'") 
                st.write(result_content)
            
            elif result_type == "chat":
                st.subheader(f"Chat with '{result_title}'") 
                st.write(f"**Question:** {result.get('query', 'Unknown')}")
                st.write(f"**Answer:** {result_content}")
            
            else:
                st.write(result_content)
                
        except TypeError as e:
            st.error(f"Error processing query: The AI system returned an unexpected response format. Please try a different question.")
            st.info("Technical details: " + str(e))
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            st.info("Please try a different question or analyze another book.")


# Sidebar - URL input and analyze button
st.sidebar.title("Project Gutenberg AI")
st.sidebar.image("https://www.gutenberg.org/gutenberg/pg-logo-129x80.png")

# Option to analyze a book by URL
st.sidebar.header("Book Analysis")
book_url = st.sidebar.text_input("Enter Project Gutenberg Book URL", placeholder="https://www.gutenberg.org/ebooks/...")
if st.sidebar.button("Analyze This Book"):
    analyze_url(book_url)

# Main content - Show analysis and query interaction
st.title("📚 Project Gutenberg AI with Gemini")

# Display current analysis
if st.session_state.current_analysis:
    st.header("Analyzed Book")
    display_analysis_info(st.session_state.current_analysis)
    
    # Option to interact with AI agents
    st.header("AI Agent Interaction")
    user_query = st.text_area("Enter your query about this book", key="user_query", 
                            placeholder="Ask about the book, request a summary, or inquire about the genre...")
    
    # Add helpful examples
    with st.expander("Example queries"):
        st.markdown("- **Summarize this book**")
        st.markdown("- **What is the genre of the book?**")
        st.markdown("- **Who is the main character in the book?**")
        st.markdown("- **What are the major themes in this book?**")
        st.markdown("- **Tell me about the author's writing style**")
    
    if st.button("Process Query"):
        if not user_query:
            st.error("Please enter a query!")
        else:
            # Pass the current URL to the process_user_query function
            process_user_query(user_query, st.session_state.current_url)
else:
    st.info("Please enter a Project Gutenberg book URL in the sidebar and click 'Analyze This Book' to get started.")

# Footer
st.markdown("---")
st.write("Powered by Google Gemini 2.0 Flash | Data from Project Gutenberg")

# Add a note about referring to "the book" or "this book"
st.info("💡 Tip: You can refer to 'the book' or 'this book' in your queries, and the system will understand you're talking about the currently displayed book.")


if __name__ == "__main__":
    # This is already handled by Streamlit
    pass
