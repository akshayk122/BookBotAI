"""
URL Analyzer using Gemini AI to directly analyze content from URLs.
"""
import re
from typing import Dict, Any, Optional

import requests
from bs4 import BeautifulSoup

from src.config.llm_wrapper import get_gemini_llm


class URLAnalyzer:
    """Analyzer for web content using Gemini AI."""
    
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        """
        Initialize the URLAnalyzer.
        
        Args:
            model_name: Name of the Gemini model to use
        """
        self.model_name = model_name
        self.llm = get_gemini_llm(model_name=model_name)
        
        # Set up requests session with headers for consistent requests
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9"
        })
    
    def extract_text_from_url(self, url: str) -> str:
        """
        Extract clean text content from a URL.
        
        Args:
            url: URL to extract content from
            
        Returns:
            Clean text content
        """
        try:
            response = self.session.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Get text
            text = soup.get_text()
            
            # Break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
            
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            
            # Remove blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            return f"Error extracting content from URL: {str(e)}"
    
    def analyze_url(self, url: str) -> Dict[str, Any]:
        """
        Analyze a URL using Gemini AI.
        
        Args:
            url: URL to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # Check if it's a Project Gutenberg URL
        if "gutenberg.org" not in url:
            return {
                "error": "This URL does not appear to be from Project Gutenberg. Please provide a valid Project Gutenberg URL."
            }
        
        try:
            # Extract metadata from URL
            metadata = self._extract_metadata_from_url(url)
            
            # Get content URL if available
            content_url = self._get_content_url(url)
            
            # Extract text content
            if content_url:
                content = self.extract_text_from_url(content_url)
            else:
                content = self.extract_text_from_url(url)
            
            # Analyze content with Gemini
            analysis = self._analyze_content_with_gemini(content, metadata)
            
            # Combine metadata and analysis
            result = {**metadata, **analysis, "url": url}
            
            return result
        except Exception as e:
            return {"error": f"Error analyzing URL: {str(e)}"}
    
    def _extract_metadata_from_url(self, url: str) -> Dict[str, str]:
        """
        Extract metadata from a Project Gutenberg URL.
        
        Args:
            url: Project Gutenberg URL
            
        Returns:
            Dictionary with metadata
        """
        try:
            response = self.session.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Initialize metadata
            metadata = {
                "title": "",
                "author": "",
                "language": "English",  # Default
                "year": ""
            }
            
            # Extract title
            title_elem = soup.find("h1", {"itemprop": "name"})
            if title_elem:
                metadata["title"] = title_elem.text.strip()
            
            # Extract author
            author_elem = soup.find("a", {"itemprop": "creator"})
            if author_elem:
                metadata["author"] = author_elem.text.strip()
            
            # Extract language
            lang_elem = soup.find("tr", string=re.compile("Language"))
            if lang_elem and lang_elem.find_next("td"):
                metadata["language"] = lang_elem.find_next("td").text.strip()
            
            # Extract year
            year_elem = soup.find("th", string=re.compile("Release Date"))
            if year_elem and year_elem.find_next("td"):
                year_text = year_elem.find_next("td").text.strip()
                year_match = re.search(r"\d{4}", year_text)
                if year_match:
                    metadata["year"] = year_match.group(0)
            
            return metadata
        except Exception as e:
            print(f"Error extracting metadata: {e}")
            return {
                "title": "Unknown Title",
                "author": "Unknown Author",
                "language": "Unknown",
                "year": "Unknown"
            }
    
    def _get_content_url(self, url: str) -> Optional[str]:
        """
        Get the URL for the plain text content of a book.
        
        Args:
            url: Project Gutenberg book URL
            
        Returns:
            URL to plain text content or None if not found
        """
        try:
            response = self.session.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Find text content URL (prefer plain text)
            for a_tag in soup.find_all("a"):
                if "Plain Text UTF-8" in a_tag.text:
                    return a_tag["href"]
            
            # If no plain text found, try other formats
            for a_tag in soup.find_all("a"):
                if ".txt" in a_tag.get("href", ""):
                    return a_tag["href"]
            
            return None
        except Exception as e:
            print(f"Error getting content URL: {e}")
            return None
    
    def _analyze_content_with_gemini(self, content: str, metadata: Dict[str, str]) -> Dict[str, Any]:
        """
        Analyze content with Gemini AI.
        
        Args:
            content: Text content to analyze
            metadata: Book metadata
            
        Returns:
            Dictionary with analysis results
        """
        title = metadata.get("title", "Unknown Title")
        author = metadata.get("author", "Unknown Author")
        
        # Truncate content if too long for Gemini
        # Gemini 2.0 has a larger context window, so we can use more content
        max_content_length = 16000
        if len(content) > max_content_length:
            # Take samples from beginning, middle, and end
            beginning = content[:int(max_content_length/3)]
            middle_start = int(len(content)/2 - max_content_length/6)
            middle = content[middle_start:middle_start + int(max_content_length/3)]
            end = content[-int(max_content_length/3):]
            content_sample = f"{beginning}\n\n[...middle section omitted...]\n\n{middle}\n\n[...later section omitted...]\n\n{end}"
        else:
            content_sample = content
        
        try:
            # Generate summary
            summary_prompt = f"""
            Please provide a comprehensive summary of the book "{title}" by {author}.
            
            Include the following in your summary:
            1. Main plot points and narrative arc
            2. Key characters and their development
            3. Major themes and motifs
            4. Writing style and tone
            5. Historical or cultural context (if relevant)
            
            Make the summary detailed enough to give a good understanding of the book, but concise enough to be readable in a few minutes.
            
            Here is a sample of the book content to summarize:
            {content_sample}
            
            Provide a well-structured, engaging summary that captures the essence of the book.
            """
            
            summary = self.llm.invoke(summary_prompt)
            
            # Classify genre
            genre_prompt = f"""
            Based on the following excerpt from "{title}" by {author}, classify the genre of this book.
            Consider elements such as setting, themes, style, and plot elements.
            Provide a single primary genre and up to three subgenres if applicable.
            
            Excerpt:
            {content_sample[:8000]}
            
            Format your response as: Primary Genre: [genre], Subgenres: [subgenre1, subgenre2, subgenre3]
            """
            
            genre = self.llm.invoke(genre_prompt)
            
            return {
                "summary": summary,
                "genre": genre,
                "content": content_sample  # Store the sample for chat functionality
            }
        except Exception as e:
            return {
                "summary": f"Error generating summary: {str(e)}",
                "genre": "Unknown",
                "content": content_sample[:4000]  # Store a shorter sample if there was an error
            }
