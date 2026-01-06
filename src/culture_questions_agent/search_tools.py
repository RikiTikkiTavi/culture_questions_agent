"""Wikipedia and web search tools using DDGS (multi-engine search)."""
import logging
from typing import List, Optional
from llama_index.tools.wikipedia import WikipediaToolSpec
from ddgs.ddgs import DDGS

logger = logging.getLogger(__name__)


class SearchEngine:
    """Search engine using Wikipedia and DDGS (multi-engine web search) as fallback."""
    
    def __init__(self, max_chars: int = 2500, include_title: bool = True):
        """
        Initialize search tools.
        
        Args:
            max_chars: Maximum characters to keep from search results
            include_title: Whether to include titles in search snippets
        """
        logger.info("Initializing search tools...")
        self.max_chars = max_chars
        self.include_title = include_title
        logger.info(f"  Max search result length: {max_chars} chars")
        logger.info(f"  Include titles in snippets: {include_title}")
        
        # Wikipedia search (primary)
        self.wikipedia_tool = WikipediaToolSpec()
        
        # DDGS web search (fallback) - uses multiple search engines
        self.ddgs: DDGS | None = None
        try:
            self.ddgs = DDGS()
            logger.info("DDGS web search initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize DDGS: {e}")
    
    def search_wikipedia(self, query: str) -> str:
        """
        Search Wikipedia for a query.
        
        Args:
            query: Search query
            
        Returns:
            Wikipedia page content or empty string if not found
        """
        logger.info(f"Wikipedia search: '{query}'")
        
        try:
            # search_data returns a string (page content) or an error message
            wiki_text = self.wikipedia_tool.search_data(query)
            
            # Check for hard-coded failure strings from the tool spec
            if wiki_text and "No search results" not in wiki_text and "Unable to load" not in wiki_text:
                if len(wiki_text) > 100:  # Ensure we actually have substance
                    logger.info(f"✓ Wikipedia found content ({len(wiki_text)} chars)")
                    return wiki_text[:self.max_chars]  # Cap context window
            
            logger.debug(f"Wikipedia returned no useful content: {wiki_text[:50] if wiki_text else 'None'}...")
            
        except Exception as e:
            logger.debug(f"Wikipedia tool error: {e}")
        
        return ""


    def search_web(self, query: str, max_results: int = 3, return_list: bool = False) -> str | List[str]:
        """
        Search the web using DDGS (multi-engine search).
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            return_list: If True, return list of snippets; if False, return concatenated string
            
        Returns:
            List of search snippets or concatenated string (based on return_list)
        """
        if not self.ddgs:
            logger.warning("DDGS not initialized")
            return ""
        
        logger.info(f"DDGS web search: '{query}'")
        
        try:
            # Use text search from ddgs
            results = list(self.ddgs.text(query, max_results=max_results))
            
            if results:
                cleaned_snippets = []
                for item in results:
                    # ddgs returns dicts with 'title', 'href', 'body'
                    title = item.get('title', '')
                    body = item.get('body', '')
                    
                    if self.include_title:
                        snippet = f"{title}:\n{body}"
                    else:
                        snippet = body
                    
                    if len(body) > 20:
                        cleaned_snippets.append(snippet)
                        
                if cleaned_snippets:
                    if return_list:
                        logger.info(f"✓ DDGS found {len(cleaned_snippets)} snippets")
                        return cleaned_snippets
                    else:
                        combined_text = "\n---\n".join(cleaned_snippets)
                        logger.info(f"✓ DDGS found {len(cleaned_snippets)} snippets")
                        return combined_text[:self.max_chars]
                    
        except Exception as e:
            logger.info(f"DDGS search failed: {e}")
        
        return [] if return_list else ""
    
    def search(self, query: str, max_results: int = 3) -> str:
        """
        Search for information using Wikipedia, fallback to DDGS web search.
        
        Args:
            query: Search query
            max_results: Maximum number of results for DDGS
            
        Returns:
            Search results as text
        """
        logger.info(f"Searching for: '{query}'")
        
        # Try Wikipedia first
        wiki_result = self.search_wikipedia(query)
        if wiki_result:
            return wiki_result
        
        # Fallback to DDGS web search
        logger.info("Falling back to DDGS web search...")
        return self.search_web(query, max_results, return_list=False)  # type: ignore
    
    def search_option_with_context(
        self, 
        option_text: str, 
        context_str: str
    ) -> str:
        """
        Search for an option with contextual keywords.
        
        Args:
            option_text: The answer option text
            context_str: Context keywords from question
            
        Returns:
            Search results
        """
        query = f"{option_text} definition cultural meaning {context_str}"
        return self.search(query)
