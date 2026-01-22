"""Wikipedia and web search tools using DDGS (multi-engine search)."""
import logging
from typing import List, Optional
from llama_index.tools.wikipedia import WikipediaToolSpec
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from ddgs.ddgs import DDGS

logger = logging.getLogger(__name__)


class SearchEngine(BaseRetriever):
    """Search engine DDGS (multi-engine web search).
    
    Implements BaseRetriever interface for consistent integration with llama-index.
    """
    
    def __init__(self, max_chars: int = 2500, include_title: bool = True, ddgs_backend: str = "yandex,yahoo,wikipedia,grokipedia", similarity_top_k: int = 3):
        """
        Initialize search tools.
        
        Args:
            max_chars: Maximum characters to keep from search results
            include_title: Whether to include titles in search snippets
            ddgs_backend: Comma-separated list of DDGS backends to use
            similarity_top_k: Maximum number of web search results to return
        """
        super().__init__()
        logger.info("Initializing search tools...")
        self.max_chars = max_chars
        self.include_title = include_title
        self.ddgs_backend = ddgs_backend
        self.similarity_top_k = similarity_top_k
        logger.info(f"  Max search result length: {max_chars} chars")
        logger.info(f"  Include titles in snippets: {include_title}")
        logger.info(f"  DDGS backends: {ddgs_backend}")
        logger.info(f"  Top-k results: {similarity_top_k}")
        
        # Wikipedia search (primary)
        self.wikipedia_tool = WikipediaToolSpec()
        
        # DDGS web search (fallback) - uses multiple search engines
        self.ddgs: DDGS | None = None
        try:
            self.ddgs = DDGS()
            logger.info("DDGS web search initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize DDGS: {e}")


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
        
        logger.debug(f"DDGS web search: '{query}'")
        
        try:
            # Use text search from ddgs
            results = list(self.ddgs.text(query, max_results=max_results, region="us-en", backend=self.ddgs_backend),)
            
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
                        logger.debug(f"✓ DDGS found {len(cleaned_snippets)} snippets")
                        return cleaned_snippets
                    else:
                        combined_text = "\n---\n".join(cleaned_snippets)
                        logger.debug(f"✓ DDGS found {len(cleaned_snippets)} snippets")
                        return combined_text[:self.max_chars]
                    
        except Exception as e:
            logger.debug(f"DDGS search failed: {e}")
        
        return [] if return_list else ""
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve nodes using web search (required by BaseRetriever).
        
        Args:
            query_bundle: Query bundle containing the search query
            
        Returns:
            List of NodeWithScore objects
        """
        query_str = query_bundle.query_str
        logger.debug(f"Web retrieval for: '{query_str}'")
        
        web_snippets = self.search_web(query_str, max_results=self.similarity_top_k, return_list=True)
        
        if isinstance(web_snippets, list) and web_snippets:
            nodes = []
            for i, snippet in enumerate(web_snippets):
                node = TextNode(
                    text=snippet,
                    metadata={"source": "web", "query": query_str, "rank": i}
                )
                # Score decreases with rank
                score = 1.0 / (i + 1)
                nodes.append(NodeWithScore(node=node, score=score))
            return nodes
        
        return []
