"""Reranker for selecting top relevant search results."""
import logging
from typing import List, Tuple
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class Reranker:
    """Reranker using BAAI/bge-reranker-v2-m3 to select top relevant results."""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", cache_dir: str | None = None):
        """
        Initialize reranker model.
        
        Args:
            model_name: Name of the reranker model
            cache_dir: Cache directory for model files
        """
        logger.info(f"Initializing reranker: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            device_map="auto"
        )
        
        self.model.eval()
        logger.info("✓ Reranker initialized successfully")
    
    def rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: int = 3
    ) -> List[Tuple[int, str, float]]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: The search query or question
            documents: List of document texts to rerank
            top_k: Number of top documents to return
            
        Returns:
            List of (original_index, document_text, score) tuples, sorted by score
        """
        if not documents:
            return []
        
        logger.debug(f"Reranking {len(documents)} documents, selecting top {top_k}")
        
        # Prepare pairs for reranking
        pairs = [[query, doc] for doc in documents]
        
        # Tokenize
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            ).to(self.model.device)
            
            # Get scores
            scores = self.model(**inputs, return_dict=True).logits.view(-1).float()
            scores = scores.cpu().numpy()
        
        # Create list of (index, doc, score) and sort by score
        scored_docs = [(i, doc, float(score)) for i, (doc, score) in enumerate(zip(documents, scores))]
        scored_docs.sort(key=lambda x: x[2], reverse=True)
        
        # Return top-k
        top_docs = scored_docs[:top_k]
        
        logger.debug(f"✓ Selected top {len(top_docs)} documents")
        for i, (orig_idx, doc, score) in enumerate(top_docs):
            logger.debug(f"  {i+1}. Score: {score:.4f} - {doc[:500]}...")
        
        return top_docs
