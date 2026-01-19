"""Multi-retriever orchestration for SOTA cultural QA retrieval.

Orchestrates ColBERT (late-interaction), dense (BGE-M3), and sparse (BM25)
retrievers with fusion and cross-encoder reranking for maximum answer quality.
"""
import json
import logging
import mlflow
from pathlib import Path
from typing import List, Optional, Dict, Any
from collections import defaultdict

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.schema import NodeWithScore, BaseNode
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever
from llama_index.core import QueryBundle
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import SentenceTransformerRerank

from culture_questions_agent.colbert_retriever import ColBERTRetriever
from culture_questions_agent.search_tools import SearchEngine

from mlflow.entities import Document

logger = logging.getLogger(__name__)


class MultiRetrieverOrchestrator:
    """
    Orchestrates multiple retrieval strategies following dependency inversion.
    
    Accepts a list of BaseRetriever objects and orchestrates their execution,
    combining results through fusion.
    
    Retrieval Pipeline:
    1. Execute all provided retrievers in parallel
    2. Fusion: Union all results (deduplicated)
    
    Optimized for answer quality over latency.
    """
    
    def __init__(
        self,
        retrievers: List[BaseRetriever],
        names: list[str],
    ):
        """
        Initialize multi-retriever orchestrator.
        
        Args:
            retrievers: List of BaseRetriever instances to orchestrate
        """
        logger.info("="*80)
        logger.info("Initializing Multi-Retriever Orchestrator")
        logger.info("="*80)
        
        self.retrievers = retrievers
        self.names = names
        
        logger.info(f"Configured with {len(retrievers)} retrievers:")
        for i, retriever in enumerate(retrievers, 1):
            retriever_name = retriever.__class__.__name__
            logger.info(f"  [{i}] {retriever_name}")
        
        logger.info("="*80)
        logger.info("✓ Multi-Retriever Orchestrator Ready")
        logger.info("="*80)
    
    def _deduplicate(self, nodes: list[NodeWithScore]) -> list[NodeWithScore]:
        """Deduplicate NodeWithScore list by node_id."""
        seen = set()
        unique_nodes = []
        for nws in nodes:
            if nws.node.node_id not in seen:
                seen.add(nws.node.node_id)
                unique_nodes.append(nws)
        return unique_nodes

    def _deduplicate_by_content(self, nodes: list[NodeWithScore]) -> list[NodeWithScore]:
        """Deduplicate NodeWithScore list by content."""
        seen = set()
        unique_nodes = []
        for nws in nodes:
            content = nws.node.get_content()
            if content not in seen:
                seen.add(content)
                unique_nodes.append(nws)
        return unique_nodes

    def retrieve(
        self,
        query: str,
    ) -> List[NodeWithScore]:
        """
        Retrieve documents using multi-retriever fusion.
        
        Pipeline:
        1. Execute all retrievers
        2. Union results (deduplicate by node_id)
        3. Return results
        
        Args:
            query: Search query
            
        Returns:
            List of NodeWithScore objects
        """
        logger.debug("="*60)
        logger.debug(f"Multi-Retrieval Query: '{query}'")
        logger.debug("="*60)
        
        all_results = []
        source_tracking = defaultdict(list)
        
        # Execute all retrievers
        for i, retriever in enumerate(self.retrievers):
            retriever_name = self.names[i]            
            try:
                # All retrievers implement BaseRetriever.retrieve()
                results = retriever.retrieve(query)
                results = self._deduplicate(results)
                
                logger.info(f"✓ Retrieved {len(results)} results from retriever={retriever_name} for question='{query}'")
                
                # Track sources
                for node_with_score in results:
                    all_results.append(node_with_score)
                    source_tracking[node_with_score.node.node_id].append(retriever_name)
                    
            except Exception as e:
                logger.error(f"  ✗ {retriever_name} failed: {e}")
                continue
        
        # Deduplicate all results by node_id
        final_results = self._deduplicate(all_results)
        logger.info(f"Removed {len(all_results) - len(final_results)} duplicate results; {len(final_results)} unique results remain.")
        
        # Attach source metadata
        for node_with_score in final_results:
            node_id = node_with_score.node.node_id
            sources = source_tracking[node_id]
            node_with_score.node.metadata["source"] = sources
        
        logger.debug("="*60)
        logger.debug(f"✓ Final: {len(final_results)} results")
        logger.debug("="*60)
        
        return final_results
    
    def retrieve_batch(
        self,
        queries: List[str],
    ) -> List[NodeWithScore]:
        """
        Retrieve for multiple queries and deduplicate.
        
        Args:
            queries: List of query strings
            
        Returns:
            Combined list of NodeWithScore objects
        """
        logger.debug(f"Batch retrieval for {len(queries)} queries...")
        
        all_results = []
        for i, query in enumerate(queries, 1):
            logger.debug(f"\n--- Query {i}/{len(queries)} ---")
            results = self.retrieve(query)
            all_results.extend(results)
        
        len_before = len(all_results)
        all_results = self._deduplicate(all_results)
        
        logger.debug(
            f"\nBatch deduplication: {len_before} → {len(all_results)} unique"
        )
        
        return all_results

    
    def retrieve_texts_batch(
        self,
        queries: List[str],
        include_metadata: bool = False,
    ) -> List[str]:
        """
        Retrieve for multiple queries and return text content only.
        
        Args:
            queries: List of query strings
            include_metadata: Include metadata in returned text
            
        Returns:
            List of text strings
        """
        results = self.retrieve_batch(queries)
        
        texts = []
        for node_with_score in results:
            text = node_with_score.node.get_content()
            
            if include_metadata:
                metadata = node_with_score.node.metadata
                text = f"[{metadata.get('source', 'unknown')}] {text}"
            
            texts.append(text)
        
        return texts
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the retrieval system.
        
        Returns:
            Dictionary with stats
        """
        stats = {
            "num_retrievers": len(self.retrievers),
            "retrievers": [r.__class__.__name__ for r in self.retrievers],
        }
        
        return stats
