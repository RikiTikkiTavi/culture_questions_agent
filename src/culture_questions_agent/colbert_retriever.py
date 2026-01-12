"""ColBERT-style late-interaction retrieval for cultural QA.

Wrapper around llama-index's ColbertIndex for cultural knowledge retrieval.
Uses the official llama-index implementation with ColBERT v2.
"""
import logging
import shutil
import tempfile
from typing import List, Optional, Any
from pathlib import Path

from llama_index.core.schema import BaseNode
from llama_index.core.retrievers import BaseRetriever
from llama_index.indices.managed.colbert import ColbertIndex
from llama_index.core.storage.storage_context import StorageContext

logger = logging.getLogger(__name__)


def build_colbert_index(
    nodes: List[BaseNode],
    model_name: str = "colbert-ir/colbertv2.0",
    index_path: str = "storage/colbert",
    index_name: str = "colbert_main",
    device: str = "cuda",
    max_query_length: int = 60,
    max_doc_length: int = 120,
    **kwargs: Any,
) -> None:
    """
    Build and persist a ColBERT index from nodes.
    
    This function should be called by builder.py to create indexes.
    
    Args:
        nodes: List of nodes to index
        model_name: ColBERT model name (e.g., "colbert-ir/colbertv2.0")
        index_path: Path where to save the index directory
        device: Device for computation ("cuda" or "cpu")
        max_query_length: Maximum query tokens (query_maxlen)
        max_doc_length: Maximum document tokens (doc_maxlen)
        **kwargs: Additional arguments passed to ColbertIndex
    """
    logger.info(f"Building ColBERT index with {len(nodes)} nodes...")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Index path: {index_path}")
    logger.info(f"  Index name: {index_name}")
    
    # Determine GPUs based on device
    gpus = 1 if device == "cuda" else 0
    
    # Build index - just pass index_path as index_name
    # ColBERT will use default index_root (storage/colbert_index)

    colbert_index = ColbertIndex(
        nodes=nodes,
        model_name=model_name,
        index_name=index_name,
        overwrite=True,
        gpus=gpus,
        doc_maxlen=max_doc_length,
        query_maxlen=max_query_length,
        **kwargs,
    )
    
    # Persist to storage/colbert_index (creates docstore.json, etc.)
    Path(index_path).mkdir(parents=True, exist_ok=True)
    colbert_index.storage_context.persist(persist_dir=index_path)
    
    shutil.rmtree("storage/colbert_index")
    logger.info(f"  ✓ Index built and persisted to {index_path}/{index_name}")


def ColBERTRetriever(
    model_name: str = "colbert-ir/colbertv2.0",
    nodes: Optional[List[BaseNode]] = None,
    similarity_top_k: int = 10,
    device: str = "cuda",
    cache_dir: Optional[str] = None,
    index_path: Optional[str] = None,
    max_query_length: int = 60,
    max_doc_length: int = 120,
    **kwargs: Any,
) -> BaseRetriever:
    """
    Load pre-built ColBERT index and return retriever.
    
    IMPORTANT: This function expects indexes to be pre-built by the builder.
    It will NOT create new indexes - use builder.py to create indexes first.
    
    Args:
        model_name: ColBERT model name (e.g., "colbert-ir/colbertv2.0") - not used for loading
        nodes: Not used - kept for API compatibility
        similarity_top_k: Number of top results to return
        device: Not used - kept for API compatibility
        cache_dir: Not used - kept for API compatibility
        index_path: Path to the pre-built index directory
                   Format: "storage/colbert_index" (directory, not .pkl file)
        max_query_length: Not used - kept for API compatibility
        max_doc_length: Not used - kept for API compatibility
        **kwargs: Additional arguments (not used)
        
    Returns:
        BaseRetriever: ColBERT retriever from llama-index
        
    Raises:
        ValueError: If index_path is not provided or index doesn't exist
    """
    logger.info("Loading pre-built ColBERT index (llama-index)")
    logger.info(f"  Top-k: {similarity_top_k}")
    logger.info(f"  Index name: {index_path}")
    
    if not index_path:
        raise ValueError("index_path is required to load pre-built ColBERT index")
    
    try:
        # Load using the proper persist_dir and index_name structure
        colbert_index = ColbertIndex.load_from_disk(
            persist_dir=index_path,
        )
        logger.info(f"  ✓ Index loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load ColBERT index '{index_path}': {e}")
    
    logger.info("✓ ColBERT Retriever initialized")
    
    # Return retriever with specified top_k
    return colbert_index.as_retriever(similarity_top_k=similarity_top_k)
