"""Hybrid retrieval using sparse (BM25) + dense (vector) search over local documents."""
import logging
from pathlib import Path
from typing import List, Tuple

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retriever combining:
    - Dense retrieval: Vector similarity search using embeddings
    - Sparse retrieval: BM25 keyword-based search
    
    Fusion strategy: Reciprocal Rank Fusion (RRF)
    """
    
    def __init__(
        self,
        persist_dir: str,
        embedding_model_name: str,
        cache_dir: str | None = None,
        sparse_top_k: int = 10,
        dense_top_k: int = 10,
        final_top_k: int = 5,
        rrf_k: int = 60,
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            persist_dir: Directory containing the persisted index
            embedding_model_name: Name of the embedding model to use
            cache_dir: Cache directory for models
            sparse_top_k: Number of results to retrieve from BM25
            dense_top_k: Number of results to retrieve from vector search
            final_top_k: Number of final results after fusion
            rrf_k: RRF constant (typically 60)
        """
        logger.info("Initializing Hybrid Retriever (Sparse + Dense)")
        logger.info(f"  Persist directory: {persist_dir}")
        logger.info(f"  Embedding model: {embedding_model_name}")
        logger.info(f"  Sparse top-k: {sparse_top_k}")
        logger.info(f"  Dense top-k: {dense_top_k}")
        logger.info(f"  Final top-k: {final_top_k}")
        
        self.sparse_top_k = sparse_top_k
        self.dense_top_k = dense_top_k
        self.final_top_k = final_top_k
        self.rrf_k = rrf_k
        
        # Load embedding model
        logger.info("Loading embedding model...")
        embed_model = HuggingFaceEmbedding(
            model_name=embedding_model_name,
            cache_folder=cache_dir,
        )
        Settings.embed_model = embed_model
        
        # Load persisted index
        persist_path = Path(persist_dir)
        if not persist_path.exists():
            raise FileNotFoundError(
                f"Index not found at {persist_dir}. "
                "Please run builder.py first to create the vector index."
            )
        
        logger.info(f"Loading index from {persist_dir}...")
        storage_context = StorageContext.from_defaults(persist_dir=str(persist_path))
        self.index = load_index_from_storage(storage_context)
        
        logger.info("Initializing retrievers...")
        
        # Initialize dense retriever (vector similarity)
        self.dense_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=dense_top_k,
        )
        
        # Initialize sparse retriever (BM25)
        # BM25 needs access to all nodes
        nodes = list(self.index.docstore.docs.values())
        logger.info(f"  BM25 indexing {len(nodes)} nodes...")
        self.sparse_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=sparse_top_k,
        )
        
        logger.info("✓ Hybrid retriever initialized successfully")
    
    def reciprocal_rank_fusion(
        self,
        sparse_results: List[NodeWithScore],
        dense_results: List[NodeWithScore],
    ) -> List[Tuple[NodeWithScore, float]]:
        """
        Combine sparse and dense results using Reciprocal Rank Fusion.
        
        RRF formula: score(d) = sum(1 / (k + rank(d))) for each retriever
        
        Args:
            sparse_results: Results from BM25
            dense_results: Results from vector search
            
        Returns:
            List of (node, fused_score) tuples, sorted by score
        """
        # Build score dictionaries
        scores = {}
        
        # Add sparse results
        for rank, node_with_score in enumerate(sparse_results, start=1):
            node_id = node_with_score.node.node_id
            rrf_score = 1.0 / (self.rrf_k + rank)
            scores[node_id] = scores.get(node_id, 0.0) + rrf_score
        
        # Add dense results
        for rank, node_with_score in enumerate(dense_results, start=1):
            node_id = node_with_score.node.node_id
            rrf_score = 1.0 / (self.rrf_k + rank)
            scores[node_id] = scores.get(node_id, 0.0) + rrf_score
        
        # Create node lookup
        node_lookup = {}
        for node_with_score in sparse_results + dense_results:
            node_lookup[node_with_score.node.node_id] = node_with_score
        
        # Sort by fused score
        fused_results = [
            (node_lookup[node_id], score)
            for node_id, score in sorted(
                scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
        ]
        
        return fused_results[:self.final_top_k]
    
    def retrieve(self, query: str) -> List[str]:
        """
        Retrieve relevant documents using hybrid search.
        
        Args:
            query: Search query
            
        Returns:
            List of retrieved document texts
        """
        logger.info(f"Hybrid retrieval for: '{query}'")
        
        # Execute sparse retrieval (BM25)
        logger.info(f"  [Sparse] BM25 retrieval (top-{self.sparse_top_k})...")
        sparse_results = self.sparse_retriever.retrieve(query)
        logger.info(f"    Retrieved {len(sparse_results)} results")
        
        # Execute dense retrieval (vector similarity)
        logger.info(f"  [Dense] Vector retrieval (top-{self.dense_top_k})...")
        dense_results = self.dense_retriever.retrieve(query)
        logger.info(f"    Retrieved {len(dense_results)} results")
        
        # Fuse results using RRF
        logger.info(f"  [Fusion] Reciprocal Rank Fusion (k={self.rrf_k})...")
        fused_results = self.reciprocal_rank_fusion(sparse_results, dense_results)
        logger.info(f"    Final top-{len(fused_results)} results selected")
        
        # Extract text from nodes
        documents = []
        for i, (node_with_score, rrf_score) in enumerate(fused_results, start=1):
            text = node_with_score.node.get_content()
            metadata = node_with_score.node.metadata
            source = metadata.get("source", "unknown")
            title = metadata.get("title", "")
            
            logger.info(
                f"    {i}. RRF={rrf_score:.4f} | {source} | {title[:50]} | "
                f"{text[:100]}..."
            )
            documents.append(text)
        
        return documents
    
    def retrieve_batch(self, queries: List[str]) -> List[str]:
        """
        Retrieve documents for multiple queries and deduplicate.
        
        Args:
            queries: List of search queries
            
        Returns:
            Deduplicated list of retrieved documents
        """
        all_docs = []
        
        for query in queries:
            docs = self.retrieve(query)
            all_docs.extend(docs)
        
        # Deduplicate while preserving order
        seen = set()
        deduplicated = []
        for doc in all_docs:
            if doc not in seen:
                seen.add(doc)
                deduplicated.append(doc)
        
        if len(all_docs) > len(deduplicated):
            logger.info(
                f"Deduplicated: {len(all_docs)} → {len(deduplicated)} documents"
            )
        
        return deduplicated
