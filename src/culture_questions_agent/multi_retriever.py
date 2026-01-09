"""Multi-retriever orchestration for SOTA cultural QA retrieval.

Orchestrates ColBERT (late-interaction), dense (BGE-M3), and sparse (BM25)
retrievers with fusion and cross-encoder reranking for maximum answer quality.
"""
import json
import logging
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

logger = logging.getLogger(__name__)


class MultiRetrieverOrchestrator:
    """
    Orchestrates multiple retrieval strategies for maximum recall.
    
    Retrieval Pipeline:
    1. ColBERT (late-interaction): High-precision token matching
    2. Dense (BGE-M3): Semantic similarity for broad recall
    3. Sparse (BM25): Lexical matching for rare terms
    4. Fusion: Union all results
    5. Reranking: Cross-encoder scores final candidates
    
    Optimized for answer quality over latency.
    """
    
    def __init__(
        self,
        index: VectorStoreIndex,
        nodes: List[BaseNode],
        colbert_retriever: Optional[ColBERTRetriever] = None,
        dense_top_k: int = 50,
        sparse_top_k: int = 50,
        colbert_top_k: int = 50,
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        reranker_top_k: int = 10,
        cache_dir: Optional[str] = None,
        use_colbert: bool = True,
        use_dense: bool = True,
        use_sparse: bool = True,
        use_reranker: bool = True,
    ):
        """
        Initialize multi-retriever orchestrator.
        
        Args:
            index: VectorStoreIndex for dense retrieval
            nodes: All indexed nodes (for BM25 and ColBERT)
            colbert_retriever: Pre-initialized ColBERT retriever (optional)
            dense_top_k: Top-k for dense retrieval
            sparse_top_k: Top-k for sparse retrieval
            colbert_top_k: Top-k for ColBERT retrieval
            reranker_model: Cross-encoder model for reranking
            reranker_top_k: Final top-k after reranking
            cache_dir: Cache directory for models
            use_colbert: Enable ColBERT retrieval
            use_dense: Enable dense retrieval
            use_sparse: Enable sparse retrieval
            use_reranker: Enable cross-encoder reranking
        """
        logger.info("="*80)
        logger.info("Initializing Multi-Retriever Orchestrator")
        logger.info("="*80)
        
        self.index = index
        self.nodes = nodes
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.colbert_top_k = colbert_top_k
        self.reranker_top_k = reranker_top_k
        
        self.use_colbert = use_colbert
        self.use_dense = use_dense
        self.use_sparse = use_sparse
        self.use_reranker = use_reranker
        
        # Initialize retrievers
        logger.info(f"[1/4] Initializing Retrievers")
        
        # Dense retriever (vector similarity)
        if self.use_dense:
            logger.info(f"  ✓ Dense retriever (BGE-M3, top-{dense_top_k})")
            self.dense_retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=dense_top_k,
            )
        else:
            logger.info(f"  ⊗ Dense retriever disabled")
            self.dense_retriever = None
        
        # Sparse retriever (BM25)
        if self.use_sparse:
            logger.info(f"  ✓ Sparse retriever (BM25, top-{sparse_top_k})")
            logger.info(f"    Indexing {len(nodes)} nodes...")
            self.sparse_retriever = BM25Retriever.from_defaults(
                nodes=nodes,
                similarity_top_k=sparse_top_k,
            )
        else:
            logger.info(f"  ⊗ Sparse retriever disabled")
            self.sparse_retriever = None
        
        # ColBERT retriever (late-interaction)
        if self.use_colbert:
            if colbert_retriever:
                logger.info(f"  ✓ ColBERT retriever (provided, top-{colbert_top_k})")
                self.colbert_retriever = colbert_retriever
            else:
                logger.info(f"  ✓ ColBERT retriever (initializing, top-{colbert_top_k})")
                logger.warning(
                    "    WARNING: Initializing ColBERT on-the-fly. "
                    "For production, pre-initialize and pass via constructor."
                )
                self.colbert_retriever = ColBERTRetriever(
                    nodes=nodes,
                    similarity_top_k=colbert_top_k,
                    cache_dir=cache_dir,
                )
        else:
            logger.info(f"  ⊗ ColBERT retriever disabled")
            self.colbert_retriever = None
        
        # Reranker (cross-encoder)
        if self.use_reranker:
            logger.info(f"[2/4] Initializing Reranker")
            logger.info(f"  Model: {reranker_model}")
            logger.info(f"  Top-k: {reranker_top_k}")
            
            self.reranker = SentenceTransformerRerank(
                model=reranker_model,
                top_n=reranker_top_k,
            )
        else:
            logger.info(f"[2/4] Reranker disabled")
            self.reranker = None
        
        logger.info("="*80)
        logger.info("✓ Multi-Retriever Orchestrator Ready")
        logger.info("="*80)
    
    @classmethod
    def from_persist_dir(
        cls,
        persist_dir: str,
        embedding_model_name: str = "BAAI/bge-m3",
        cache_dir: Optional[str] = None,
        device: str = "cuda",
    ) -> "MultiRetrieverOrchestrator":
        """
        Load orchestrator from persisted directory.
        
        Args:
            persist_dir: Directory containing persisted index and config
            embedding_model_name: Name of embedding model to use
            cache_dir: Cache directory for models
            device: Device for ColBERT ('cuda' or 'cpu')
            
        Returns:
            Initialized MultiRetrieverOrchestrator
        """
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.core import Settings
        
        persist_path = Path(persist_dir)
        
        if not persist_path.exists():
            raise FileNotFoundError(f"Persist directory not found: {persist_dir}")
        
        # Load orchestrator configuration
        config_path = persist_path / "orchestrator_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Orchestrator config not found: {config_path}\n"
                "Please rebuild the index with the latest builder.py"
            )
        
        logger.info(f"Loading SOTA Multi-Retriever from {persist_dir}...")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Initialize embedding model BEFORE loading index to avoid OpenAI default
        logger.info(f"  Initializing embedding model: {embedding_model_name}...")
        embed_model = HuggingFaceEmbedding(
            model_name=embedding_model_name,
            cache_folder=cache_dir,
        )
        Settings.embed_model = embed_model
        
        # Load index
        logger.info("  Loading vector index...")
        storage_context = StorageContext.from_defaults(persist_dir=str(persist_path))
        index = load_index_from_storage(storage_context)
        
        # Ensure it's a VectorStoreIndex
        if not isinstance(index, VectorStoreIndex):
            raise TypeError(f"Loaded index is not a VectorStoreIndex: {type(index)}")
        
        # Get all nodes
        nodes = list(index.docstore.docs.values())
        logger.info(f"  Loaded {len(nodes)} nodes")
        
        # Initialize ColBERT if enabled
        colbert_retriever = None
        if config.get('use_colbert', False):
            colbert_index_path = config.get('colbert_index_path')
            if colbert_index_path and Path(colbert_index_path).exists():
                logger.info(f"  Loading ColBERT retriever...")
                colbert_retriever = ColBERTRetriever(
                    model_name=config.get('colbert_model', 'colbert-ir/colbertv2.0'),
                    nodes=nodes,
                    similarity_top_k=config.get('colbert_top_k', 50),
                    device=device,
                    cache_dir=cache_dir,
                    index_path=colbert_index_path,
                )
            else:
                logger.warning(f"  ColBERT enabled but index not found, disabling ColBERT")
                config['use_colbert'] = False
        
        # Create orchestrator
        return cls(
            index=index,
            nodes=nodes,
            colbert_retriever=colbert_retriever,
            dense_top_k=config.get('dense_top_k', 50),
            sparse_top_k=config.get('sparse_top_k', 50),
            colbert_top_k=config.get('colbert_top_k', 50),
            reranker_model=config.get('reranker_model', 'BAAI/bge-reranker-v2-m3'),
            reranker_top_k=config.get('reranker_top_k', 10),
            cache_dir=cache_dir,
            use_colbert=config.get('use_colbert', True),
            use_dense=config.get('use_dense', True),
            use_sparse=config.get('use_sparse', True),
            use_reranker=config.get('use_reranker', True),
        )
    
    def retrieve(
        self,
        query: str,
        return_sources: bool = False,
    ) -> List[NodeWithScore]:
        """
        Retrieve documents using multi-retriever fusion.
        
        Pipeline:
        1. Execute all enabled retrievers
        2. Union results (deduplicate by node_id)
        3. Rerank if enabled
        4. Return top-k
        
        Args:
            query: Search query
            return_sources: If True, return metadata about which retrievers matched each node
            
        Returns:
            List of NodeWithScore objects (reranked if enabled)
        """
        logger.info("="*60)
        logger.info(f"Multi-Retrieval Query: '{query}'")
        logger.info("="*60)
        
        all_results = []
        source_tracking = defaultdict(list)
        
        # Execute dense retrieval
        if self.dense_retriever:
            logger.info(f"[1/3] Dense Retrieval (top-{self.dense_top_k})...")
            dense_results = self.dense_retriever.retrieve(query)
            logger.info(f"  ✓ Retrieved {len(dense_results)} dense results")
            
            for node_with_score in dense_results:
                all_results.append(node_with_score)
                source_tracking[node_with_score.node.node_id].append("dense")
        
        # Execute sparse retrieval
        if self.sparse_retriever:
            logger.info(f"[2/3] Sparse Retrieval (top-{self.sparse_top_k})...")
            sparse_results = self.sparse_retriever.retrieve(query)
            logger.info(f"  ✓ Retrieved {len(sparse_results)} sparse results")
            
            for node_with_score in sparse_results:
                all_results.append(node_with_score)
                source_tracking[node_with_score.node.node_id].append("sparse")
        
        # Execute ColBERT retrieval
        if self.colbert_retriever:
            logger.info(f"[3/3] ColBERT Late-Interaction Retrieval (top-{self.colbert_top_k})...")
            query_bundle = QueryBundle(query_str=query)
            colbert_results = self.colbert_retriever.retrieve(query_bundle)
            logger.info(f"  ✓ Retrieved {len(colbert_results)} ColBERT results")
            
            for node_with_score in colbert_results:
                all_results.append(node_with_score)
                source_tracking[node_with_score.node.node_id].append("colbert")
        
        # Deduplicate by node_id
        seen = set()
        unique_results = []
        for node_with_score in all_results:
            if node_with_score.node.node_id not in seen:
                seen.add(node_with_score.node.node_id)
                unique_results.append(node_with_score)
        
        logger.info(f"[Fusion] Union: {len(all_results)} → {len(unique_results)} unique results")
        
        # Log source statistics
        multi_source_count = sum(1 for sources in source_tracking.values() if len(sources) > 1)
        logger.info(f"  Multi-source matches: {multi_source_count}")
        
        # Rerank if enabled
        if self.reranker and unique_results:
            logger.info(f"[Reranking] Cross-Encoder (top-{self.reranker_top_k})...")
            logger.info(f"  Reranking {len(unique_results)} candidates...")
            
            reranked_results = self.reranker.postprocess_nodes(
                unique_results,
                query_str=query,
            )
            
            logger.info(f"  ✓ Reranked → {len(reranked_results)} final results")
            
            if reranked_results:
                logger.info(f"    Top rerank score: {reranked_results[0].score:.4f}")
                logger.info(f"    Bottom rerank score: {reranked_results[-1].score:.4f}")
            
            final_results = reranked_results
        else:
            final_results = unique_results[:self.reranker_top_k]
        
        logger.info("="*60)
        logger.info(f"✓ Final: {len(final_results)} results")
        logger.info("="*60)
        
        # Optionally attach source metadata
        if return_sources:
            for node_with_score in final_results:
                node_id = node_with_score.node.node_id
                sources = source_tracking.get(node_id, [])
                node_with_score.node.metadata["retrieval_sources"] = sources
        
        return final_results
    
    def retrieve_batch(
        self,
        queries: List[str],
        deduplicate: bool = True,
    ) -> List[NodeWithScore]:
        """
        Retrieve for multiple queries and optionally deduplicate.
        
        Args:
            queries: List of query strings
            deduplicate: Remove duplicate nodes across queries
            
        Returns:
            Combined list of NodeWithScore objects
        """
        logger.info(f"Batch retrieval for {len(queries)} queries...")
        
        all_results = []
        for i, query in enumerate(queries, 1):
            logger.info(f"\n--- Query {i}/{len(queries)} ---")
            results = self.retrieve(query)
            all_results.extend(results)
        
        if deduplicate:
            seen = set()
            unique_results = []
            for node_with_score in all_results:
                if node_with_score.node.node_id not in seen:
                    seen.add(node_with_score.node.node_id)
                    unique_results.append(node_with_score)
            
            logger.info(
                f"\nBatch deduplication: {len(all_results)} → {len(unique_results)} unique"
            )
            return unique_results
        
        return all_results
    
    def retrieve_texts(
        self,
        query: str,
        include_metadata: bool = False,
    ) -> List[str]:
        """
        Retrieve and return text content only.
        
        Args:
            query: Search query
            include_metadata: Include metadata in returned text
            
        Returns:
            List of text strings
        """
        results = self.retrieve(query, return_sources=True)
        
        texts = []
        for node_with_score in results:
            text = node_with_score.node.get_content()
            
            if include_metadata:
                metadata = node_with_score.node.metadata
                text = f"[{metadata.get('source', 'unknown')}] {text}"
            
            texts.append(text)
        
        return texts
    
    def retrieve_texts_batch(
        self,
        queries: List[str],
        deduplicate: bool = True,
        include_metadata: bool = False,
    ) -> List[str]:
        """
        Retrieve for multiple queries and return text content only.
        
        Args:
            queries: List of query strings
            deduplicate: Remove duplicate texts across queries
            include_metadata: Include metadata in returned text
            
        Returns:
            List of text strings
        """
        results = self.retrieve_batch(queries, deduplicate=deduplicate)
        
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
            "total_nodes": len(self.nodes),
            "retrievers": {
                "dense": self.use_dense,
                "sparse": self.use_sparse,
                "colbert": self.use_colbert,
            },
            "top_k": {
                "dense": self.dense_top_k if self.use_dense else 0,
                "sparse": self.sparse_top_k if self.use_sparse else 0,
                "colbert": self.colbert_top_k if self.use_colbert else 0,
            },
            "reranker": {
                "enabled": self.use_reranker,
                "top_k": self.reranker_top_k if self.use_reranker else 0,
            },
        }
        
        return stats
