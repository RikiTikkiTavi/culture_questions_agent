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
        search_engine: Optional[SearchEngine] = None,
        training_index: Optional[VectorStoreIndex] = None,
        training_nodes: Optional[List[BaseNode]] = None,
        dense_top_k: int = 50,
        sparse_top_k: int = 50,
        colbert_top_k: int = 50,
        web_top_k: int = 5,
        training_top_k: int = 10,
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        reranker_top_k: int = 10,
        cache_dir: Optional[str] = None,
        use_colbert: bool = True,
        use_dense: bool = True,
        use_sparse: bool = True,
        use_web: bool = False,
        use_training: bool = False,
        use_reranker: bool = True,
    ):
        """
        Initialize multi-retriever orchestrator.
        
        Args:
            index: VectorStoreIndex for dense retrieval
            nodes: All indexed nodes (for BM25 and ColBERT)
            colbert_retriever: Pre-initialized ColBERT retriever (optional)
            search_engine: Pre-initialized SearchEngine for web search (optional)
            training_index: VectorStoreIndex for training data (optional)
            training_nodes: All training data nodes (optional)
            dense_top_k: Top-k for dense retrieval
            sparse_top_k: Top-k for sparse retrieval
            colbert_top_k: Top-k for ColBERT retrieval
            web_top_k: Top-k for web search results
            training_top_k: Top-k for training data retrieval
            reranker_model: Cross-encoder model for reranking
            reranker_top_k: Final top-k after reranking
            cache_dir: Cache directory for models
            use_colbert: Enable ColBERT retrieval
            use_dense: Enable dense retrieval
            use_sparse: Enable sparse retrieval
            use_web: Enable web search retrieval
            use_training: Enable training data retrieval
            use_reranker: Enable cross-encoder reranking
        """
        logger.info("="*80)
        logger.info("Initializing Multi-Retriever Orchestrator")
        logger.info("="*80)
        
        self.index = index
        self.nodes = nodes
        self.training_index = training_index
        self.training_nodes = training_nodes
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.colbert_top_k = colbert_top_k
        self.web_top_k = web_top_k
        self.training_top_k = training_top_k
        self.reranker_top_k = reranker_top_k
        
        self.use_colbert = use_colbert
        self.use_dense = use_dense
        self.use_sparse = use_sparse
        self.use_web = use_web
        self.use_training = use_training
        self.use_reranker = use_reranker
        
        self.search_engine = search_engine
        
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
        
        # Web search engine
        if self.use_web:
            if search_engine:
                logger.info(f"  ✓ Web search engine (provided, max-{web_top_k} results)")
                self.search_engine = search_engine
            else:
                logger.warning(f"  ⊗ Web search enabled but no SearchEngine provided")
                self.use_web = False
                self.search_engine = None
        else:
            logger.info(f"  ⊗ Web search disabled")
            self.search_engine = None
        
        # Training data retriever
        if self.use_training:
            if training_index is not None:
                logger.info(f"  ✓ Training data retriever (top-{training_top_k})")
                self.training_retriever = VectorIndexRetriever(
                    index=training_index,
                    similarity_top_k=training_top_k,
                )
            else:
                logger.warning(f"  ⊗ Training retrieval enabled but no training index provided")
                self.use_training = False
                self.training_retriever = None
        else:
            logger.info(f"  ⊗ Training data retrieval disabled")
            self.training_retriever = None
        
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
        use_reranker: bool = True,
        use_web: bool = False,
        use_training: Optional[bool] = None,
        use_colbert: Optional[bool] = None,
        use_dense: Optional[bool] = None,
        use_sparse: Optional[bool] = None,
        training_persist_dir: Optional[str] = None,
        colbert_model: str = "colbert-ir/colbertv2.0",
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        dense_top_k: int = 50,
        sparse_top_k: int = 50,
        colbert_top_k: int = 50,
        web_top_k: int = 5,
        training_top_k: int = 10,
        reranker_top_k: int = 10,
        search_engine: Optional[SearchEngine] = None,
    ) -> "MultiRetrieverOrchestrator":
        """
        Load orchestrator from persisted directory.
        
        All settings come from config.yaml - no orchestrator_config.json needed.
        
        Args:
            persist_dir: Directory containing persisted index
            embedding_model_name: Name of embedding model to use
            cache_dir: Cache directory for models
            device: Device for ColBERT ('cuda' or 'cpu')
            use_reranker: Enable cross-encoder reranking
            use_web: Enable web search retrieval
            use_training: Enable training data retrieval (None = auto-detect from training_persist_dir)
            use_colbert: Enable ColBERT retrieval (None = default True)
            use_dense: Enable dense retrieval (None = default True)
            use_sparse: Enable sparse retrieval (None = default True)
            training_persist_dir: Directory containing training index (if use_training)
            colbert_model: ColBERT model name
            reranker_model: Reranker model name
            dense_top_k: Top-k for dense retrieval
            sparse_top_k: Top-k for sparse retrieval
            colbert_top_k: Top-k for ColBERT retrieval
            web_top_k: Top-k for web search
            training_top_k: Top-k for training retrieval
            reranker_top_k: Final top-k after reranking
            search_engine: Pre-initialized SearchEngine for web search (optional)
            
        Returns:
            Initialized MultiRetrieverOrchestrator
        """
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.core import Settings
        
        persist_path = Path(persist_dir)
        
        if not persist_path.exists():
            raise FileNotFoundError(f"Persist directory not found: {persist_dir}")
        
        logger.info(f"Loading SOTA Multi-Retriever from {persist_dir}...")
        
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
        
        # Load training index if enabled
        training_index = None
        training_nodes = None
        
        # Auto-detect training index from directory if not explicitly disabled
        if use_training is None:
            use_training = training_persist_dir is not None and Path(training_persist_dir).exists()
        
        if use_training and training_persist_dir:
            training_path = Path(training_persist_dir)
            if training_path.exists():
                logger.info(f"  Loading training data index from {training_persist_dir}...")
                try:
                    training_storage_context = StorageContext.from_defaults(
                        persist_dir=str(training_path)
                    )
                    training_index = load_index_from_storage(training_storage_context)
                    training_nodes = list(training_index.docstore.docs.values())
                    logger.info(f"  ✓ Loaded training index with {len(training_nodes)} nodes")
                except Exception as e:
                    logger.warning(f"  Failed to load training index: {e}")
                    use_training = False
            else:
                logger.warning(f"  Training index enabled but directory not found: {training_persist_dir}")
                use_training = False
        
        # Use runtime parameters for all behavior flags
        use_colbert_final = use_colbert if use_colbert is not None else True
        use_dense_final = use_dense if use_dense is not None else True
        use_sparse_final = use_sparse if use_sparse is not None else True
        
        # Load ColBERT if enabled
        colbert_retriever = None
        if use_colbert_final:
            # ColBERT index path is always persist_dir/colbert_index.pkl
            colbert_index_path = persist_path / "colbert_index.pkl"
            if colbert_index_path.exists():
                logger.info(f"  Loading ColBERT retriever from {colbert_index_path}...")
                colbert_retriever = ColBERTRetriever(
                    model_name=colbert_model,
                    nodes=nodes,
                    similarity_top_k=colbert_top_k,
                    device=device,
                    cache_dir=cache_dir,
                    index_path=str(colbert_index_path),
                )
            else:
                logger.warning(
                    f"  ⚠ ColBERT enabled in config.yaml but index not found at: {colbert_index_path}\n"
                    f"    Disabling ColBERT retrieval. Run builder.py to create the index.\n"
                    f"    To disable this warning, set retrieval.use_colbert=false in config.yaml"
                )
                use_colbert_final = False
        
        # Create orchestrator
        return cls(
            index=index,
            nodes=nodes,
            colbert_retriever=colbert_retriever,
            search_engine=search_engine,
            training_index=training_index,
            training_nodes=training_nodes,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            colbert_top_k=colbert_top_k,
            web_top_k=web_top_k,
            training_top_k=training_top_k,
            reranker_model=reranker_model,
            reranker_top_k=reranker_top_k,
            cache_dir=cache_dir,
            use_colbert=use_colbert_final,
            use_dense=use_dense_final,
            use_sparse=use_sparse_final,
            use_web=use_web,
            use_training=use_training,
            use_reranker=use_reranker,
        )
    
    def retrieve(
        self,
        query: str,
        return_sources: bool = False,
        include_options_in_web_query: bool = False,
        options: Optional[Dict[str, str]] = None,
    ) -> List[NodeWithScore]:
        """
        Retrieve documents using multi-retriever fusion including web search.
        
        Pipeline:
        1. Execute all enabled retrievers (dense, sparse, ColBERT, web)
        2. Union results (deduplicate by node_id)
        3. Rerank if enabled
        4. Return top-k
        
        Args:
            query: Search query
            return_sources: If True, return metadata about which retrievers matched each node
            include_options_in_web_query: If True and web search enabled, append options to query
            options: Options to include in web query (if include_options_in_web_query=True)
            
        Returns:
            List of NodeWithScore objects (reranked if enabled)
        """
        logger.debug("="*60)
        logger.debug(f"Multi-Retrieval Query: '{query}'")
        logger.debug("="*60)
        
        all_results = []
        source_tracking = defaultdict(list)
        training_results: List[NodeWithScore] = []
        
        # Execute dense retrieval
        if self.dense_retriever:
            logger.debug(f"[1/3] Dense Retrieval (top-{self.dense_top_k})...")
            with mlflow.start_span(name="dense_retrieval", span_type="RETRIEVER") as span:
                span.set_inputs({"query": query, "top_k": self.dense_top_k})
                dense_results = self.dense_retriever.retrieve(query)
                span.set_outputs({
                    "num_results": len(dense_results),
                    "results": [
                        {
                            "text": node.node.get_content()[:200],  # First 200 chars
                            "score": float(node.score) if node.score else None,
                            "metadata": node.node.metadata,
                        }
                        for node in dense_results[:5]  # Top 5 only to avoid large payloads
                    ]
                })
            logger.debug(f"  ✓ Retrieved {len(dense_results)} dense results")
            
            for node_with_score in dense_results:
                all_results.append(node_with_score)
                source_tracking[node_with_score.node.node_id].append("dense")
        
        # Execute sparse retrieval
        if self.sparse_retriever:
            logger.debug(f"[2/3] Sparse Retrieval (top-{self.sparse_top_k})...")
            with mlflow.start_span(name="sparse_retrieval", span_type="RETRIEVER") as span:
                span.set_inputs({"query": query, "top_k": self.sparse_top_k})
                sparse_results = self.sparse_retriever.retrieve(query)
                span.set_outputs({
                    "num_results": len(sparse_results),
                    "results": [
                        {
                            "text": node.node.get_content()[:200],
                            "score": float(node.score) if node.score else None,
                            "metadata": node.node.metadata,
                        }
                        for node in sparse_results[:5]
                    ]
                })
            logger.debug(f"  ✓ Retrieved {len(sparse_results)} sparse results")
            
            for node_with_score in sparse_results:
                all_results.append(node_with_score)
                source_tracking[node_with_score.node.node_id].append("sparse")
        
        # Execute ColBERT retrieval
        if self.colbert_retriever:
            logger.debug(f"[3/4] ColBERT Late-Interaction Retrieval (top-{self.colbert_top_k})...")
            with mlflow.start_span(name="colbert_retrieval", span_type="RETRIEVER") as span:
                span.set_inputs({"query": query, "top_k": self.colbert_top_k})
                query_bundle = QueryBundle(query_str=query)
                colbert_results = self.colbert_retriever.retrieve(query_bundle)
                span.set_outputs({
                    "num_results": len(colbert_results),
                    "results": [
                        {
                            "text": node.node.get_content()[:200],
                            "score": float(node.score) if node.score else None,
                            "metadata": node.node.metadata,
                        }
                        for node in colbert_results[:5]
                    ]
                })
            logger.debug(f"  ✓ Retrieved {len(colbert_results)} ColBERT results")
            
            for node_with_score in colbert_results:
                all_results.append(node_with_score)
                source_tracking[node_with_score.node.node_id].append("colbert")
        
        # Execute training data retrieval
        if self.training_retriever:
            logger.debug(f"[4/5] Training Data Retrieval (top-{self.training_top_k})...")
            with mlflow.start_span(name="training_retrieval", span_type="RETRIEVER") as span:
                span.set_inputs({"query": query, "top_k": self.training_top_k})
                training_results = self.training_retriever.retrieve(query)
                span.set_outputs({
                    "num_results": len(training_results),
                    "results": [
                        {
                            "text": node.node.get_content()[:200],
                            "score": float(node.score) if node.score else None,
                            "metadata": node.node.metadata,
                        }
                        for node in training_results[:5]
                    ]
                })
            logger.debug(f"  ✓ Retrieved {len(training_results)} training results")
            
            for node_with_score in training_results:
                all_results.append(node_with_score)
                source_tracking[node_with_score.node.node_id].append("training")
        
        # Execute web search
        if self.use_web and self.search_engine:
            logger.debug(f"[5/5] Web Search (top-{self.web_top_k})...")
            
            # Build query with options if configured
            web_query = query
            if include_options_in_web_query and options:
                web_query = query + " " + " ".join(options.values())
            
            logger.debug(f"  Searching: '{web_query}'")
            
            with mlflow.start_span(name="web_search", span_type="RETRIEVER") as span:
                span.set_inputs({"query": web_query, "max_results": self.web_top_k})
                # Get web search results as list to maintain individual snippets
                web_results = self.search_engine.search_web(
                    web_query,
                    max_results=self.web_top_k,
                    return_list=True
                )
                span.set_outputs({
                    "num_results": len(web_results) if web_results else 0,
                    "results": [
                        {"text": text[:200]}
                        for text in (web_results[:5] if web_results else [])
                    ]
                })
            
            if web_results:
                logger.debug(f"  ✓ Retrieved {len(web_results)} web results")
                
                # Convert web results to NodeWithScore objects
                # Use a synthetic node_id to allow deduplication
                for i, text in enumerate(web_results):
                    if isinstance(text, str) and len(text) > 20:
                        # Create a simple node from web result
                        from llama_index.core.schema import TextNode
                        
                        node = TextNode(
                            text=text,
                            id_=f"web_{hash(text)}",  # Hash for deduplication
                            metadata={"source": "web"}
                        )
                        node_with_score = NodeWithScore(node=node, score=1.0)
                        all_results.append(node_with_score)
                        source_tracking[node.node_id].append("web")
            else:
                logger.debug(f"  No web results found")
        
        # Deduplicate by node_id
        seen = set()
        unique_results = []
        for node_with_score in all_results:
            if node_with_score.node.node_id not in seen:
                seen.add(node_with_score.node.node_id)
                unique_results.append(node_with_score)
        
        logger.debug(f"[Fusion] Union: {len(all_results)} → {len(unique_results)} unique results")
        
        # Log source statistics
        multi_source_count = sum(1 for sources in source_tracking.values() if len(sources) > 1)
        logger.debug(f"  Multi-source matches: {multi_source_count}")
        
        # Rerank if enabled
        if self.reranker and unique_results:
            logger.debug(f"[Reranking] Cross-Encoder (top-{self.reranker_top_k})...")
            logger.debug(f"  Reranking {len(unique_results)} candidates...")
            
            with mlflow.start_span(name="reranking", span_type="RETRIEVER") as span:
                span.set_inputs({"query": query, "num_candidates": len(unique_results), "top_k": self.reranker_top_k})
                reranked_results = self.reranker.postprocess_nodes(
                    unique_results,
                    query_str=query,
                )
                span.set_outputs({
                    "num_results": len(reranked_results),
                    "results": [
                        {
                            "text": node.node.get_content()[:200],
                            "score": float(node.score) if node.score else None,
                            "metadata": node.node.metadata,
                        }
                        for node in reranked_results
                    ]
                })
            
            # Convert numpy float32 scores to Python float to avoid Pydantic warnings
            for node_with_score in reranked_results:
                if node_with_score.score is not None:
                    node_with_score.score = float(node_with_score.score)
            
            logger.debug(f"  ✓ Reranked → {len(reranked_results)} final results")
            
            if reranked_results:
                logger.debug(f"    Top rerank score: {reranked_results[0].score:.4f}")
                logger.debug(f"    Bottom rerank score: {reranked_results[-1].score:.4f}")
            
            final_results = reranked_results
        else:
            final_results = unique_results[:self.reranker_top_k]

        # Reserve the first two slots for training-index results (if enabled).
        # This protects short-but-useful training nodes from being filtered out
        # when mixed with much longer wiki/web nodes.
        if self.training_retriever and training_results:
            reserved_training_ids: List[str] = []
            seen_training = set()
            for nws in training_results:
                nid = nws.node.node_id
                if nid in seen_training:
                    continue
                seen_training.add(nid)
                reserved_training_ids.append(nid)
                if len(reserved_training_ids) >= 2:
                    break

            if reserved_training_ids:
                final_by_id = {nws.node.node_id: nws for nws in final_results}
                reserved_training = [
                    final_by_id.get(nid, next((n for n in training_results if n.node.node_id == nid), None))
                    for nid in reserved_training_ids
                ]
                reserved_training = [n for n in reserved_training if n is not None]

                reserved_set = set(reserved_training_ids)
                remainder = [nws for nws in final_results if nws.node.node_id not in reserved_set]
                final_results = (reserved_training + remainder)[: self.reranker_top_k]
        
        logger.debug("="*60)
        logger.debug(f"✓ Final: {len(final_results)} results")
        logger.debug("="*60)
        
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
        include_options_in_web_query: bool = False,
        options: Optional[Dict[str, str]] = None,
    ) -> List[NodeWithScore]:
        """
        Retrieve for multiple queries and optionally deduplicate.
        
        Args:
            queries: List of query strings
            deduplicate: Remove duplicate nodes across queries
            include_options_in_web_query: If True and web search enabled, append options to query
            options: Options to include in web query (if include_options_in_web_query=True)
            
        Returns:
            Combined list of NodeWithScore objects
        """
        logger.debug(f"Batch retrieval for {len(queries)} queries...")
        
        all_results = []
        for i, query in enumerate(queries, 1):
            logger.debug(f"\n--- Query {i}/{len(queries)} ---")
            results = self.retrieve(
                query,
                include_options_in_web_query=include_options_in_web_query,
                options=options
            )
            all_results.extend(results)
        
        if deduplicate:
            seen = set()
            unique_results = []
            
            # Track source statistics for deduplication logging
            source_counts = defaultdict(int)
            deduplicated_counts = defaultdict(int)
            
            for node_with_score in all_results:
                # Track source before deduplication
                source = node_with_score.node.metadata.get("source", "unknown")
                source_counts[source] += 1
                
                if node_with_score.node.node_id not in seen:
                    seen.add(node_with_score.node.node_id)
                    unique_results.append(node_with_score)
                else:
                    # Track what got deduplicated
                    deduplicated_counts[source] += 1
            
            logger.debug(
                f"\nBatch deduplication: {len(all_results)} → {len(unique_results)} unique"
            )
            
            # Log per-source deduplication stats
            for source in sorted(source_counts.keys()):
                total = source_counts[source]
                deduped = deduplicated_counts.get(source, 0)
                unique = total - deduped
                if deduped > 0:
                    logger.debug(f"  {source}: {total} → {unique} unique ({deduped} duplicates removed)")
            
            return unique_results
        
        return all_results
    
    def retrieve_texts(
        self,
        query: str,
        include_metadata: bool = False,
        include_options_in_web_query: bool = False,
        options: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """
        Retrieve and return text content only.
        
        Args:
            query: Search query
            include_metadata: Include metadata in returned text
            include_options_in_web_query: If True and web search enabled, append options to query
            options: Options to include in web query (if include_options_in_web_query=True)
            
        Returns:
            List of text strings
        """
        results = self.retrieve(
            query,
            return_sources=True,
            include_options_in_web_query=include_options_in_web_query,
            options=options
        )
        
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
        include_options_in_web_query: bool = False,
        options: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """
        Retrieve for multiple queries and return text content only.
        
        Args:
            queries: List of query strings
            deduplicate: Remove duplicate texts across queries
            include_metadata: Include metadata in returned text
            include_options_in_web_query: If True and web search enabled, append options to query
            options: Options to include in web query (if include_options_in_web_query=True)
            
        Returns:
            List of text strings
        """
        results = self.retrieve_batch(
            queries,
            deduplicate=deduplicate,
            include_options_in_web_query=include_options_in_web_query,
            options=options
        )
        
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
                "web": self.use_web,
            },
            "top_k": {
                "dense": self.dense_top_k if self.use_dense else 0,
                "sparse": self.sparse_top_k if self.use_sparse else 0,
                "colbert": self.colbert_top_k if self.use_colbert else 0,
                "web": self.web_top_k if self.use_web else 0,
            },
            "reranker": {
                "enabled": self.use_reranker,
                "top_k": self.reranker_top_k if self.use_reranker else 0,
            },
        }
        
        return stats
