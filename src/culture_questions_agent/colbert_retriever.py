"""ColBERT-style late-interaction retrieval for cultural QA.

Implements token-level matching with late interaction scoring for high-precision
retrieval of cultural knowledge. Optimized for H100 GPUs with quality over latency.
"""
import logging
import torch
import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path
import pickle

from llama_index.core.schema import NodeWithScore, TextNode, BaseNode
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import QueryBundle
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


class ColBERTRetriever(BaseRetriever):
    """
    Late-interaction retriever using ColBERT-style token embeddings.
    
    Key features:
    - Token-level embeddings for queries and documents
    - Late interaction: compute similarity at query time
    - MaxSim aggregation for token-to-token matching
    - Optimized for rare cultural terms and sub-passage matching
    """
    
    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        nodes: Optional[List[BaseNode]] = None,
        similarity_top_k: int = 10,
        device: str = "cuda",
        cache_dir: Optional[str] = None,
        index_path: Optional[str] = None,
        max_query_length: int = 128,
        max_doc_length: int = 512,
    ):
        """
        Initialize ColBERT retriever.
        
        Args:
            model_name: ColBERT model name (e.g., "colbert-ir/colbertv2.0")
            nodes: List of nodes to index
            similarity_top_k: Number of top results to return
            device: Device for computation ("cuda" or "cpu")
            cache_dir: Cache directory for models
            index_path: Path to pre-computed index (if available)
            max_query_length: Maximum query tokens
            max_doc_length: Maximum document tokens
        """
        super().__init__()
        
        logger.info("Initializing ColBERT Late-Interaction Retriever")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Top-k: {similarity_top_k}")
        
        self.model_name = model_name
        self.similarity_top_k = similarity_top_k
        self.device = device
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        
        # Load model and tokenizer
        logger.info("  Loading ColBERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        ).to(device)
        self.model.eval()
        
        logger.info(f"  ✓ Model loaded on {device}")
        
        # Index storage
        self.nodes = nodes or []
        self.doc_embeddings = []  # List of (node_id, token_embeddings)
        self.index_path = Path(index_path) if index_path else None
        
        # Load or build index
        if self.index_path and self.index_path.exists():
            logger.info(f"  Loading pre-computed index from {self.index_path}...")
            self._load_index()
            logger.info(f"  VERIFY: doc_embeddings after load: {len(self.doc_embeddings)}")
            
            # Validate that loaded embeddings match current nodes
            if self.nodes and self.doc_embeddings:
                current_node_ids = {node.node_id for node in self.nodes}
                loaded_node_ids = {doc["node_id"] for doc in self.doc_embeddings}
                
                # Check for mismatches
                common_ids = current_node_ids & loaded_node_ids
                mismatch_ratio = 1 - (len(common_ids) / len(current_node_ids))
                
                logger.info(f"  Node ID validation: {len(common_ids)}/{len(current_node_ids)} match ({(1-mismatch_ratio)*100:.1f}%)")
                
                if mismatch_ratio > 0.5:  # More than 50% mismatch
                    logger.warning(f"  WARNING: Node ID mismatch detected! {mismatch_ratio*100:.1f}% of IDs don't match")
                    logger.warning(f"  This means the index was built with different nodes.")
                    logger.warning(f"  Rebuilding ColBERT index to match current nodes...")
                    self._build_index()
                    logger.info(f"  VERIFY: doc_embeddings after rebuild: {len(self.doc_embeddings)}")
        elif self.nodes:
            logger.info(f"  Building index for {len(self.nodes)} nodes...")
            self._build_index()
            logger.info(f"  VERIFY: doc_embeddings after build: {len(self.doc_embeddings)}")
        else:
            logger.warning("  WARNING: No index to load and no nodes to build index from!")
        
        logger.info("✓ ColBERT Retriever initialized")
        logger.info(f"  Final state: {len(self.doc_embeddings)} embeddings, {len(self.nodes)} nodes")
    
    def _encode_query(self, query: str) -> torch.Tensor:
        """
        Encode query into token-level embeddings.
        
        Args:
            query: Query text
            
        Returns:
            Tensor of shape (num_tokens, embedding_dim)
        """
        # Tokenize
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            max_length=self.max_query_length,
            truncation=True,
            padding=True,
        ).to(self.device)
        
        # Get token embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use last hidden state as token embeddings
            embeddings = outputs.last_hidden_state.squeeze(0)  # (seq_len, hidden_dim)
        
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def _encode_document(self, text: str) -> torch.Tensor:
        """
        Encode document into token-level embeddings.
        
        Args:
            text: Document text
            
        Returns:
            Tensor of shape (num_tokens, embedding_dim)
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_doc_length,
            truncation=True,
            padding=True,
        ).to(self.device)
        
        # Get token embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.squeeze(0)  # (seq_len, hidden_dim)
        
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def _late_interaction_score(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
    ) -> float:
        """
        Compute late interaction score using MaxSim.
        
        MaxSim: For each query token, find max similarity to any doc token,
        then sum over all query tokens.
        
        Args:
            query_embeddings: (Q, D) query token embeddings
            doc_embeddings: (D_len, D) document token embeddings
            
        Returns:
            Late interaction score
        """
        # Compute similarity matrix: (Q, D_len)
        similarity_matrix = torch.matmul(
            query_embeddings,
            doc_embeddings.T,
        )
        
        # MaxSim: max over document tokens, sum over query tokens
        max_scores = torch.max(similarity_matrix, dim=1)[0]  # (Q,)
        score = torch.sum(max_scores).item()
        
        return score
    
    def _encode_documents_batch(self, texts: List[str], batch_size: int = 32) -> List[torch.Tensor]:
        """
        Encode multiple documents in batches for efficiency.
        
        Args:
            texts: List of document texts
            batch_size: Batch size for encoding
            
        Returns:
            List of token embedding tensors
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=self.max_doc_length,
                truncation=True,
                padding=True,
            ).to(self.device)
            
            # Get token embeddings for batch
            with torch.no_grad():
                outputs = self.model(**inputs)
                # outputs.last_hidden_state: (batch_size, seq_len, hidden_dim)
                batch_embeddings = outputs.last_hidden_state
            
            # Normalize and split by document
            for j in range(len(batch_texts)):
                doc_embeddings = batch_embeddings[j]  # (seq_len, hidden_dim)
                doc_embeddings = torch.nn.functional.normalize(doc_embeddings, p=2, dim=1)
                all_embeddings.append(doc_embeddings)
        
        return all_embeddings
    
    def _build_index(self):
        """Build ColBERT index by encoding all documents in batches."""
        logger.info("Building ColBERT index with batch encoding...")
        
        # Extract all texts
        texts = [node.get_content() for node in self.nodes]
        node_ids = [node.node_id for node in self.nodes]
        
        # Encode in batches
        batch_size = 32  # Adjust based on GPU memory
        logger.info(f"  Encoding {len(texts)} documents in batches of {batch_size}...")
        
        all_embeddings = self._encode_documents_batch(texts, batch_size=batch_size)
        
        # Store embeddings
        self.doc_embeddings = []
        for node_id, embeddings in zip(node_ids, all_embeddings):
            self.doc_embeddings.append({
                "node_id": node_id,
                "embeddings": embeddings.cpu(),  # Move to CPU to save GPU memory
            })
        
        logger.info(f"  ✓ Index built: {len(self.doc_embeddings)} documents")
        
        # Save index if path specified
        if self.index_path:
            self._save_index()
    
    def _save_index(self):
        """Save index to disk."""
        if self.index_path is None:
            logger.warning("No index_path specified, skipping save")
            return
            
        logger.info(f"Saving ColBERT index to {self.index_path}...")
        
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.doc_embeddings, f)
        
        logger.info("  ✓ Index saved")
    
    def _load_index(self):
        """Load index from disk."""
        if self.index_path is None:
            logger.warning("No index_path specified, skipping load")
            return
            
        logger.info(f"Loading ColBERT index from {self.index_path}...")
        
        with open(self.index_path, 'rb') as f:
            self.doc_embeddings = pickle.load(f)
        
        logger.info(f"  ✓ Index loaded: {len(self.doc_embeddings)} documents")
        if self.doc_embeddings:
            logger.info(f"  Sample embedding shape: {self.doc_embeddings[0]['embeddings'].shape}")
            logger.info(f"  Sample node_id: {self.doc_embeddings[0]['node_id']}")
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve documents using late interaction.
        
        Args:
            query_bundle: Query bundle
            
        Returns:
            List of NodeWithScore objects
        """
        query = query_bundle.query_str
        
        logger.debug(f"ColBERT retrieval for: '{query}'")
        logger.debug(f"  doc_embeddings count: {len(self.doc_embeddings)}")
        logger.debug(f"  nodes count: {len(self.nodes)}")
        
        if not self.doc_embeddings:
            logger.warning("  WARNING: No doc_embeddings available! Returning empty results.")
            return []
        
        # Encode query
        query_embeddings = self._encode_query(query)
        logger.debug(f"  Query embeddings shape: {query_embeddings.shape}")
        
        # Score all documents
        scores = []
        for doc_info in self.doc_embeddings:
            doc_embeddings = doc_info["embeddings"].to(self.device)
            score = self._late_interaction_score(query_embeddings, doc_embeddings)
            scores.append((doc_info["node_id"], score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k
        top_k = scores[:self.similarity_top_k]
        
        # Build NodeWithScore objects
        node_dict = {node.node_id: node for node in self.nodes}
        logger.debug(f"  node_dict size: {len(node_dict)}")
        logger.debug(f"  top_k candidates: {len(top_k)}")
        
        results = []
        matched = 0
        unmatched = 0
        
        for node_id, score in top_k:
            if node_id in node_dict:
                results.append(NodeWithScore(
                    node=node_dict[node_id],
                    score=score,
                ))
                matched += 1
            else:
                unmatched += 1
                if unmatched <= 3:  # Log first 3 unmatched
                    logger.warning(f"  WARNING: node_id {node_id[:50]} not found in node_dict")
        
        if unmatched > 0:
            logger.warning(f"  Total unmatched node_ids: {unmatched}/{len(top_k)}")
        
        logger.debug(f"  ✓ Retrieved {len(results)} results (matched: {matched}, unmatched: {unmatched})")
        if results:
            logger.debug(f"    Top score: {results[0].score:.4f}")
            logger.debug(f"    Bottom score: {results[-1].score:.4f}")
        
        return results
    
    def update_nodes(self, nodes: List[BaseNode]):
        """
        Update nodes and rebuild index.
        
        Args:
            nodes: New list of nodes
        """
        self.nodes = nodes
        self._build_index()


class ColBERTLiteRetriever(BaseRetriever):
    """
    Lightweight ColBERT-style retriever using pre-encoded embeddings.
    
    For production use when you want to pre-encode documents offline
    and only perform late interaction at query time.
    """
    
    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        embedding_index_path: Optional[str] = None,
        nodes: Optional[List[BaseNode]] = None,
        similarity_top_k: int = 10,
        device: str = "cuda",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize lightweight ColBERT retriever.
        
        Args:
            model_name: ColBERT model
            embedding_index_path: Path to pre-encoded embeddings
            nodes: List of nodes (for on-the-fly encoding)
            similarity_top_k: Top-k results
            device: Computation device
            cache_dir: Model cache directory
        """
        super().__init__()
        
        logger.info("Initializing ColBERT-Lite Retriever")
        
        self.model_name = model_name
        self.similarity_top_k = similarity_top_k
        self.device = device
        
        # Load model (query encoder only)
        logger.info("  Loading query encoder...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        ).to(device)
        self.model.eval()
        
        # Load or create index
        if embedding_index_path and Path(embedding_index_path).exists():
            logger.info(f"  Loading embedding index from {embedding_index_path}...")
            with open(embedding_index_path, 'rb') as f:
                self.doc_embeddings = pickle.load(f)
            logger.info(f"  ✓ Loaded {len(self.doc_embeddings)} document embeddings")
        else:
            logger.warning("  No embedding index found - retrieval will fail")
            self.doc_embeddings = []
        
        self.nodes = nodes or []
        
        logger.info("✓ ColBERT-Lite Retriever initialized")
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve using late interaction (same as ColBERTRetriever)."""
        # For now, raise NotImplementedError or delegate to parent class logic
        # In production, you'd copy the logic from ColBERTRetriever._retrieve
        raise NotImplementedError(
            "ColBERTLite._retrieve not yet implemented. "
            "Use ColBERTRetriever for full functionality."
        )
