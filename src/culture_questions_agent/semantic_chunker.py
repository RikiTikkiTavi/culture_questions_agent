"""Semantic chunking with multi-granularity support for answer-centric RAG.

This module implements token-aware, semantically coherent chunking at multiple
granularities to optimize for cross-encoder reranking and answer extraction.
"""
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

from llama_index.core import Document
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class GranularityConfig:
    """Configuration for a specific chunk granularity level."""
    name: str  # "small", "medium", "large"
    min_tokens: int
    max_tokens: int
    buffer_size: int  # Semantic splitter buffer
    breakpoint_percentile_threshold: int  # Semantic boundary threshold
    
    def __repr__(self):
        return f"{self.name}({self.min_tokens}-{self.max_tokens} tokens)"


class MultiGranularitySemanticChunker:
    """
    Create multiple chunk granularities from the same source content.
    
    Produces:
    - Small chunks: 200-400 tokens (facts, definitions)
    - Medium chunks: 600-900 tokens (explanations, context)
    - Large chunks: 1500-3000 tokens (comprehensive cultural context)
    
    All chunks are semantically coherent and respect discourse boundaries.
    """
    
    # Granularity presets optimized for cultural QA and reranking
    GRANULARITY_PRESETS = {
        "small": GranularityConfig(
            name="small",
            min_tokens=200,
            max_tokens=400,
            buffer_size=2,
            breakpoint_percentile_threshold=80,
        ),
        "medium": GranularityConfig(
            name="medium",
            min_tokens=600,
            max_tokens=900,
            buffer_size=3,
            breakpoint_percentile_threshold=85,
        ),
        "large": GranularityConfig(
            name="large",
            min_tokens=1500,
            max_tokens=3000,
            buffer_size=5,
            breakpoint_percentile_threshold=90,
        ),
    }
    
    def __init__(
        self,
        embedding_model_name: str,
        tokenizer_name: str = "BAAI/bge-m3",
        cache_dir: Optional[str] = None,
        granularities: Optional[List[str]] = None,
    ):
        """
        Initialize multi-granularity semantic chunker.
        
        Args:
            embedding_model_name: Model for semantic similarity
            tokenizer_name: Tokenizer for token counting
            cache_dir: Cache directory for models
            granularities: List of granularities to use (default: all)
        """
        logger.info("Initializing Multi-Granularity Semantic Chunker")
        logger.info(f"  Embedding model: {embedding_model_name}")
        logger.info(f"  Tokenizer: {tokenizer_name}")
        
        # Load embedding model for semantic splitting
        self.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model_name,
            cache_folder=cache_dir,
        )
        
        # Load tokenizer for accurate token counting
        logger.info("  Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=cache_dir,
        )
        
        # Select granularities
        self.granularities = granularities or ["small", "medium", "large"]
        logger.info(f"  Active granularities: {self.granularities}")
        
        # Initialize parsers for each granularity
        self.parsers = {}
        for granularity_name in self.granularities:
            if granularity_name not in self.GRANULARITY_PRESETS:
                raise ValueError(f"Unknown granularity: {granularity_name}")
            
            config = self.GRANULARITY_PRESETS[granularity_name]
            
            # Create semantic splitter with granularity-specific settings
            parser = SemanticSplitterNodeParser(
                buffer_size=config.buffer_size,
                breakpoint_percentile_threshold=config.breakpoint_percentile_threshold,
                embed_model=self.embed_model,
            )
            
            self.parsers[granularity_name] = {
                "parser": parser,
                "config": config,
            }
            
            logger.info(f"    ✓ {config}")
        
        logger.info("✓ Multi-Granularity Semantic Chunker initialized")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using the configured tokenizer.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    
    def chunk_document(
        self,
        document: Document,
        granularity: str,
    ) -> List[TextNode]:
        """
        Chunk a single document at a specific granularity.
        
        Args:
            document: LlamaIndex Document
            granularity: "small", "medium", or "large"
            
        Returns:
            List of TextNode objects with metadata
        """
        if granularity not in self.parsers:
            raise ValueError(f"Unknown granularity: {granularity}")
        
        parser_info = self.parsers[granularity]
        parser = parser_info["parser"]
        config = parser_info["config"]
        
        # Use semantic splitter
        nodes = parser.get_nodes_from_documents([document])
        
        # Filter and validate by token count
        valid_nodes = []
        for node in nodes:
            token_count = self.count_tokens(node.get_content())
            
            # Keep nodes within target token range
            # Allow some flexibility on max_tokens for semantic completeness
            if token_count >= config.min_tokens and token_count <= config.max_tokens * 1.5:
                # Add granularity to metadata
                node.metadata["granularity"] = granularity
                node.metadata["token_count"] = token_count
                valid_nodes.append(node)
            else:
                logger.debug(
                    f"Filtered {granularity} chunk: {token_count} tokens "
                    f"(target: {config.min_tokens}-{config.max_tokens})"
                )
        
        return valid_nodes
    
    def chunk_document_multi_granularity(
        self,
        document: Document,
        granularities: Optional[List[str]] = None,
    ) -> Dict[str, List[TextNode]]:
        """
        Chunk document at multiple granularities simultaneously.
        
        Args:
            document: LlamaIndex Document
            granularities: Specific granularities to use (default: all configured)
            
        Returns:
            Dict mapping granularity name to list of nodes
        """
        granularities = granularities or self.granularities
        
        results = {}
        for granularity in granularities:
            nodes = self.chunk_document(document, granularity)
            results[granularity] = nodes
            
            logger.debug(
                f"  [{granularity}] {len(nodes)} chunks from '{document.metadata.get('title', 'untitled')}'"
            )
        
        return results
    
    def chunk_documents_multi_granularity(
        self,
        documents: List[Document],
        granularities: Optional[List[str]] = None,
    ) -> List[TextNode]:
        """
        Chunk multiple documents at all granularities.
        
        Returns a flat list of all nodes with granularity metadata.
        
        Args:
            documents: List of LlamaIndex Documents
            granularities: Specific granularities to use (default: all)
            
        Returns:
            Flat list of all TextNode objects from all granularities
        """
        granularities = granularities or self.granularities
        
        all_nodes = []
        stats = {g: 0 for g in granularities}
        
        logger.info(f"Chunking {len(documents)} documents at {len(granularities)} granularities...")
        
        for document in tqdm(documents, desc="Chunking documents"):
            multi_grain_nodes = self.chunk_document_multi_granularity(
                document,
                granularities,
            )
            
            for granularity, nodes in multi_grain_nodes.items():
                all_nodes.extend(nodes)
                stats[granularity] += len(nodes)
        
        logger.info(f"✓ Multi-granularity chunking complete:")
        for granularity in granularities:
            logger.info(f"    {granularity}: {stats[granularity]} chunks")
        logger.info(f"  Total: {len(all_nodes)} chunks")
        
        return all_nodes


class RerankerAlignedChunker:
    """
    Simple sentence-based chunker for reranker-friendly chunks.
    
    Alternative to semantic chunking when you want more control over chunk size.
    Produces 700-1200 token chunks that are complete sentences.
    """
    
    def __init__(
        self,
        tokenizer_name: str = "BAAI/bge-m3",
        target_tokens: int = 900,
        max_tokens: int = 1200,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize reranker-aligned chunker.
        
        Args:
            tokenizer_name: Tokenizer for token counting
            target_tokens: Target chunk size
            max_tokens: Maximum chunk size
            cache_dir: Cache directory
        """
        logger.info("Initializing Reranker-Aligned Chunker")
        logger.info(f"  Target tokens: {target_tokens}")
        logger.info(f"  Max tokens: {max_tokens}")
        
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=cache_dir,
        )
        
        # Use sentence splitter as base
        self.sentence_splitter = SentenceSplitter(
            chunk_size=target_tokens * 4,  # Conservative estimate (chars ≈ 4*tokens)
            chunk_overlap=50,
        )
        
        logger.info("✓ Reranker-Aligned Chunker initialized")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    
    def chunk_documents(self, documents: List[Document]) -> List[TextNode]:
        """
        Chunk documents into reranker-friendly sizes.
        
        Args:
            documents: List of Documents
            
        Returns:
            List of TextNode objects
        """
        logger.info(f"Chunking {len(documents)} documents for reranking...")
        
        nodes = []
        for document in documents:
            # Split into sentences
            sentence_nodes = self.sentence_splitter.get_nodes_from_documents([document])
            
            # Merge sentences to reach target token count
            current_chunk = []
            current_tokens = 0
            
            for node in sentence_nodes:
                text = node.get_content()
                token_count = self.count_tokens(text)
                
                if current_tokens + token_count > self.max_tokens and current_chunk:
                    # Create chunk from accumulated sentences
                    chunk_text = " ".join(current_chunk)
                    chunk_node = TextNode(
                        text=chunk_text,
                        metadata={
                            **document.metadata,
                            "token_count": self.count_tokens(chunk_text),
                            "granularity": "reranker_aligned",
                        }
                    )
                    nodes.append(chunk_node)
                    
                    # Start new chunk
                    current_chunk = [text]
                    current_tokens = token_count
                else:
                    current_chunk.append(text)
                    current_tokens += token_count
            
            # Add remaining chunk
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk_node = TextNode(
                    text=chunk_text,
                    metadata={
                        **document.metadata,
                        "token_count": self.count_tokens(chunk_text),
                        "granularity": "reranker_aligned",
                    }
                )
                nodes.append(chunk_node)
        
        logger.info(f"✓ Created {len(nodes)} reranker-aligned chunks")
        return nodes
