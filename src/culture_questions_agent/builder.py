"""Index builder for Cultural QA RAG system."""
import os
from pathlib import Path
from typing import List

import pycountry
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.wikipedia import WikipediaReader


def build_atlas(cfg) -> VectorStoreIndex:
    """
    Build a VectorStoreIndex from Wikipedia pages.
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        VectorStoreIndex: The built index
    """
    print("="*80)
    print("Building Cultural Knowledge Atlas...")
    print("="*80)
    
    # Set cache directory for HuggingFace models
    os.environ["HF_HOME"] = cfg.model.cache_dir
    
    # Initialize embedding model
    print(f"\n[1/4] Loading embedding model: {cfg.model.embed_name}")
    embed_model = HuggingFaceEmbedding(
        model_name=cfg.model.embed_name,
        cache_folder=cfg.model.cache_dir,
    )
    
    # Configure global settings
    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    
    # Generate topics from templates using pycountry
    print(f"\n[2/4] Generating topics from templates using pycountry...")
    topics = []
    
    for country in pycountry.countries:
        # Use common_name if available, else name
        c_name = getattr(country, 'common_name', country.name)
        
        for template in cfg.target_topic_templates:
            topics.append(template.format(c_name))
    
    print(f"  ✓ Generated {len(topics)} topics from {len(cfg.target_topic_templates)} templates and {len(list(pycountry.countries))} countries")
    
    # Load Wikipedia pages
    print(f"\n[3/4] Loading Wikipedia pages...")
    reader = WikipediaReader()
    documents = []
    
    for topic in topics:
        try:
            print(f"  - Fetching: {topic}")
            docs = reader.load_data(pages=[topic], auto_suggest=False)
            documents.extend(docs)
        except Exception as e:
            print(f"  ⚠ Warning: Failed to load '{topic}': {e}")
    
    if not documents:
        raise ValueError("No documents were successfully loaded!")
    
    print(f"\n  ✓ Loaded {len(documents)} documents")
    
    # Build index
    print(f"\n[4/4] Building vector index...")
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True,
    )
    
    # Persist index
    persist_dir = Path(cfg.storage.persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[5/5] Persisting index to: {persist_dir}")
    index.storage_context.persist(persist_dir=str(persist_dir))
    
    print("\n" + "="*80)
    print("✓ Atlas built successfully!")
    print("="*80 + "\n")
    
    return index
