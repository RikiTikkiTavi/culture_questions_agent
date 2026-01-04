"""Index builder for Cultural QA RAG system."""
import logging
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
from tqdm import tqdm

logger = logging.getLogger(__name__)

def build_atlas(cfg) -> VectorStoreIndex:
    """
    Build a VectorStoreIndex from Wikipedia pages.
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        VectorStoreIndex: The built index
    """
    logger.info("="*80)
    logger.info("Building Cultural Knowledge Atlas...")
    logger.info("="*80)
    
    # Set cache directory for HuggingFace models
    os.environ["HF_HOME"] = cfg.model.cache_dir
    
    # Initialize embedding model
    logger.info(f"[1/4] Loading embedding model: {cfg.model.embed_name}")
    embed_model = HuggingFaceEmbedding(
        model_name=cfg.model.embed_name,
        cache_folder=cfg.model.cache_dir,
    )
    
    # Configure global settings
    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    
    # Generate topics from templates using pycountry
    logger.info(f"[2/4] Generating topics from templates using pycountry...")
    topics = []
    
    OVERRIDES = {
        "Viet Nam": "Vietnam",
        "Russian Federation": "Russia",
        "Korea, Republic of": "South Korea",
        "Korea, Democratic People's Republic of": "North Korea",
        "Lao People's Democratic Republic": "Laos",
        "Syrian Arab Republic": "Syria",
        "Iran, Islamic Republic of": "Iran",
        "Tanzania, United Republic of": "Tanzania",
        "Micronesia, Federated States of": "Micronesia",
        "Moldova, Republic of": "Moldova",
        "Venezuela, Bolivarian Republic of": "Venezuela",
        "Brunei Darussalam": "Brunei",
        "Eswatini": "Swaziland" # Wikipedia often redirects, but Swaziland is safer for older dumps
    }

    for country in pycountry.countries:
        # Use official_name if available, else official name
        c_name = getattr(country, "name", None)

        if c_name is None:
            logger.warning("Skipping country with no name")
            continue
        
        for template in cfg.target_topic_templates:
            topics.append(template.format(c_name))
    
    logger.info(f"  ✓ Generated {len(topics)} topics from {len(cfg.target_topic_templates)} templates and {len(list(pycountry.countries))} countries")
    
    # Load Wikipedia pages
    logger.info(f"[3/4] Loading Wikipedia pages...")
    reader = WikipediaReader()
    documents = []
    
    for topic in tqdm(topics, desc="Loading Wikipedia Pages"):
        try:
            docs = reader.load_data(pages=[topic], auto_suggest=False)
            documents.extend(docs)
            logger.debug(f"Fetched: {topic}")
        except Exception as e:
            logger.warning(f"Failed to load '{topic}': {e}")
    
    if not documents:
        raise ValueError("No documents were successfully loaded!")
    
    logger.info(f"  ✓ Loaded {len(documents)} documents")
    
    # Build index
    logger.info(f"[4/4] Building vector index...")
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True,
    )
    
    # Persist index
    persist_dir = Path(cfg.storage.persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"[5/5] Persisting index to: {persist_dir}")
    index.storage_context.persist(persist_dir=str(persist_dir))
    
    logger.info("="*80)
    logger.info("✓ Atlas built successfully!")
    logger.info("="*80)
    
    return index
