"""Index builder for Cultural QA RAG system."""
import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple

import hydra
import pycountry
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    Document,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.wikipedia import WikipediaReader
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Country name overrides for Wikipedia-friendly names
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
    "Eswatini": "Swaziland",
    "United Arab Emirates": "UAE",
}


def get_country_list() -> List[str]:
    """
    Get list of standardized country names.
    
    Returns:
        List of country names with overrides applied
    """
    countries = []
    for country in pycountry.countries:
        c_name = getattr(country, "name", None)
        if c_name is None:
            continue
        # Apply overrides for Wikipedia-friendly names
        c_name = OVERRIDES.get(c_name, c_name)
        countries.append(c_name)
    return countries


def split_into_passages(text: str, chunk_size: int = 1000, step_size: int = 500) -> List[str]:
    """
    Split text into overlapping passages using sliding window.
    
    Args:
        text: Text to split
        chunk_size: Size of each chunk in characters
        step_size: Size of overlap with previous chunk in characters
        
    Returns:
        List of text passages
    """
    text = text.strip()
    if not text:
        return []

    passages = []
    start = 0
    length = len(text)

    while start < length:
        end = start + chunk_size
        passages.append(text[start:end].strip())
        start += step_size

    return passages


def clean_wiki_markup(text: str) -> str:
    """
    Clean common Wiki markup from text.
    
    Args:
        text: Raw wiki text
        
    Returns:
        Cleaned text
    """
    # Remove templates
    text_clean = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)
    # Remove category, file, and namespaces
    text_clean = re.sub(r"\[\[[A-Za-z]+:[^\]]+\]\]", "", text_clean)
    # Handle wiki links with alternate text
    text_clean = re.sub(r"\[\[[^\|\]]*\|([^\]]+)\]\]", r"\1", text_clean)
    # Handle simple wiki links
    text_clean = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text_clean)
    # Remove bold/italic markup
    text_clean = re.sub(r"'{2,}", "", text_clean)
    
    return text_clean


def load_wikivoyage_documents(xml_path: str, chunk_size: int = 1000, step_size: int = 500) -> List[Document]:
    """
    Load and parse Wikivoyage XML dump into LlamaIndex documents.
    
    Args:
        xml_path: Path to Wikivoyage XML dump
        chunk_size: Size of each chunk in characters
        step_size: Size of overlap with previous chunk in characters
        
    Returns:
        List of LlamaIndex Document objects
    """
    logger.info(f"Loading Wikivoyage from: {xml_path}")
    
    if not os.path.exists(xml_path):
        logger.warning(f"Wikivoyage file not found: {xml_path}")
        return []
    
    ns = {"mw": "http://www.mediawiki.org/xml/export-0.11/"}
    documents = []
    passage_id = 0
    
    # Parse XML
    context = ET.iterparse(xml_path, events=("start", "end"))
    _, root = next(context)
    
    for event, elem in tqdm(context, desc="Parsing Wikivoyage XML"):
        if event == "end" and elem.tag.endswith("page"):
            title = elem.find("mw:title", ns)
            revision = elem.find("mw:revision", ns)
            text = revision.find("mw:text", ns) if revision is not None else None
            
            title = title.text if title is not None else None
            text = text.text if text is not None else ""
            
            # Skip empty pages and redirects
            if not text or text.strip().upper().startswith("#REDIRECT"):
                elem.clear()
                root.clear()
                continue
            
            # Clean wiki markup
            text_clean = clean_wiki_markup(text)
            
            # Split article into passages
            passages = split_into_passages(text_clean, chunk_size, step_size)
            
            for passage in passages:
                if passage.strip():
                    doc = Document(
                        text=passage.strip(),
                        metadata={
                            "source": "wikivoyage",
                            "title": title,
                            "passage_id": passage_id,
                        }
                    )
                    documents.append(doc)
                    passage_id += 1
            
            elem.clear()
            root.clear()
    
    logger.info(f"  ✓ Loaded {len(documents)} passages from Wikivoyage")
    return documents


def load_wikipedia_documents(cfg) -> Tuple[List[Document], Dict]:
    """
    Load Wikipedia pages based on country topics.
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        Tuple of (List of LlamaIndex Document objects, extraction report dict)
    """
    logger.info("Generating topics from templates using pycountry...")
    topics = []
    countries = get_country_list()
    
    # Hardcoded topic templates
    templates = [
        "Culture of {}",
        "Public holidays in {}",
        "Cuisine of {}",
        "Education in {}",
        "Music of {}",
    ]
    
    for country in countries[:10]:
        for template in templates:
            topics.append(template.format(country))
    
    logger.info(f"  ✓ Generated {len(topics)} topics from {len(templates)} templates and {len(countries)} countries")
    
    # Load Wikipedia pages
    logger.info("Loading Wikipedia pages...")
    reader = WikipediaReader()
    documents = []
    successful_pages = []
    failed_pages = []
    
    for topic in tqdm(topics, desc="Loading Wikipedia Pages"):
        try:
            docs = reader.load_data(pages=[topic], auto_suggest=cfg.vector_store.auto_suggest)
            # Add source metadata
            for doc in docs:
                doc.metadata["source"] = "wikipedia"
            documents.extend(docs)
            successful_pages.append(topic)
            logger.debug(f"Fetched: {topic}")
        except Exception as e:
            failed_pages.append({"topic": topic, "error": str(e)})
            logger.warning(f"Failed to load '{topic}': {e}")
    
    logger.info(f"  ✓ Loaded {len(documents)} documents from Wikipedia")
    
    # Create extraction report
    report = {
        "total_topics": len(topics),
        "successful": len(successful_pages),
        "failed": len(failed_pages),
        "successful_pages": successful_pages,
        "failed_pages": failed_pages
    }
    
    return documents, report


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
    os.environ["HF_HOME"] = cfg.vector_store.cache_dir
    
    # Initialize embedding model
    logger.info(f"[1/4] Loading embedding model: {cfg.vector_store.embedding_model_name}")
    embed_model = HuggingFaceEmbedding(
        model_name=cfg.vector_store.embedding_model_name,
        cache_folder=cfg.vector_store.cache_dir,
    )
    
    # Configure global settings
    Settings.embed_model = embed_model
    Settings.chunk_size = cfg.vector_store.chunk_size
    Settings.chunk_overlap = cfg.vector_store.chunk_overlap
    
    # Load documents from multiple sources
    all_documents = []
    
    # Load Wikipedia documents
    logger.info("[2/5] Loading Wikipedia documents...")
    wiki_docs, extraction_report = load_wikipedia_documents(cfg)
    all_documents.extend(wiki_docs)
    
    # Load Wikivoyage documents (if path provided)
    logger.info("[3/5] Loading Wikivoyage documents...")
    if hasattr(cfg.vector_store, 'wikivoyage_xml_path') and cfg.vector_store.wikivoyage_xml_path:
        wikivoyage_docs = load_wikivoyage_documents(
            xml_path=cfg.vector_store.wikivoyage_xml_path,
            chunk_size=cfg.vector_store.get('wikivoyage_chunk_size', 1000),
            step_size=cfg.vector_store.get('wikivoyage_step_size', 500)
        )
        all_documents.extend(wikivoyage_docs)
    else:
        logger.info("  ⚠ Wikivoyage path not configured, skipping...")
    
    if not all_documents:
        raise ValueError("No documents were successfully loaded!")
    
    logger.info(f"  ✓ Total documents loaded: {len(all_documents)}")
    
    # Build index
    logger.info(f"[4/5] Building vector index...")
    index = VectorStoreIndex.from_documents(
        all_documents,
        show_progress=True,
    )
    
    # Persist index
    persist_dir = Path(cfg.vector_store.persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"[5/5] Persisting index to: {persist_dir}")
    index.storage_context.persist(persist_dir=str(persist_dir))
    
    # Save extraction report to hydra output directory
    report_path = persist_dir / "wikipedia_extraction_report.json"
    with open(report_path, 'w') as f:
        json.dump(extraction_report, f, indent=2)
    logger.info(f"  ✓ Extraction report saved to: {report_path}")
    
    logger.info("="*80)
    logger.info("✓ Atlas built successfully!")
    logger.info(f"  Wikipedia: {extraction_report['successful']}/{extraction_report['total_topics']} pages successful")
    logger.info("="*80)
    
    return index

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg):
    """Main entry point for building the Cultural Knowledge Atlas."""
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    build_atlas(cfg)

if __name__ == "__main__":
    main()