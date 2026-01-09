"""SOTA Index builder for Cultural QA RAG system.

Implements:
- Section-aware Wikipedia/Wikivoyage parsing
- Semantic chunking with multi-granularity indexing
- Rich metadata extraction (country, culture_domain, etc.)
- Multi-retriever support (ColBERT + Dense + Sparse)
- Answer-centric optimization for cross-encoder reranking
"""
import json
import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple, Optional, cast
from multiprocessing import Pool, cpu_count
from functools import partial

import hydra
import pycountry
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    Document,
)
from llama_index.core.schema import BaseNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.wikipedia import WikipediaReader
from tqdm import tqdm

from culture_questions_agent.section_parser import (
    WikipediaSectionParser,
    WikivoyageSectionParser,
    Section,
)
from culture_questions_agent.metadata_extractor import (
    MetadataExtractor,
    CulturalMetadata,
)
from culture_questions_agent.semantic_chunker import (
    MultiGranularitySemanticChunker,
    RerankerAlignedChunker,
)
from culture_questions_agent.colbert_retriever import ColBERTRetriever
from culture_questions_agent.multi_retriever import MultiRetrieverOrchestrator

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


def get_country_list(limit: Optional[int] = None, filter_list: Optional[List[str]] = None) -> List[str]:
    """
    Get list of standardized country names.
    
    Args:
        limit: Maximum number of countries to return (None for all)
        filter_list: Specific list of country names to include (None for all)
    
    Returns:
        List of country names with overrides applied
    """
    # Use custom filter list if provided
    if filter_list:
        return filter_list
    
    countries = []
    for country in pycountry.countries:
        c_name = getattr(country, "name", None)
        if c_name is None:
            continue
        # Apply overrides for Wikipedia-friendly names
        c_name = OVERRIDES.get(c_name, c_name)
        countries.append(c_name)
    
    # Apply limit if specified
    if limit is not None:
        countries = countries[:limit]
    
    return countries


def load_wikipedia_sections(
    cfg,
    metadata_extractor: MetadataExtractor,
) -> Tuple[List[Document], Dict]:
    """
    Load Wikipedia pages and split into sections with metadata.
    Uses caching to avoid re-downloading pages.
    
    Args:
        cfg: Hydra configuration
        metadata_extractor: Metadata extraction utility
        
    Returns:
        Tuple of (List of Document objects with section-level metadata, extraction report)
    """
    logger.info("="*80)
    logger.info("[Wikipedia] Section-Aware Loading with Caching")
    logger.info("="*80)
    
    # Setup cache directory
    cache_dir = Path(cfg.vector_store.get('wikipedia_cache_dir', 'cache/wikipedia'))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "wikipedia_pages.json"
    
    # Generate topics from templates
    logger.info("Generating topics from templates...")
    topics = []
    
    # Get country filtering options from config
    country_limit = cfg.vector_store.get('country_limit', None)
    country_filter_list = cfg.vector_store.get('country_filter_list', None)
    
    countries = get_country_list(limit=country_limit, filter_list=country_filter_list)
    
    # Log filtering info
    if country_filter_list:
        logger.info(f"  Using custom country filter list: {len(countries)} countries")
    elif country_limit:
        logger.info(f"  Limited to first {country_limit} countries")
    else:
        logger.info(f"  Using all countries: {len(countries)} countries")
    
    # Topic templates for cultural knowledge
    templates = cfg.vector_store.get('topic_templates', [
        "Culture of {}",
        "Public holidays in {}",
        "Cuisine of {}",
        "Education in {}",
        "Music of {}",
    ])
    
    for country in countries:
        for template in templates:
            topics.append(template.format(country))
    
    logger.info(f"  ✓ Generated {len(topics)} topics ({len(templates)} templates × {len(countries)} countries)")
    
    # Try to load from cache
    cached_pages = {}
    if cache_file.exists() and not cfg.vector_store.get('force_wikipedia_refresh', False):
        logger.info(f"Loading cached Wikipedia pages from {cache_file}...")
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_pages = json.load(f)
            logger.info(f"  ✓ Loaded {len(cached_pages)} cached pages")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            cached_pages = {}
    
    # Load Wikipedia pages (from cache or API)
    logger.info("Loading Wikipedia pages...")
    reader = WikipediaReader()
    section_parser = WikipediaSectionParser(min_section_length=100)
    
    documents = []
    successful_pages = []
    failed_pages = []
    section_count = 0
    newly_downloaded = 0
    
    for topic in tqdm(topics, desc="Loading Wikipedia"):
        try:
            # Check cache first
            if topic in cached_pages:
                page_title = cached_pages[topic]['title']
                page_text = cached_pages[topic]['text']
            else:
                # Download from Wikipedia
                page_docs = reader.load_data(pages=[topic], auto_suggest=cfg.vector_store.auto_suggest)
                
                if not page_docs:
                    continue
                
                page_doc = page_docs[0]
                page_title = page_doc.metadata.get("title", topic)
                page_text = page_doc.get_content()
                
                # Cache the downloaded page
                cached_pages[topic] = {
                    'title': page_title,
                    'text': page_text
                }
                newly_downloaded += 1
            
            # Parse into sections
            sections = section_parser.parse_sections(page_title, page_text)
            
            # Create documents for each section
            for section in sections:
                # Clean section text
                clean_text = section_parser.clean_section_text(section.content)
                
                if len(clean_text) < 100:
                    continue
                
                # Extract metadata
                metadata = metadata_extractor.extract_metadata_from_wikipedia(
                    page_title=section.page_title,
                    section_title=section.title if section.title != page_title else None,
                    section_text=clean_text,
                )
                
                # Create document with rich metadata
                doc = Document(
                    text=clean_text,
                    metadata=metadata.to_dict(),
                )
                documents.append(doc)
                section_count += 1
            
            successful_pages.append(topic)
            
        except Exception as e:
            failed_pages.append({"topic": topic, "error": str(e)})
            logger.debug(f"Failed to load '{topic}': {e}")
    
    # Save updated cache
    if newly_downloaded > 0:
        logger.info(f"Saving {newly_downloaded} newly downloaded pages to cache...")
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cached_pages, f, ensure_ascii=False, indent=2)
            logger.info(f"  ✓ Cache saved to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    logger.info("="*80)
    logger.info(f"✓ Wikipedia Loading Complete")
    logger.info(f"  Pages: {len(successful_pages)}/{len(topics)} successful")
    logger.info(f"  Sections: {section_count}")
    logger.info(f"  Documents: {len(documents)}")
    logger.info("="*80)
    
    # Create report
    report = {
        "total_topics": len(topics),
        "successful": len(successful_pages),
        "failed": len(failed_pages),
        "sections_extracted": section_count,
        "documents_created": len(documents),
        "successful_pages": successful_pages,
        "failed_pages": failed_pages,
    }
    
    return documents, report


def _process_wikivoyage_pages_chunk(
    pages_chunk: List[Tuple[str, str]],
    country_filter: Optional[List[str]],
) -> List[Document]:
    """
    Process a chunk of Wikivoyage pages in parallel.
    
    Args:
        pages_chunk: List of (title, text) tuples
        country_filter: Optional list of country names to filter by
        
    Returns:
        List of Document objects
    """
    from culture_questions_agent.section_parser import WikivoyageSectionParser
    from culture_questions_agent.metadata_extractor import MetadataExtractor
    
    section_parser = WikivoyageSectionParser(min_section_length=100)
    metadata_extractor = MetadataExtractor()
    documents = []
    
    # Normalize country filter for matching
    normalized_filter = None
    if country_filter:
        normalized_filter = {country.lower() for country in country_filter}
    
    for title, text in pages_chunk:
        # Apply country filter if specified
        if normalized_filter:
            title_lower = title.lower()
            country_match = any(country in title_lower for country in normalized_filter)
            
            if not country_match:
                continue
        
        # Parse into sections
        sections = section_parser.parse_sections(title, text)
        
        # Create documents for each section
        for section in sections:
            # Clean section text
            clean_text = section_parser.clean_section_text(section.content)
            
            if len(clean_text) < 100:
                continue
            
            # Extract metadata
            metadata = metadata_extractor.extract_metadata_from_wikivoyage(
                page_title=section.page_title,
                section_title=section.title if section.title != title else None,
                section_text=clean_text,
            )
            
            # Double-check country match in metadata if filtering
            if normalized_filter:
                doc_country = metadata.country.lower() if metadata.country else ""
                if doc_country and doc_country not in normalized_filter:
                    continue
            
            # Create document
            doc = Document(
                text=clean_text,
                metadata=metadata.to_dict(),
            )
            documents.append(doc)
    
    return documents


def load_wikivoyage_sections(
    xml_path: str,
    metadata_extractor: MetadataExtractor,
    country_filter: Optional[List[str]] = None,
    num_workers: Optional[int] = None,
) -> List[Document]:
    """
    Load Wikivoyage XML dump and split into sections with metadata.
    Uses multiprocessing for faster parsing.
    
    Args:
        xml_path: Path to Wikivoyage XML dump
        metadata_extractor: Metadata extraction utility
        country_filter: Optional list of country names to filter by
        num_workers: Number of parallel workers (None = CPU count - 1)
        
    Returns:
        List of Document objects with section-level metadata
    """
    logger.info("="*80)
    logger.info("[Wikivoyage] Section-Aware Loading with Parallel Processing")
    logger.info("="*80)
    
    if not os.path.exists(xml_path):
        logger.warning(f"Wikivoyage file not found: {xml_path}")
        return []
    
    # Set number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    logger.info(f"  Using {num_workers} parallel workers")
    
    # Log country filter
    if country_filter:
        logger.info(f"  Filtering by {len(country_filter)} countries: {country_filter}")
    else:
        logger.info("  No country filter (processing all pages)")
    
    ns = {"mw": "http://www.mediawiki.org/xml/export-0.11/"}
    
    # First pass: Extract all pages into memory
    logger.info(f"[1/2] Extracting pages from XML: {xml_path}")
    pages = []
    pages_filtered = 0
    
    # Normalize country filter for early filtering
    normalized_filter = None
    if country_filter:
        normalized_filter = {country.lower() for country in country_filter}
    
    context = ET.iterparse(xml_path, events=("end",))
    
    for event, elem in tqdm(context, desc="Extracting pages"):
        if elem.tag.endswith("page"):
            title_elem = elem.find("mw:title", ns)
            revision = elem.find("mw:revision", ns)
            text_elem = revision.find("mw:text", ns) if revision is not None else None
            
            title = title_elem.text if title_elem is not None else None
            text = text_elem.text if text_elem is not None else ""
            
            # Skip empty pages and redirects
            if not text or not title or text.strip().upper().startswith("#REDIRECT"):
                elem.clear()
                continue
            
            # Apply early country filter on title (before storing in memory)
            if normalized_filter:
                title_lower = title.lower()
                country_match = any(country in title_lower for country in normalized_filter)
                
                if not country_match:
                    pages_filtered += 1
                    elem.clear()
                    continue
            
            pages.append((title, text))
            elem.clear()
    
    logger.info(f"  Extracted {len(pages)} pages")
    if normalized_filter:
        logger.info(f"  Filtered out {pages_filtered} pages by country")
    
    if not pages:
        logger.warning("  No pages to process!")
        return []
    
    # Second pass: Process pages in parallel
    logger.info(f"[2/2] Processing {len(pages)} pages with {num_workers} workers...")
    
    # Split pages into chunks for parallel processing
    chunk_size = max(1, len(pages) // num_workers)
    chunks = [pages[i:i + chunk_size] for i in range(0, len(pages), chunk_size)]
    
    logger.info(f"  Split into {len(chunks)} chunks of ~{chunk_size} pages each")
    
    # Process chunks in parallel
    process_fn = partial(
        _process_wikivoyage_pages_chunk,
        country_filter=country_filter,
    )
    
    all_documents = []
    
    if num_workers > 1:
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_fn, chunks),
                total=len(chunks),
                desc="Processing chunks"
            ))
        
        # Flatten results
        for chunk_docs in results:
            all_documents.extend(chunk_docs)
    else:
        # Single process mode (for debugging)
        for chunk in tqdm(chunks, desc="Processing chunks"):
            chunk_docs = process_fn(chunk)
            all_documents.extend(chunk_docs)
    
    logger.info("="*80)
    logger.info(f"✓ Wikivoyage Loading Complete")
    logger.info(f"  Pages processed: {len(pages)}")
    logger.info(f"  Documents created: {len(all_documents)}")
    logger.info("="*80)
    
    return all_documents


def build_atlas(cfg) -> Tuple[VectorStoreIndex, MultiRetrieverOrchestrator]:
    """
    Build a SOTA VectorStoreIndex with multi-granularity semantic chunking,
    rich metadata, and multi-retriever support (ColBERT + Dense + Sparse).
    
    Pipeline:
    1. Load Wikipedia and Wikivoyage sections
    2. Extract rich metadata (country, culture_domain, etc.)
    3. Apply semantic chunking at multiple granularities
    4. Build vector index
    5. Initialize multi-retriever orchestrator
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        Tuple of (VectorStoreIndex, MultiRetrieverOrchestrator)
    """
    logger.info("="*80)
    logger.info("SOTA CULTURAL KNOWLEDGE ATLAS BUILDER")
    logger.info("Answer-Centric RAG with Late-Interaction Retrieval")
    logger.info("="*80)
    
    # Set cache directory for HuggingFace models
    os.environ["HF_HOME"] = cfg.vector_store.cache_dir
    
    # [1] Initialize embedding model
    logger.info(f"\n[1/7] Loading Embedding Model")
    logger.info(f"  Model: {cfg.vector_store.embedding_model_name}")
    
    embed_model = HuggingFaceEmbedding(
        model_name=cfg.vector_store.embedding_model_name,
        cache_folder=cfg.vector_store.cache_dir,
    )
    
    Settings.embed_model = embed_model
    logger.info(f"  ✓ Embedding model loaded")
    
    # [2] Initialize metadata extractor
    logger.info(f"\n[2/7] Initializing Metadata Extractor")
    metadata_extractor = MetadataExtractor()
    logger.info(f"  ✓ Metadata extractor initialized")
    
    # [3] Load documents with section-aware parsing
    logger.info(f"\n[3/7] Loading Documents (Section-Aware)")
    all_documents = []
    
    # Load Wikipedia sections
    wiki_docs, extraction_report = load_wikipedia_sections(cfg, metadata_extractor)
    all_documents.extend(wiki_docs)
    
    # Load Wikivoyage sections (if enabled)
    use_wikivoyage = cfg.vector_store.get('use_wikivoyage', False)
    
    if use_wikivoyage:
        if hasattr(cfg.vector_store, 'wikivoyage_xml_path') and cfg.vector_store.wikivoyage_xml_path:
            # Get country filter (same as Wikipedia)
            country_limit = cfg.vector_store.get('country_limit', None)
            country_filter_list = cfg.vector_store.get('country_filter_list', None)
            wikivoyage_countries = get_country_list(limit=country_limit, filter_list=country_filter_list)
            
            # Get number of workers for parallel processing
            num_workers = cfg.vector_store.get('wikivoyage_num_workers', None)
            
            wikivoyage_docs = load_wikivoyage_sections(
                cfg.vector_store.wikivoyage_xml_path,
                metadata_extractor,
                country_filter=wikivoyage_countries,
                num_workers=num_workers,
            )
            all_documents.extend(wikivoyage_docs)
        else:
            logger.warning("\n[Wikivoyage] Enabled but no wikivoyage_xml_path configured")
    else:
        logger.info("\n[Wikivoyage] Skipped (use_wikivoyage=False)")
    
    if not all_documents:
        raise ValueError("No documents were successfully loaded!")
    
    logger.info(f"\n✓ Total documents loaded: {len(all_documents)}")
    
    # [4] Apply multi-granularity semantic chunking
    logger.info(f"\n[4/7] Multi-Granularity Semantic Chunking")
    
    chunking_strategy = cfg.vector_store.get('chunking_strategy', 'multi_granularity')
    
    if chunking_strategy == 'multi_granularity':
        logger.info("  Strategy: Multi-Granularity Semantic Chunking")
        
        granularities = cfg.vector_store.get('granularities', ['small', 'medium', 'large'])
        
        chunker = MultiGranularitySemanticChunker(
            embedding_model_name=cfg.vector_store.embedding_model_name,
            tokenizer_name=cfg.vector_store.get('tokenizer_name', 'BAAI/bge-m3'),
            cache_dir=cfg.vector_store.cache_dir,
            granularities=granularities,
        )
        
        nodes = chunker.chunk_documents_multi_granularity(all_documents, granularities)
        
    elif chunking_strategy == 'reranker_aligned':
        logger.info("  Strategy: Reranker-Aligned Chunking")
        
        chunker = RerankerAlignedChunker(
            tokenizer_name=cfg.vector_store.get('tokenizer_name', 'BAAI/bge-m3'),
            target_tokens=cfg.vector_store.get('target_tokens', 900),
            max_tokens=cfg.vector_store.get('max_tokens', 1200),
            cache_dir=cfg.vector_store.cache_dir,
        )
        
        nodes = chunker.chunk_documents(all_documents)
    
    else:
        raise ValueError(f"Unknown chunking strategy: {chunking_strategy}")
    
    logger.info(f"  ✓ Created {len(nodes)} chunks")
    
    # [5] Build vector index with batch embeddings
    logger.info(f"\n[5/7] Building Vector Index (Batch Embeddings)")
    
    # Cast nodes to BaseNode list for type compatibility
    base_nodes = cast(List[BaseNode], nodes)
    
    # Configure batch size for embeddings (if supported by embed_model)
    batch_size = cfg.vector_store.get('embed_batch_size', 128)
    logger.info(f"  Recommended batch size: {batch_size}")
    logger.info(f"  Note: LlamaIndex will batch embeddings automatically during indexing")
    
    index = VectorStoreIndex(
        nodes=base_nodes,
        embed_model=embed_model,
        show_progress=True,
    )
    
    logger.info(f"  ✓ Vector index built ({len(nodes)} nodes)")
    
    # [6] Persist index
    persist_dir = Path(cfg.vector_store.persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n[6/7] Persisting Index")
    logger.info(f"  Directory: {persist_dir}")
    
    index.storage_context.persist(persist_dir=str(persist_dir))
    
    # Save extraction report
    report_path = persist_dir / "wikipedia_extraction_report.json"
    with open(report_path, 'w') as f:
        json.dump(extraction_report, f, indent=2)
    
    logger.info(f"  ✓ Index persisted")
    logger.info(f"  ✓ Extraction report saved: {report_path}")
    
    # [7] Initialize multi-retriever orchestrator
    logger.info(f"\n[7/7] Initializing Multi-Retriever Orchestrator")
    
    use_colbert = cfg.retrieval.get('use_colbert', True)
    use_dense = cfg.retrieval.get('use_dense', True)
    use_sparse = cfg.retrieval.get('use_sparse', True)
    use_reranker = cfg.retrieval.get('use_reranker', True)
    
    # Initialize ColBERT if enabled
    colbert_retriever = None
    if use_colbert:
        colbert_index_path = persist_dir / "colbert_index.pkl"
        
        logger.info("  Building ColBERT retriever (this may take time on H100)...")
        colbert_retriever = ColBERTRetriever(
            model_name=cfg.retrieval.get('colbert_model', 'colbert-ir/colbertv2.0'),
            nodes=cast(List[BaseNode], nodes),
            similarity_top_k=cfg.retrieval.get('colbert_top_k', 50),
            device=cfg.retrieval.get('device', 'cuda'),
            cache_dir=cfg.vector_store.cache_dir,
            index_path=str(colbert_index_path),
        )
    
    # Build orchestrator
    orchestrator = MultiRetrieverOrchestrator(
        index=index,
        nodes=cast(List[BaseNode], nodes),
        colbert_retriever=colbert_retriever,
        dense_top_k=cfg.retrieval.get('hybrid_dense_top_k', 50),
        sparse_top_k=cfg.retrieval.get('hybrid_sparse_top_k', 50),
        colbert_top_k=cfg.retrieval.get('colbert_top_k', 50),
        reranker_model=cfg.model.get('reranker_name', 'BAAI/bge-reranker-v2-m3'),
        reranker_top_k=cfg.retrieval.get('reranker_top_k', 10),
        cache_dir=cfg.vector_store.cache_dir,
        use_colbert=use_colbert,
        use_dense=use_dense,
        use_sparse=use_sparse,
        use_reranker=use_reranker,
    )
    
    # Save orchestrator configuration for loading in workflow
    orchestrator_config = {
        "use_colbert": use_colbert,
        "use_dense": use_dense,
        "use_sparse": use_sparse,
        "use_reranker": use_reranker,
        "dense_top_k": cfg.retrieval.get('hybrid_dense_top_k', 50),
        "sparse_top_k": cfg.retrieval.get('hybrid_sparse_top_k', 50),
        "colbert_top_k": cfg.retrieval.get('colbert_top_k', 50),
        "reranker_top_k": cfg.retrieval.get('reranker_top_k', 10),
        "colbert_model": cfg.retrieval.get('colbert_model', 'colbert-ir/colbertv2.0'),
        "reranker_model": cfg.model.get('reranker_name', 'BAAI/bge-reranker-v2-m3'),
        "colbert_index_path": str(persist_dir / "colbert_index.pkl") if use_colbert else None,
    }
    
    config_path = persist_dir / "orchestrator_config.json"
    with open(config_path, 'w') as f:
        json.dump(orchestrator_config, f, indent=2)
    
    # Save orchestrator stats
    stats_path = persist_dir / "retriever_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(orchestrator.get_retrieval_stats(), f, indent=2)
    
    logger.info(f"  ✓ Orchestrator config saved: {config_path}")
    logger.info(f"  ✓ Retriever stats saved: {stats_path}")
    
    logger.info("\n" + "="*80)
    logger.info("✓ SOTA ATLAS BUILT SUCCESSFULLY!")
    logger.info("="*80)
    logger.info(f"Wikipedia Pages: {extraction_report['successful']}/{extraction_report['total_topics']}")
    logger.info(f"Sections Extracted: {extraction_report.get('sections_extracted', 'N/A')}")
    logger.info(f"Total Chunks: {len(nodes)}")
    logger.info(f"Retrievers: ColBERT={use_colbert}, Dense={use_dense}, Sparse={use_sparse}")
    logger.info(f"Reranker: {use_reranker}")
    logger.info("="*80)
    
    return index, orchestrator

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg):
    """Main entry point for building the SOTA Cultural Knowledge Atlas."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Build atlas
    index, orchestrator = build_atlas(cfg)
    
    # Optional: Run test query to validate system
    if cfg.get('run_test_query', False):
        logger.info("\n" + "="*80)
        logger.info("RUNNING TEST QUERY")
        logger.info("="*80)
        
        test_query = "What are traditional Japanese festivals?"
        logger.info(f"Query: {test_query}")
        
        results = orchestrator.retrieve(test_query)
        
        logger.info(f"\n✓ Retrieved {len(results)} results")
        for i, node_with_score in enumerate(results[:3], 1):
            logger.info(f"\n[Result {i}]")
            logger.info(f"  Score: {node_with_score.score:.4f}")
            logger.info(f"  Metadata: {node_with_score.node.metadata}")
            logger.info(f"  Text: {node_with_score.node.get_content()[:200]}...")

if __name__ == "__main__":
    main()