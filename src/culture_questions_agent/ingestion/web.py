import logging
from pathlib import Path
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.readers.base import ReaderConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from culture_questions_agent.ingestion.wikipedia import WikipediaTopicReader
from culture_questions_agent.ingestion.wikivoyage import WikivoyageReader
from culture_questions_agent.ingestion.questions import TrainingDataReader
from culture_questions_agent.query_generator import QueryGenerator

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore

from llama_index.llms.huggingface import HuggingFaceLLM
from jinja2 import Environment, FileSystemLoader
from ddgs.ddgs import DDGS

from llama_index.readers.web import TrafilaturaWebReader

logger = logging.getLogger(__name__)

def obtain_web_docs(cfg):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.llm_name,
        cache_dir=cfg.model.cache_dir
    )    
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.llm_name,
        cache_dir=cfg.model.cache_dir,
        device_map="auto",
    )
    model.eval()

    generator = QueryGenerator(
        model=model,
        tokenizer=tokenizer
    )

    kwargs = {
        "saq_path": cfg.vector_store.get('training_saq_path', 'data/saq_training_data.tsv'),
        "mcq_path": cfg.vector_store.get('training_mcq_path', 'data/mcq_training_data.tsv'),
    }
    reader = TrainingDataReader()

    questions = reader.load_data(**kwargs)

    ddgs = DDGS()

    search_results = []

    for question in tqdm(questions[:5], desc="Performing web searches"):
        country = question.metadata.get("country")
        queries = generator.generate_queries(question.get_content(), num_queries=3)
        
        regions = ["us-en"]
        if country == "China":
            regions.extend(["cn-zh", "hk-zh", "cn-en"])
        if country == "Iran":
            regions.extend(["ir-fa", "ir-en"])
        if country == "United Kingdom":
            regions.extend(["gb-en"])

        for region in regions:
            
            for query in queries:            
                
                try:
                    web_results = ddgs.text(
                        query, 
                        max_results=3, 
                        region=region, 
                        backend=cfg.vector_store.get('ddgs_backend', 'yandex,yahoo')
                    )
                    
                    search_results.extend(web_results)
                except Exception:
                    logger.warning("Failed to perform web search for query '%s' in region '%s'", query, region)

    logger.info(f"Total web search results obtained: {len(search_results)}")
    web_reader = TrafilaturaWebReader()
    
    web_docs = web_reader.load_data(
        urls=[res["href"] for res in search_results],
        include_comments=False,
    )

    logger.info(f"Total web documents loaded: {len(web_docs)}")

    for search_res, doc in zip(search_results, web_docs):
        print("Title:", search_res["title"])
        print(doc.get_content()[:200])
        doc.metadata.update({"source": "web_search", "title": search_res["title"], "url": search_res["href"]})
    
    return web_docs