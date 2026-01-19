import logging
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.readers.base import ReaderConfig

from culture_questions_agent.ingestion.wikipedia import WikipediaTopicReader
from culture_questions_agent.ingestion.wikivoyage import WikivoyageReader
from culture_questions_agent.ingestion.questions import TrainingDataReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore

import hydra

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg):
    logger.info("Starting ingestion pipeline...")

    sentence_splitter = SentenceSplitter(
        chunk_size=cfg.vector_store.get("chunk_size", 512),
        chunk_overlap=cfg.vector_store.get("chunk_overlap", 50),
    )

    embedding_transform = HuggingFaceEmbedding(
        model_name=cfg.vector_store.embedding_model_name,
        cache_folder=cfg.vector_store.cache_dir,
    )

    # Wikipedia and Wikivoyage ingestion
    if not cfg.ingestion.get("skip_wiki", False):
        logger.info("Skipping Wikipedia and Wikivoyage ingestion as per config.")
        
        countries = cfg.vector_store.get("country_filter_list", [])
        topics = cfg.vector_store.get("topic_templates", [])
        additional_pages = cfg.vector_store.get("additional_wikipedia_pages", [])

        wiki_doc_store = SimpleDocumentStore()
        wiki_store = LanceDBVectorStore(uri=cfg.vector_store.get("lancedb_path", "storage/lancedb"), table_name="wiki_like")


        wiki_reader_cfg = ReaderConfig(
            reader=WikipediaTopicReader(auto_suggest=cfg.vector_store.auto_suggest),
            reader_kwargs={
                "templates": topics,
                "country_list": countries,
                "additional_pages": additional_pages,
            },
        )

        wikivoyage_reader_cfg = ReaderConfig(
            reader=WikivoyageReader(),
            reader_kwargs={
                "xml_path": cfg.vector_store.get(
                    "wikivoyage_xml_path", "data/wikivoyage.xml"
                ),
                "country_filter": countries,
            },
        )

        pipeline_wikipedia = IngestionPipeline(
            readers=[
                wiki_reader_cfg,
                wikivoyage_reader_cfg,
            ],
            transformations=[sentence_splitter, embedding_transform],
            vector_store=wiki_store,
            docstore=wiki_doc_store,
        )
        logger.info("Starting Wikipedia and Wikivoyage ingestion...")
        pipeline_wikipedia.run(show_progress=True)

    # Training data ingestion
    if not cfg.ingestion.get("skip_training_data", False):
        questions_like_store = LanceDBVectorStore(uri=cfg.vector_store.get("lancedb_path", "storage/lancedb"), table_name="question_like")

        pipeline_training_data = IngestionPipeline(
            readers=[
                ReaderConfig(
                    reader=TrainingDataReader(),
                    reader_kwargs={
                        "saq_path": cfg.vector_store.get(
                            "training_saq_path", "data/saq_training_data.tsv"
                        ),
                        "mcq_path": cfg.vector_store.get(
                            "training_mcq_path", "data/mcq_training_data.tsv"
                        ),
                    },
                )
            ],
            transformations=[embedding_transform],
            vector_store=questions_like_store,
            docstore=SimpleDocumentStore(),
        )
        logger.info("Starting training data ingestion...")
        pipeline_training_data.run(show_progress=True)

    # Web ingestion
    if not cfg.ingestion.get("skip_web", False):
        web_like_store = LanceDBVectorStore(uri=cfg.vector_store.get("lancedb_path", "storage/lancedb"), table_name="web_like")
        web_reader = hydra.utils.instantiate(cfg.ingestion.web_reader)
        docs = web_reader.lazy_load_data()
        pipeline_web = IngestionPipeline(
            documents=docs,
            transformations=[sentence_splitter, embedding_transform],
            vector_store=web_like_store,
            docstore=SimpleDocumentStore(),
        )
        logger.info("Starting web ingestion...")
        pipeline_web.run(show_progress=True, num_workers=6)


if __name__ == "__main__":
    main()
