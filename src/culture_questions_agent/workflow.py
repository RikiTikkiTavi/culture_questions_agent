"""Cultural QA Workflow using LLM-based Query Generation, Wikipedia/DuckDuckGo Search, and NLL-based prediction."""

import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Union

import mlflow
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.schema import TextNode, NodeWithScore

from culture_questions_agent.structures import MCQQuestion, SAQQuestion
from culture_questions_agent.query_generator import QueryGenerator
from culture_questions_agent.search_tools import SearchEngine
from culture_questions_agent.predictor.generative_predictor import GenerativePredictor
from culture_questions_agent.multi_retriever import MultiRetrieverOrchestrator
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from lancedb.rerankers import ColbertReranker

from mlflow.entities import Document

logger = logging.getLogger(__name__)


class QueryGenerationEvent(Event):
    """Event carrying generated queries from LLM."""

    question: str
    question_queries: list[str]  # Queries for question context
    options: Optional[Dict[str, str]]  # None for SAQ questions
    is_mcq: bool  # True for MCQ, False for SAQ


class RetrievalEvent(Event):
    """Event carrying raw retrieved documents before reranking."""

    question: str
    options: Optional[Dict[str, str]]  # None for SAQ questions
    raw_documents: list[NodeWithScore]  # List of NodeWithScore from retrieval
    is_mcq: bool  # True for MCQ, False for SAQ


class CombinedRetrievalEvent(Event):
    """Event carrying combined and reranked results."""

    question: str
    options: Optional[Dict[str, str]]  # None for SAQ questions
    question_context: list[str]  # Final combined context
    is_mcq: bool  # True for MCQ, False for SAQ


class SearchEvent(Event):
    """Event carrying search results for question and each option."""

    question: str
    options: Optional[Dict[str, str]]  # None for SAQ questions
    question_context: list[str]  # Search result for question
    option_contexts: Optional[Dict[str, str]]  # option_key -> search result text, None for SAQ
    is_mcq: bool  # True for MCQ, False for SAQ


class CulturalQAWorkflow(Workflow):
    """
    Event-driven workflow for Cultural QA using:
    - LLM-based query generation
    - Wikipedia/DDGS web search for context retrieval
    - NLL-based prediction using HuggingFace transformers
    """

    def __init__(self, cfg, *args, **kwargs):
        """
        Initialize the workflow.

        Args:
            cfg: Hydra configuration object
        """
        super().__init__(*args, **kwargs)
        self.cfg = cfg

        logger.info("=" * 80)
        logger.info(
            "Initializing Cultural QA Workflow (LLM Query Generation + Configurable Prediction)"
        )
        logger.info("=" * 80)

        # Set cache directories for all HuggingFace components
        os.environ["HF_HOME"] = cfg.model.cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = cfg.model.cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cfg.model.cache_dir
        os.environ["HF_DATASETS_CACHE"] = cfg.model.cache_dir
        # For sentence-transformers (used by FlagEmbeddingReranker)
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = cfg.model.cache_dir

        # Get predictor type from config
        predictor_type = cfg.model.get("predictor_type", "discriminative")
        logger.info(f"Predictor type: {predictor_type}")
        
        # Store predictor type for validation
        self.predictor_type = predictor_type

        # [1] Initialize Predictor using factory
        logger.info(
            f"[1/5] Initializing {predictor_type.capitalize()} Predictor: {cfg.model.llm_name}"
        )
        self.predictor = GenerativePredictor(
            model_name=cfg.model.llm_name,
            cache_dir=cfg.model.cache_dir,
            device="auto",
            max_new_tokens=cfg.model.get("max_new_tokens", 10),
            temperature=cfg.model.get("temperature", 0.0),
        )

        # [2] Initialize Query Generator (reuses the same model/tokenizer for discriminative)
        logger.info(f"[2/5] Initializing Query Generator")
        self.query_generator = QueryGenerator(
            model=self.predictor.model, tokenizer=self.predictor.tokenizer
        )

        retrievers = []
        retriever_names = []
        
        if cfg.retrieval.get("use_wiki_retrieval", True):
            logger.info(f"Initializing Wiki-Retriever")
            wiki_store = LanceDBVectorStore(uri=cfg.vector_store.get("lancedb_path", "storage/lancedb"), table_name="wiki_like")
            if cfg.retrieval.get("use_colbert", True):
                reranker = ColbertReranker()
                wiki_store._add_reranker(reranker)
            wiki_index = VectorStoreIndex.from_vector_store(
                vector_store=wiki_store,
                embed_model=HuggingFaceEmbedding(
                    model_name=cfg.vector_store.embedding_model_name,
                    cache_folder=cfg.model.cache_dir,
                ),
            )
            wiki_retriever = wiki_index.as_retriever(
                similarity_top_k=cfg.retrieval.get("wiki_top_k", 10),
                vector_store_query_mode="hybrid",
                alpha=0.5
            )
            retrievers.append(wiki_retriever)
            retriever_names.append("wiki")
        
        if cfg.retrieval.get("use_train_data_retrieval", True):
            logger.info(f"Initializing train-data retriever")
            q_store = LanceDBVectorStore(uri=cfg.vector_store.get("lancedb_path", "storage/lancedb"), table_name="question_like")
            if cfg.retrieval.get("use_colbert", True):
                reranker = ColbertReranker()
                q_store._add_reranker(reranker)
            q_index = VectorStoreIndex.from_vector_store(
                vector_store=q_store,
                embed_model=HuggingFaceEmbedding(
                    model_name=cfg.vector_store.embedding_model_name,
                    cache_folder=cfg.model.cache_dir,
                ),
            )
            q_retriever = q_index.as_retriever(
                similarity_top_k=cfg.retrieval.get("train_data_top_k", 4),
            )
            retrievers.append(q_retriever)
            retriever_names.append("train_data")
            
        if cfg.retrieval.get("use_web_retrieval", False):
            # [3] Initialize Search Engine
            logger.info(f"Initializing Search Engine")
            search_engine_retriever = SearchEngine(
                max_chars=cfg.retrieval.get("max_search_chars", 2500),
                include_title=cfg.retrieval.get("include_title", True),
                ddgs_backend=cfg.retrieval.get(
                    "ddgs_backend", "yandex,yahoo,wikipedia,grokipedia"
                ),
                similarity_top_k=cfg.retrieval.get("web_top_k", 5),
            )
            retrievers.append(search_engine_retriever)
            retriever_names.append("web")
        
        self.retriever = MultiRetrieverOrchestrator(retrievers=retrievers, names=retriever_names)
        
        # [5] Unified reranker for combined results (local + web)
        # Controlled by single use_reranker config option
        self.use_reranker = cfg.retrieval.get("use_reranker", True)

        if self.use_reranker:
            logger.info(
                f"[5/5] Initializing FlagEmbeddingReranker: {cfg.model.get('reranker_name', 'BAAI/bge-reranker-v2-m3')}"
            )
            # Note: cache_dir is set globally via HF_HOME environment variable above
            self.reranker = FlagEmbeddingReranker(
                model=cfg.model.get("reranker_name", "BAAI/bge-reranker-v2-m3"),
                top_n=10,  # Default, will be overridden per rerank call
            )
        
        logger.info("=" * 80)
        logger.info("✓ Workflow initialized successfully!")
        logger.info("=" * 80)

    @step
    async def generate_queries(self, ev: StartEvent) -> QueryGenerationEvent:
        """
        Generate search queries for dual-path retrieval.
        Path 1: Question context queries (LLM-generated)
        Path 2: Option verification queries (rule-based, MCQ only)

        Args:
            ev: StartEvent containing question field with Union[MCQQuestion, SAQQuestion]

        Returns:
            QueryGenerationEvent with generated queries
        """
        # Get the question and determine its type
        question_obj: Union[MCQQuestion, SAQQuestion] = ev.get("question")
        if not question_obj:
            raise ValueError("question field is required in StartEvent")
        
        # Check question type
        if isinstance(question_obj, MCQQuestion):
            is_mcq = True
            options = question_obj.options
        elif isinstance(question_obj, SAQQuestion):
            is_mcq = False
            options = None
            # Validate that SAQ uses generative predictor
            if self.predictor_type != "generative":
                raise ValueError(
                    f"SAQ questions require generative predictor, but {self.predictor_type} is configured"
                )
        else:
            raise TypeError(f"question must be MCQQuestion or SAQQuestion, got {type(question_obj)}")

        question = question_obj.question
        
        question_type = "MCQ" if is_mcq else "SAQ"
        logger.info(f"🧠 Query Generation for {question_type}: '{question}'")
        logger.info(f"  Path 1: Generating context queries for question...")

        # Path 1: Generate queries or use direct question based on config
        use_direct_question = self.cfg.retrieval.get("use_direct_question", False)

        if use_direct_question:
            logger.info(
                f"  Using direct question for search (use_direct_question=True)"
            )
            question_queries = [question]
        else:
            question_queries = await asyncio.to_thread(
                self.query_generator.generate_queries,
                question=question,
                num_queries=self.cfg.retrieval.get("num_queries", 3)
            )
        
        if self.cfg.retrieval.get("add_question_to_queries", True):
            question_queries.append(question)

        logger.info(f"  ✓ Generated {len(question_queries)} question queries")
        
        if is_mcq:
            logger.info(f"  Path 2: Planning option verification queries...")
        else:
            logger.info(f"  Path 2: Skipped (SAQ has no options)")

        return QueryGenerationEvent(
            question=question, 
            question_queries=question_queries, 
            options=options,
            is_mcq=is_mcq
        )

    @step
    async def retrieve_documents(
        self, ev: QueryGenerationEvent
    ) -> RetrievalEvent:
        """
        Retrieve documents from local index and web using SOTA multi-retriever orchestrator.

        Args:
            ev: QueryGenerationEvent with question queries

        Returns:
            RetrievalEvent with raw retrieved documents
        """
        logger.info(f"📚 [Step 1/4] Multi-Source Document Retrieval (Local + Web)")

        all_docs = []

        if self.retriever:

            # Retrieve from all sources using orchestrator
            all_docs = await asyncio.to_thread(self.retriever.retrieve_batch, ev.question_queries)

            if all_docs:
                logger.info(f"  ✓ Retrieved {len(all_docs)} documents from all sources")
            else:
                logger.info(f"  No results from retrieval")
        else:
            logger.info(f"  Multi-retrieval disabled or unavailable")

        return RetrievalEvent(
            question=ev.question,
            options=ev.options,
            raw_documents=all_docs,
            is_mcq=ev.is_mcq,
        )

    @step
    async def rerank_documents(
        self, ev: RetrievalEvent
    ) -> SearchEvent:
        """
        Rerank retrieved documents using cross-encoder reranker.

        Args:
            ev: RetrievalEvent with raw retrieved documents

        Returns:
            CombinedRetrievalEvent with reranked context snippets
        """
        logger.info(f"🔄 [Step 2/4] Document Reranking")

        all_docs = ev.raw_documents
        context_snippets = []

        if self.use_reranker and all_docs and self.reranker:
            # Rerank in groups based on config
            for group_cfg in self.cfg.retrieval.get("reranking_groups", []):
                sources = group_cfg.get("sources", [])
                top_k = group_cfg.get("top_k", 5)

                group_docs = []
                for node_with_score in all_docs:
                    if set(sources).intersection(node_with_score.node.metadata["source"]):
                        group_docs.append(node_with_score.node.get_content())

                if group_docs:
                    logger.info(
                        f"  Reranking group for sources {sources} with {len(group_docs)} docs..."
                    )
                    with mlflow.start_span(
                        name="reranking_group_" + "_".join(sources),
                        span_type="RERANKER",
                    ) as span:
                        span.set_inputs(
                            [
                                Document(group_docs[i], metadata={"source": sources})
                                for i in range(len(group_docs))
                            ]
                        )
                        # Convert documents to nodes for reranking
                        nodes = [
                            NodeWithScore(node=TextNode(text=doc), score=0.0)
                            for doc in group_docs
                        ]
                        # Rerank all nodes and slice to top_k (thread-safe - no state mutation)
                        reranked_nodes = await asyncio.to_thread(self.reranker.postprocess_nodes, nodes, query_str=ev.question)
                        
                        # Extract documents and scores
                        selected_docs = [node.node.get_content() for node in reranked_nodes]
                        scores = [node.score for node in reranked_nodes]
                        logger.info(
                            f"  ✓ Selected top {len(selected_docs)} snippets after reranking for group"
                        )
                        context_snippets.extend(selected_docs)
                        span.set_outputs(
                            [
                                Document(
                                    selected_docs[i], metadata={"score": scores[i]}
                                )
                                for i in range(len(selected_docs))
                            ]
                        )
                else:
                    logger.info(
                        f"  No documents found for sources {sources}, skipping reranking for this group"
                    )

        else:
            # Even without reranking, limit to top_k to avoid context overflow
            context_snippets = [
                node_with_score.node.get_content() for node_with_score in all_docs[:10]
            ]
            logger.info(
                f"  ✓ Limited to top {len(context_snippets)} snippets (no reranking)"
            )

        return SearchEvent(
            question=ev.question,
            options=ev.options,
            question_context=context_snippets,
            is_mcq=ev.is_mcq,
            option_contexts=None,
        )

    @step
    async def predict_answer(self, ev: SearchEvent) -> StopEvent:
        """
        Predict best option (MCQ) or generate answer (SAQ) using the configured prediction strategy.

        Args:
            ev: SearchEvent with question context and option contexts

        Returns:
            StopEvent with the final answer
        """
        question_type = "MCQ" if ev.is_mcq else "SAQ"
        logger.info(
            f"🎯 [Step 4/4] Predicting {question_type} answer using {self.cfg.model.get('predictor_type', 'discriminative')} strategy..."
        )

        # Get config flags for context enrichment
        use_question_ctx = self.cfg.retrieval.get("use_question_context", True)
        use_option_ctx = self.cfg.retrieval.get("use_option_context", True) and ev.is_mcq

        question_ctx = ev.question_context if use_question_ctx else [""]
        option_ctx = ev.option_contexts if (use_option_ctx and ev.option_contexts) else {}

        if ev.is_mcq:
            # MCQ: Predict using configured predictor
            best_option = await asyncio.to_thread(self.predictor.predict_best_option,
                question=ev.question,
                options=ev.options,
                option_contexts=option_ctx,
                question_contexts=question_ctx,
            )
            logger.info(f"Predicted MCQ Best Option: {best_option} for question: '{ev.question}'")
            return StopEvent(result=best_option)
        else:
            # SAQ: Generate answer using generative predictor
            answer = await asyncio.to_thread(self.predictor.predict_short_answer,
                question=ev.question,
                question_contexts=question_ctx,
            )
            logger.info(f"Generated SAQ Answer: {answer} for question: '{ev.question}'")
            return StopEvent(result=answer)
