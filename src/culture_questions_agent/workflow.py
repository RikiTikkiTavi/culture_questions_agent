"""Cultural QA Workflow using LLM-based Query Generation, Wikipedia/DuckDuckGo Search, and NLL-based prediction."""

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
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.schema import TextNode, NodeWithScore

from culture_questions_agent.structures import MCQQuestion, SAQQuestion
from culture_questions_agent.query_generator import QueryGenerator
from culture_questions_agent.search_tools import SearchEngine
from culture_questions_agent.predictor_factory import PredictorFactory
from culture_questions_agent.multi_retriever import MultiRetrieverOrchestrator
from culture_questions_agent.colbert_retriever import ColBERTRetriever

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
        self.predictor = PredictorFactory.create_predictor(
            predictor_type=predictor_type,
            model_name=cfg.model.llm_name,
            cache_dir=cfg.model.cache_dir,
            device="auto",
            max_new_tokens=cfg.model.get("max_new_tokens", 10),
            temperature=cfg.model.get("temperature", 0.1),
        )

        # [2] Initialize Query Generator (reuses the same model/tokenizer for discriminative)
        logger.info(f"[2/5] Initializing Query Generator")
        self.query_generator = QueryGenerator(
            model=self.predictor.model, tokenizer=self.predictor.tokenizer
        )

        # [3] Initialize Search Engine
        logger.info(f"[3/5] Initializing Search Engine (Wikipedia + DDGS)")
        self.search_engine = SearchEngine(
            max_chars=cfg.retrieval.get("max_search_chars", 2500),
            include_title=cfg.retrieval.get("include_title", True),
            ddgs_backend=cfg.retrieval.get(
                "ddgs_backend", "yandex,yahoo,wikipedia,grokipedia"
            ),
            similarity_top_k=cfg.retrieval.get("web_top_k", 5),
        )

        # [4] Initialize SOTA Multi-Retriever (ColBERT + Dense + Sparse + Training + Web)
        use_retrieval = cfg.retrieval.get("use_hybrid_retrieval", True)
        self.use_web = cfg.retrieval.get("use_web", False)

        if use_retrieval:
            logger.info(f"[4/5] Initializing SOTA Multi-Retriever")
            
            # Load main index
            logger.info(f"  Loading vector index from {cfg.vector_store.persist_dir}...")
            embed_model = HuggingFaceEmbedding(
                model_name=cfg.vector_store.embedding_model_name,
                cache_folder=cfg.model.cache_dir,
            )
            Settings.embed_model = embed_model
            
            storage_context = StorageContext.from_defaults(persist_dir=cfg.vector_store.persist_dir)
            index = load_index_from_storage(storage_context)
            nodes = list(index.docstore.docs.values())
            logger.info(f"  ✓ Loaded {len(nodes)} nodes")
            
            # Build list of retrievers based on config
            retrievers = []
            retriever_names = []

            # Dense retriever
            if cfg.retrieval.get("use_dense", True):
                from llama_index.core.retrievers import VectorIndexRetriever
                retrievers.append(VectorIndexRetriever(
                    index=index,
                    similarity_top_k=cfg.retrieval.get("hybrid_dense_top_k", 50),
                ))
                logger.info(f"  ✓ Dense retriever (top-{cfg.retrieval.get('hybrid_dense_top_k', 50)})")
                retriever_names.append("dense")

            # Sparse retriever
            if cfg.retrieval.get("use_sparse", True):
                from llama_index.retrievers.bm25 import BM25Retriever
                retrievers.append(BM25Retriever.from_defaults(
                    nodes=nodes,
                    similarity_top_k=cfg.retrieval.get("hybrid_sparse_top_k", 50),
                ))
                logger.info(f"  ✓ Sparse retriever (top-{cfg.retrieval.get('hybrid_sparse_top_k', 50)})")
                retriever_names.append("sparse")

            # ColBERT retriever
            if cfg.retrieval.get("use_colbert", True):
                # Use full path from config (e.g., "storage/colbert_index/colbert_main")
                # But only pass the index_name part ("colbert_main") to ColBERTRetriever
                colbert_index_path = cfg.vector_store.get('colbert_index_path', 'storage/colbert_index/colbert_main')
                # Extract just the index name (last component)
                index_name = Path(colbert_index_path).name
                retrievers.append(ColBERTRetriever(
                    model_name=cfg.retrieval.get("colbert_model", "colbert-ir/colbertv2.0"),
                    nodes=nodes,
                    similarity_top_k=cfg.retrieval.get("colbert_top_k", 50),
                    device=cfg.retrieval.get("device", "cuda"),
                    cache_dir=cfg.model.cache_dir,
                    index_path=index_name,
                ))
                retriever_names.append("colbert_main")
                logger.info(f"  ✓ ColBERT retriever (top-{cfg.retrieval.get('colbert_top_k', 50)})")
            
            # Training retrievers
            if cfg.retrieval.get("use_training_retrieval", False):
                training_persist_dir = cfg.vector_store.get("training_persist_dir")
                logger.info(f"  Loading training index from {training_persist_dir}...")
                
                training_storage_context = StorageContext.from_defaults(persist_dir=training_persist_dir)
                training_index = load_index_from_storage(training_storage_context)
                training_nodes = list(training_index.docstore.docs.values())
                logger.info(f"  ✓ Loaded training index with {len(training_nodes)} nodes")
                
                # Training dense retriever
                from llama_index.core.retrievers import VectorIndexRetriever
                retrievers.append(VectorIndexRetriever(
                    index=training_index,
                    similarity_top_k=cfg.retrieval.get("training_top_k", 10),
                ))
                logger.info(f"  ✓ Training Dense retriever (top-{cfg.retrieval.get('training_top_k', 10)})")
                retriever_names.append("dense_training")

                # Training ColBERT retriever
                if cfg.retrieval.get("use_training_colbert", True):
                    # Use full path from config, extract index name
                    training_colbert_index_path = cfg.vector_store.get('training_colbert_index_path', 'storage/colbert_index/colbert_training')
                    index_name = Path(training_colbert_index_path).name
                    retrievers.append(ColBERTRetriever(
                        model_name=cfg.retrieval.get("colbert_model", "colbert-ir/colbertv2.0"),
                        nodes=training_nodes,
                        similarity_top_k=cfg.retrieval.get("training_colbert_top_k", 10),
                        device=cfg.retrieval.get("device", "cuda"),
                        cache_dir=cfg.model.cache_dir,
                        index_path="storage/colbert_index_training",  # Use separate index path for training
                    ))
                    logger.info(f"  ✓ Training ColBERT retriever (top-{cfg.retrieval.get('training_colbert_top_k', 10)})")
                    retriever_names.append("colbert_training")
            
            # Web search retriever
            if self.use_web:
                retrievers.append(self.search_engine)
                logger.info(f"  ✓ Web search retriever (top-{cfg.retrieval.get('web_top_k', 5)})")
                retriever_names.append("web")
            
            # Create orchestrator
            self.retriever = MultiRetrieverOrchestrator(retrievers=retrievers)
        else:
            logger.info(f"[4/5] Local retrieval disabled")
            self.retriever = None

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
        else:
            logger.info(f"[5/5] Reranker disabled for combined results")
            self.reranker = None

        logger.info(f"Web search: {'enabled' if self.use_web else 'disabled'}")

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
            question_queries = self.query_generator.generate_queries(
                question, num_queries=self.cfg.retrieval.get("num_queries", 3)
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
            logger.info(
                f"  SOTA Multi-Retrieval (ColBERT + Dense + Sparse + Web) with orchestration"
            )

            # Retrieve from all sources using orchestrator
            all_docs = self.retriever.retrieve_batch(ev.question_queries)

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
    ) -> CombinedRetrievalEvent:
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

                # Filter documents for this group based on source tags
                group_docs = [
                    node_with_score.node.get_content()
                    for node_with_score in all_docs
                    if any(
                        source == node_with_score.metadata["source"]
                        for source in sources
                    )
                ]

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
                        reranked_nodes = self.reranker.postprocess_nodes(nodes, query_str=ev.question)
                        
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

        return CombinedRetrievalEvent(
            question=ev.question,
            options=ev.options,
            question_context=context_snippets,
            is_mcq=ev.is_mcq,
        )

    @step
    async def verify_options(self, ev: CombinedRetrievalEvent) -> SearchEvent:
        """
        Search for option definitions using Wikipedia (MCQ only).

        Args:
            ev: CombinedRetrievalEvent with question context

        Returns:
            SearchEvent with complete search results
        """
        logger.info(f"✅ [Step 3/4] Option Verification")

        options_contexts = {}
        use_option_ctx = self.cfg.retrieval.get("use_option_context", True)
        
        # Skip option verification for SAQ questions
        if not ev.is_mcq:
            logger.info(f"  Skipping option verification for SAQ question")
            return SearchEvent(
                question=ev.question,
                options=ev.options,
                question_context=ev.question_context,
                option_contexts=None,
                is_mcq=ev.is_mcq,
            )

        if use_option_ctx:
            logger.info(f"  Searching for option definitions on Wikipedia")
            for opt_key, opt_text in ev.options.items():
                if self.query_generator.is_generic_option(opt_text):
                    continue

                # Use the "Entity" query (e.g. "Hanbok definition")
                query = self.query_generator.generate_option_query(opt_text)
                logger.debug(f"    [{opt_key}] '{opt_text}' → Query: '{query}'")

                # FORCE Wikipedia for option definitions
                search_result = self.search_engine.search_wikipedia(query)
                options_contexts[opt_key] = search_result

                logger.debug(f"    [{opt_key}] Result: '{search_result[:200]}'")

            logger.info(
                f"  ✓ Retrieved definitions for {len(options_contexts)} options"
            )
        else:
            logger.info(f"  Skipping option verification (use_option_context=False)")

        return SearchEvent(
            question=ev.question,
            options=ev.options,
            question_context=ev.question_context,
            option_contexts=options_contexts,
            is_mcq=ev.is_mcq,
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
            best_option = self.predictor.predict_best_option(
                question=ev.question,
                options=ev.options,
                option_contexts=option_ctx,
                question_contexts=question_ctx,
            )
            return StopEvent(result=best_option)
        else:
            # SAQ: Generate answer using generative predictor
            answer = self.predictor.predict_short_answer(
                question=ev.question,
                question_contexts=question_ctx,
            )
            return StopEvent(result=answer)
