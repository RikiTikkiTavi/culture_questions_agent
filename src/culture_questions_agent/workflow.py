"""Cultural QA Workflow using LLM-based Query Generation, Wikipedia/DuckDuckGo Search, and NLL-based prediction."""
import logging
import os
from typing import Dict

import mlflow
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

from culture_questions_agent.structures import MCQQuestion
from culture_questions_agent.query_generator import QueryGenerator
from culture_questions_agent.search_tools import SearchEngine
from culture_questions_agent.nll_predictor import NLLPredictor
from culture_questions_agent.reranker import Reranker
from culture_questions_agent.multi_retriever import MultiRetrieverOrchestrator

logger = logging.getLogger(__name__)


class QueryGenerationEvent(Event):
    """Event carrying generated queries from LLM."""
    question: str
    question_queries: list[str]  # Queries for question context
    options: Dict[str, str]


class CombinedRetrievalEvent(Event):
    """Event carrying combined and reranked results."""
    question: str
    options: Dict[str, str]
    question_context: list[str]  # Final combined context


class SearchEvent(Event):
    """Event carrying search results for question and each option."""
    question: str
    options: Dict[str, str]
    question_context: list[str]  # Search result for question
    option_contexts: Dict[str, str]  # option_key -> search result text


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
        
        logger.info("="*80)
        logger.info("Initializing Cultural QA Workflow (LLM Query Generation + NLL)")
        logger.info("="*80)
        
        # Set cache directory
        os.environ["HF_HOME"] = cfg.model.cache_dir
        
        # [1] Initialize NLL Predictor (loads the model)
        logger.info(f"[1/3] Initializing NLL Predictor: {cfg.model.llm_name}")
        self.nll_predictor = NLLPredictor(
            model_name=cfg.model.llm_name,
            cache_dir=cfg.model.cache_dir,
            device="auto"
        )
        
        # [2] Initialize Query Generator (reuses the same model/tokenizer)
        logger.info(f"[2/3] Initializing Query Generator (reusing model)")
        self.query_generator = QueryGenerator(
            model=self.nll_predictor.model,
            tokenizer=self.nll_predictor.tokenizer
        )
        
        # [3] Initialize Search Engine
        logger.info(f"[3/5] Initializing Search Engine (Wikipedia + DDGS)")
        self.search_engine = SearchEngine(
            max_chars=cfg.retrieval.get("max_search_chars", 2500),
            include_title=cfg.retrieval.get("include_title", True),
            ddgs_backend=cfg.retrieval.get("ddgs_backend", "yandex,yahoo,wikipedia,grokipedia")
        )
        
        # [4] Initialize SOTA Multi-Retriever (ColBERT + Dense + Sparse + Web)
        # Reranking is disabled here - done once after combining all documents
        use_retrieval = cfg.retrieval.get("use_hybrid_retrieval", True)
        if use_retrieval:
            logger.info(f"[4/5] Initializing SOTA Multi-Retriever (ColBERT + Dense + Sparse + Web)")
            logger.info(f"  Reranking disabled in orchestrator (will rerank combined results)")
            try:
                self.retriever = MultiRetrieverOrchestrator.from_persist_dir(
                    persist_dir=cfg.vector_store.persist_dir,
                    embedding_model_name=cfg.vector_store.embedding_model_name,
                    cache_dir=cfg.model.cache_dir,
                    device=cfg.retrieval.get("device", "cuda"),
                    use_reranker=False,  # Disable reranking in orchestrator
                    use_web=self.use_web,  # Enable web search in orchestrator
                    search_engine=self.search_engine if self.use_web else None,
                )
            except FileNotFoundError as e:
                logger.warning(f"SOTA retriever initialization failed: {e}")
                logger.warning("Continuing without retrieval. Run builder.py to create the index.")
                self.retriever = None
        else:
            logger.info(f"[4/5] Local retrieval disabled")
            self.retriever = None
        
        # [5] Unified reranker for combined results (local + web)
        # Controlled by single use_reranker config option
        self.use_reranker = cfg.retrieval.get("use_reranker", True)
        self.use_web = cfg.retrieval.get("use_web", False)
        
        if self.use_reranker:
            logger.info(f"[5/5] Initializing Combined Results Reranker: {cfg.model.get('reranker_name', 'BAAI/bge-reranker-v2-m3')}")
            self.reranker = Reranker(
                model_name=cfg.model.get("reranker_name", "BAAI/bge-reranker-v2-m3"),
                cache_dir=cfg.model.cache_dir
            )
        else:
            logger.info(f"[5/5] Reranker disabled for combined results")
            self.reranker = None
        
        logger.info(f"Web search: {'enabled' if self.use_web else 'disabled'}")
        
        logger.info("="*80)
        logger.info("✓ Workflow initialized successfully!")
        logger.info("="*80)
    
    @step
    async def generate_queries(self, ev: StartEvent) -> QueryGenerationEvent:
        """
        Generate search queries for dual-path retrieval.
        Path 1: Question context queries (LLM-generated)
        Path 2: Option verification queries (rule-based)
        
        Args:
            ev: StartEvent containing MCQ question
            
        Returns:
            QueryGenerationEvent with generated queries
        """
        mcq_question = ev.get("mcq_question")
        if not mcq_question:
            raise ValueError("mcq_question required")
        
        question = mcq_question.question
        options = mcq_question.options
        
        logger.info(f"🧠 Dual-Path Query Generation for: '{question}'")
        logger.info(f"  Path 1: Generating context queries for question...")
        
        # Path 1: Generate queries or use direct question based on config
        use_direct_question = self.cfg.retrieval.get("use_direct_question", False)
        
        if use_direct_question:
            logger.info(f"  Using direct question for search (use_direct_question=True)")
            question_queries = [question]
        else:
            question_queries = self.query_generator.generate_queries(
                question, 
                num_queries=self.cfg.retrieval.get("num_queries", 3)
            )
        
        logger.info(f"  ✓ Generated {len(question_queries)} question queries")
        logger.info(f"  Path 2: Planning option verification queries...")
        
        return QueryGenerationEvent(
            question=question,
            question_queries=question_queries,
            options=options
        )
    
    @step
    async def retrieve_documents(self, ev: QueryGenerationEvent) -> CombinedRetrievalEvent:
        """
        Retrieve documents from local index and web using SOTA multi-retriever orchestrator.
        
        Args:
            ev: QueryGenerationEvent with question queries
            
        Returns:
            CombinedRetrievalEvent with retrieved and combined results
        """
        logger.info(f"📚 [Step 1/3] Multi-Source Document Retrieval (Local + Web)")
        
        all_docs = []
        
        if self.retriever:
            logger.info(f"  SOTA Multi-Retrieval (ColBERT + Dense + Sparse + Web) with orchestration")
            
            # Check if we should include options in web query
            include_options = self.cfg.retrieval.get("include_options_in_query", True)
            
            # Retrieve from all sources using orchestrator
            # retrieve_texts_batch returns list of text strings from top nodes
            all_docs = self.retriever.retrieve_texts_batch(
                ev.question_queries,
                include_options_in_web_query=include_options,
                options=ev.options if include_options else None
            )
            
            if all_docs:
                logger.info(f"  ✓ Retrieved {len(all_docs)} documents from all sources")
            else:
                logger.info(f"  No results from retrieval")
        else:
            logger.info(f"  Multi-retrieval disabled or unavailable")
        
        # Rerank if enabled
        top_k = self.cfg.retrieval.get("reranker_top_k", 5)
        
        if self.use_reranker and all_docs and self.reranker:
            logger.info(f"  Reranking: Selecting top {top_k} from {len(all_docs)} total snippets...")
            reranked = self.reranker.rerank(
                query=ev.question,
                documents=all_docs,
                top_k=top_k
            )
            # Extract just the documents from reranked results
            context_snippets = [doc for _, doc, _ in reranked]
            logger.info(f"  ✓ Selected top {len(context_snippets)} snippets after reranking")
        else:
            # Even without reranking, limit to top_k to avoid context overflow
            context_snippets = all_docs[:top_k]
            logger.info(f"  ✓ Limited to top {len(context_snippets)} snippets (no reranking)")
        
        return CombinedRetrievalEvent(
            question=ev.question,
            options=ev.options,
            question_context=context_snippets
        )
    
    @step
    async def verify_options(self, ev: CombinedRetrievalEvent) -> SearchEvent:
        """
        Search for option definitions using Wikipedia.
        
        Args:
            ev: CombinedRetrievalEvent with question context
            
        Returns:
            SearchEvent with complete search results
        """
        logger.info(f"✅ [Step 2/3] Option Verification")
        
        options_contexts = {}
        use_option_ctx = self.cfg.retrieval.get("use_option_context", True)
        
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
            
            logger.info(f"  ✓ Retrieved definitions for {len(options_contexts)} options")
        else:
            logger.info(f"  Skipping option verification (use_option_context=False)")
        
        return SearchEvent(
            question=ev.question,
            options=ev.options,
            question_context=ev.question_context,
            option_contexts=options_contexts
        )
    
    @step
    async def predict_with_nll(self, ev: SearchEvent) -> StopEvent:
        """
        Predict best option using NLL loss with dual-path contexts.
        
        Args:
            ev: SearchEvent with question context and option contexts
            
        Returns:
            StopEvent with the final answer
        """
        logger.info(f"🎯 [Step 3/3] Predicting answer using NLL with dual-path evidence...")
        
        # Get config flags for context enrichment
        use_question_ctx = self.cfg.retrieval.get("use_question_context", True)
        use_option_ctx = self.cfg.retrieval.get("use_option_context", True)
        
        logger.info(f"  Context enrichment: question={use_question_ctx}, option={use_option_ctx}")
        
        # Build contexts for each option:
        # Combine question context + option-specific context based on config
        
        question_ctx = ev.question_context if use_question_ctx else [""]

        # Predict using NLL
        best_option, scores = self.nll_predictor.predict_best_option(
            question=ev.question,
            options=ev.options,
            option_contexts=ev.option_contexts,
            question_contexts=question_ctx,
        )
        
        logger.info(f"  ✓ Selected: {best_option} - {ev.options[best_option]}")
        
        return StopEvent(result=best_option)

    
