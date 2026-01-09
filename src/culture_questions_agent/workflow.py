"""Cultural QA Workflow using LLM-based Query Generation, Wikipedia/DuckDuckGo Search, and NLL-based prediction."""
import logging
import os
import time
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
        
        # [4] Initialize SOTA Multi-Retriever (ColBERT + Dense + Sparse + Reranking)
        use_retrieval = cfg.retrieval.get("use_hybrid_retrieval", True)
        if use_retrieval:
            logger.info(f"[4/5] Initializing SOTA Multi-Retriever (ColBERT + Dense + Sparse)")
            try:
                self.retriever = MultiRetrieverOrchestrator.from_persist_dir(
                    persist_dir=cfg.vector_store.persist_dir,
                    embedding_model_name=cfg.vector_store.embedding_model_name,
                    cache_dir=cfg.model.cache_dir,
                    device=cfg.retrieval.get("device", "cuda"),
                )
                # Note: Reranking is handled internally by MultiRetrieverOrchestrator
                # No need for separate reranker initialization
            except FileNotFoundError as e:
                logger.warning(f"SOTA retriever initialization failed: {e}")
                logger.warning("Continuing without retrieval. Run builder.py to create the index.")
                self.retriever = None
        else:
            logger.info(f"[4/5] Local retrieval disabled")
            self.retriever = None
        
        # [5] Reranker is now integrated into MultiRetrieverOrchestrator
        # Legacy reranker for web search results only (if needed)
        self.use_reranker = cfg.retrieval.get("use_reranker_for_web", False)
        if self.use_reranker:
            logger.info(f"[5/5] Initializing Web Search Reranker: {cfg.model.get('reranker_name', 'BAAI/bge-reranker-v2-m3')}")
            self.reranker = Reranker(
                model_name=cfg.model.get("reranker_name", "BAAI/bge-reranker-v2-m3"),
                cache_dir=cfg.model.cache_dir
            )
        else:
            logger.info(f"[5/5] Web search reranker disabled (orchestrator handles local reranking)")
            self.reranker = None
        
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
    async def search_for_options(self, ev: QueryGenerationEvent) -> SearchEvent:
        """
        Execute multi-source retrieval:
        - Source 1: Hybrid retrieval (BM25 + Vector) from local documents
        - Source 2: Web search (DDGS) for current information
        - Combine and optionally rerank all results
        
        Args:
            ev: QueryGenerationEvent with question queries and options
            
        Returns:
            SearchEvent with all search results
        """
        logger.info(f"🔍 Multi-Source Retrieval Execution")
        
        all_question_snippets = []

        # --- Source 1: SOTA Multi-Retrieval from Local Documents ---
        if self.retriever:
            logger.info(f"  [Source 1] SOTA Multi-Retrieval (ColBERT + Dense + Sparse) from local documents")
            
            # Retrieve from local documents using SOTA multi-retriever
            # retrieve_texts_batch returns list of text strings from top nodes
            local_docs = self.retriever.retrieve_texts_batch(ev.question_queries)
            
            if local_docs:
                logger.info(f"    Retrieved {len(local_docs)} documents from local index")
                all_question_snippets.extend(local_docs)
            else:
                logger.info(f"    No results from local index")
        else:
            logger.info(f"  [Source 1] Hybrid retrieval disabled or unavailable")

        # --- Source 2: Web Search (DDGS) ---
        logger.info(f"  [Source 2] Web Search (DDGS)")
        
        include_options = self.cfg.retrieval.get("include_options_in_query", True)
        
        for q in ev.question_queries:
            logger.info(f"    Searching: '{q}'")
            
            # Build query with options if configured
            query = q
            if self.use_reranker and include_options:
                query = q + " " + " ".join(ev.options.values())
            
            # Get search results as list if using reranker, else as concatenated string
            if self.use_reranker:
                search_results = self.search_engine.search_web(
                    query, 
                    max_results=self.cfg.retrieval.get("max_web_search_results", 5),
                    return_list=True
                )
                if search_results:
                    all_question_snippets.extend(search_results)
            else:
                search_result = self.search_engine.search_web(
                    query, 
                    max_results=self.cfg.retrieval.get("max_web_search_results", 3)
                )
                if search_result:
                    all_question_snippets.append(search_result)

            time.sleep(3.0)  # To avoid rate limiting

        # Deduplicate snippets before reranking
        if all_question_snippets:
            original_count = len(all_question_snippets)
            all_question_snippets = list(set(all_question_snippets))
            
            if original_count > len(all_question_snippets):
                logger.info(f"  Deduplicated: {original_count} → {len(all_question_snippets)} snippets")

        # Rerank if enabled
        if self.use_reranker and all_question_snippets and self.reranker:
            logger.info(f"  [Reranking] Selecting best from {len(all_question_snippets)} total snippets...")
            top_k = self.cfg.retrieval.get("reranker_top_k", 3)
            reranked = self.reranker.rerank(
                query=ev.question,
                documents=all_question_snippets,
                top_k=top_k
            )
            # Extract just the documents from reranked results
            context_snippets = [doc for _, doc, _ in reranked]
        else:
            context_snippets = all_question_snippets
        
        logger.info(f"  Final question context: {len(context_snippets)} snippets")
        
        options_contexts = {}

        # --- Option Verification (Use Wikipedia) ---
        # Encyclopedia is better at "Entity Definitions"
        # Only query options if use_option_context is enabled
        use_option_ctx = self.cfg.retrieval.get("use_option_context", True)
        
        if use_option_ctx:
            logger.info(f"  [Option Verification] Searching for option definitions")
            for opt_key, opt_text in ev.options.items():
                if self.query_generator.is_generic_option(opt_text):
                    continue
                
                # Use the "Entity" query (e.g. "Hanbok definition")
                query = self.query_generator.generate_option_query(opt_text)
                logger.debug(f"    [{opt_key}] '{opt_text}' → Query: '{query}'")
                
                # FORCE Wikipedia for option definitions
                search_result = self.search_engine.search_wikipedia(query)
                options_contexts[opt_key] = search_result

                logger.debug(f"    [{opt_key}] Search result: ''{search_result[:200]}''")
        else:
            logger.info(f"  [Option Verification] Skipping (use_option_context=False)")

        
        return SearchEvent(
            question=ev.question,
            options=ev.options,
            question_context=context_snippets,
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
        logger.info(f"🎯 Predicting answer using NLL with dual-path evidence...")
        
        # Get config flags for context enrichment
        use_question_ctx = self.cfg.retrieval.get("use_question_context", True)
        use_option_ctx = self.cfg.retrieval.get("use_option_context", True)
        
        logger.info(f"  Context enrichment: question={use_question_ctx}, option={use_option_ctx}")
        
        # Build contexts for each option:
        # Combine question context + option-specific context based on config
        
        question_ctx = ev.question_context if use_question_ctx else [""]

        # Predict using NLL
        best_option, losses = self.nll_predictor.predict_best_option(
            question=ev.question,
            options=ev.options,
            option_contexts=ev.option_contexts,
            question_contexts=question_ctx,
        )
        
        logger.info(f"  ✓ Selected: {best_option} - {ev.options[best_option]}")
        
        return StopEvent(result=best_option)

    
