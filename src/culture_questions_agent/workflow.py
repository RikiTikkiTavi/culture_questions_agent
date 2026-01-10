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


class LocalRetrievalEvent(Event):
    """Event carrying local retrieval results."""
    question: str
    question_queries: list[str]
    options: Dict[str, str]
    local_documents: list[str]  # Documents from local index


class WebSearchEvent(Event):
    """Event carrying web search results."""
    question: str
    question_queries: list[str]
    options: Dict[str, str]
    local_documents: list[str]
    web_documents: list[str]  # Documents from web search


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
        
        # [4] Initialize SOTA Multi-Retriever (ColBERT + Dense + Sparse)
        # Reranking is disabled here - done once after combining all documents
        use_retrieval = cfg.retrieval.get("use_hybrid_retrieval", True)
        if use_retrieval:
            logger.info(f"[4/5] Initializing SOTA Multi-Retriever (ColBERT + Dense + Sparse)")
            logger.info(f"  Reranking disabled in orchestrator (will rerank combined results)")
            try:
                self.retriever = MultiRetrieverOrchestrator.from_persist_dir(
                    persist_dir=cfg.vector_store.persist_dir,
                    embedding_model_name=cfg.vector_store.embedding_model_name,
                    cache_dir=cfg.model.cache_dir,
                    device=cfg.retrieval.get("device", "cuda"),
                    use_reranker=False,  # Disable reranking in orchestrator
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
        if self.use_reranker:
            logger.info(f"[5/5] Initializing Combined Results Reranker: {cfg.model.get('reranker_name', 'BAAI/bge-reranker-v2-m3')}")
            self.reranker = Reranker(
                model_name=cfg.model.get("reranker_name", "BAAI/bge-reranker-v2-m3"),
                cache_dir=cfg.model.cache_dir
            )
        else:
            logger.info(f"[5/5] Reranker disabled for combined results")
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
    async def retrieve_local_documents(self, ev: QueryGenerationEvent) -> LocalRetrievalEvent:
        """
        Retrieve documents from local index using SOTA multi-retriever.
        
        Args:
            ev: QueryGenerationEvent with question queries
            
        Returns:
            LocalRetrievalEvent with local retrieval results
        """
        logger.info(f"📚 [Step 1/4] Local Document Retrieval")
        
        local_docs = []
        
        if self.retriever:
            logger.info(f"  SOTA Multi-Retrieval (ColBERT + Dense + Sparse) from local documents")
            
            # Retrieve from local documents using SOTA multi-retriever
            # retrieve_texts_batch returns list of text strings from top nodes
            local_docs = self.retriever.retrieve_texts_batch(ev.question_queries)
            
            if local_docs:
                logger.info(f"  ✓ Retrieved {len(local_docs)} documents from local index")
            else:
                logger.info(f"  No results from local index")
        else:
            logger.info(f"  Hybrid retrieval disabled or unavailable")
        
        return LocalRetrievalEvent(
            question=ev.question,
            question_queries=ev.question_queries,
            options=ev.options,
            local_documents=local_docs
        )
    
    @step
    async def search_web(self, ev: LocalRetrievalEvent) -> WebSearchEvent:
        """
        Execute web search using DDGS for current information.
        
        Args:
            ev: LocalRetrievalEvent with local documents
            
        Returns:
            WebSearchEvent with web search results
        """
        logger.info(f"🌐 [Step 2/4] Web Search (DDGS)")
        
        web_docs = []
        include_options = self.cfg.retrieval.get("include_options_in_query", True)
        
        for q in ev.question_queries:
            logger.info(f"  Searching: '{q}'")
            
            # Build query with options if configured
            query = q
            if include_options:
                query = q + " " + " ".join(ev.options.values())
            
            # Always get search results as list to avoid concatenating multiple snippets
            # into a single document (which would cause \n---\n separators in reranked contexts)
            search_results = self.search_engine.search_web(
                query, 
                max_results=self.cfg.retrieval.get("max_web_search_results", 5),
                return_list=True
            )
            if search_results:
                web_docs.extend(search_results)

            time.sleep(3.0)  # To avoid rate limiting
        
        logger.info(f"  ✓ Retrieved {len(web_docs)} documents from web search")
        
        return WebSearchEvent(
            question=ev.question,
            question_queries=ev.question_queries,
            options=ev.options,
            local_documents=ev.local_documents,
            web_documents=web_docs
        )
    
    @step
    async def combine_and_rerank(self, ev: WebSearchEvent) -> CombinedRetrievalEvent:
        """
        Combine local and web results, deduplicate, and optionally rerank.
        
        Args:
            ev: WebSearchEvent with both local and web documents
            
        Returns:
            CombinedRetrievalEvent with final combined context
        """
        logger.info(f"🔄 [Step 3/4] Combining & Reranking Results")
        
        # Combine all snippets
        all_snippets = ev.local_documents + ev.web_documents
        
        # Deduplicate
        if all_snippets:
            original_count = len(all_snippets)
            all_snippets = list(set(all_snippets))
            
            if original_count > len(all_snippets):
                logger.info(f"  Deduplicated: {original_count} → {len(all_snippets)} snippets")
        
        # Rerank if enabled, otherwise limit to top_k
        top_k = self.cfg.retrieval.get("reranker_top_k", 5)
        
        if self.use_reranker and all_snippets and self.reranker:
            logger.info(f"  Reranking: Selecting top {top_k} from {len(all_snippets)} total snippets...")
            reranked = self.reranker.rerank(
                query=ev.question,
                documents=all_snippets,
                top_k=top_k
            )
            # Extract just the documents from reranked results
            context_snippets = [doc for _, doc, _ in reranked]
            logger.info(f"  ✓ Selected top {len(context_snippets)} snippets after reranking")
        else:
            # Even without reranking, limit to top_k to avoid context overflow
            context_snippets = all_snippets[:top_k]
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
        logger.info(f"✅ [Step 4/4] Option Verification")
        
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

    
