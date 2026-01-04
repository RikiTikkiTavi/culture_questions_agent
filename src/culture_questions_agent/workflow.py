"""LlamaIndex Workflow for Cultural QA using Base Model Completion."""
import logging
import os
from pathlib import Path
from typing import List

import mlflow
import torch
from jinja2 import Environment, FileSystemLoader
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    load_index_from_storage,
)
from llama_index.core.schema import NodeWithScore
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

from culture_questions_agent.structures import MCQQuestion

logger = logging.getLogger(__name__)

class QueryGenerationEvent(Event):
    """Event carrying generated queries for retrieval."""
    original_question: str
    verification_queries: List[tuple[str, str]]  # [(query, option_key), ...]
    options: dict[str, str]
    options_formatted: str


class RetrievalEvent(Event):
    """Event carrying retrieved nodes."""
    original_question: str
    queries: List[str]
    nodes: List[NodeWithScore]
    options: dict[str, str]
    options_formatted: str


class CulturalQAWorkflow(Workflow):
    """
    Event-driven workflow for Cultural QA using:
    - Hybrid retrieval (BGE-M3 + BM25)
    - Base model completion (Llama-3-8B)
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
        logger.info("Initializing Cultural QA Workflow...")
        logger.info("="*80)
        
        # Set cache directory
        os.environ["HF_HOME"] = cfg.model.cache_dir
        
        # Load Jinja2 templates
        template_dir = Path(__file__).parent.parent.parent / "prompts"
        self.jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))
        
        # [1] Load LLM (Base Model - Completion Mode)
        logger.info(f"[1/3] Loading LLM: {cfg.model.llm_name}")
        logger.info("  - Mode: Completion (Base Model)")
        logger.info("  - Dtype: bfloat16")
        logger.info("  - Device: auto (H100)")
        
        self.llm = HuggingFaceLLM(
            model_name=cfg.model.llm_name,
            tokenizer_name=cfg.model.llm_name,
            model_kwargs={
                "cache_dir": cfg.model.cache_dir,
            },
            tokenizer_kwargs={
                "cache_dir": cfg.model.cache_dir,
            },
            generate_kwargs={"temperature": 0.1},
            max_new_tokens=256,
        )
        
        # [2] Load Embedding Model
        logger.info(f"[2/3] Loading embedding model: {cfg.model.embed_name}")
        embed_model = HuggingFaceEmbedding(
            model_name=cfg.model.embed_name,
            cache_folder=cfg.model.cache_dir,
        )
        Settings.embed_model = embed_model
        
        # [3] Load Index and Setup Hybrid Retrieval
        logger.info(f"[3/3] Loading index from: {cfg.storage.persist_dir}")
        storage_context = StorageContext.from_defaults(
            persist_dir=cfg.storage.persist_dir
        )
        self.index = load_index_from_storage(storage_context)
        
        # Get all nodes for BM25
        nodes = list(self.index.docstore.docs.values())
        
        # Setup retrievers with lower top_k for focused, per-option retrieval
        vector_retriever = self.index.as_retriever(
            similarity_top_k=2  # Precise results per query
        )
        
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=2,  # Precise results per query
        )
        
        # Hybrid retrieval via QueryFusionRetriever
        # Configure for single-query mode (no internal query generation)
        self.retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            llm=self.llm,  # Use our HuggingFace LLM instead of OpenAI
            similarity_top_k=2,  # Fewer chunks per query, but more precise
            num_queries=1,  # Disable internal query generation
            mode=cfg.retrieval.mode,  # Use config mode
            use_async=False,  # Sync mode for iterative control
        )
        
        logger.info("="*80)
        logger.info("✓ Workflow initialized successfully!")
        logger.info("="*80)
    
    @step
    async def generate_queries(self, ev: StartEvent) -> QueryGenerationEvent:
        """
        Generate smart queries by filtering out generic options.
        
        Args:
            ev: StartEvent containing MCQ question and options
            
        Returns:
            QueryGenerationEvent with filtered queries
        """
        # Extract question data
        mcq_question = ev.get("mcq_question")
        if not mcq_question:
            raise ValueError("mcq_question required for smart query generation")
        
        original_question = mcq_question.question
        options = mcq_question.options
        
        # Format options for prompt
        options_formatted = "\n".join([f"{k}. {v}" for k, v in options.items()])
        
        # Define generic phrases that should NOT be searched
        generic_phrases = {
            "as usual", "unknown", "none", "other", 
            "all of the above", "various", "none of the above",
            "not applicable", "n/a", "depends", "it varies"
        }
        
        # Collect verification queries for specific options
        verification_queries = []  # [(query_str, option_key), ...]
        
        logger.info(f"🎯 Generating smart queries for: '{original_question}'")
        
        # Analyze each option
        for opt_key, opt_text in options.items():
            # Clean text
            text_clean = opt_text.lower().strip()
            
            # Filter 1: Generic phrases
            if text_clean in generic_phrases:
                logger.info(f"  ⊗ Skipping generic option {opt_key}: '{opt_text}'")
                continue
            
            # Filter 2: Very short options (less than 3 chars)
            if len(text_clean) < 3:
                logger.info(f"  ⊗ Skipping short option {opt_key}: '{opt_text}'")
                continue
            
            # Filter 3: Long sentences (likely descriptive, not entities)
            # Cultural entities are usually 1-4 words
            word_count = len(text_clean.split())
            if word_count > 4:
                logger.info(f"  ⊗ Skipping sentence option {opt_key}: '{opt_text}' ({word_count} words)")
                continue
            
            # This looks like a potential cultural entity
            # Add query with "origin meaning" to force cultural context
            entity_query = f"{opt_text} origin meaning"
            verification_queries.append((entity_query, opt_key))
            logger.info(f"  ✓ Added entity query {opt_key}: '{entity_query}'")
        
        logger.info(f"  → Total verification queries: {len(verification_queries)}")
        
        return QueryGenerationEvent(
            original_question=original_question,
            verification_queries=verification_queries,
            options=options,
            options_formatted=options_formatted
        )
    
    @step
    async def retrieve(self, ev: QueryGenerationEvent) -> RetrievalEvent:
        """
        Perform iterative retrieval to guarantee coverage for each option.
        
        Strategy:
        1. Retrieve context for main question (Top-5)
        2. Retrieve context for each option entity (Top-2 per option)
        3. Tag nodes with their retrieval source
        4. Deduplicate and combine
        
        Args:
            ev: QueryGenerationEvent with verification queries
            
        Returns:
            RetrievalEvent with source-tagged nodes
        """
        with mlflow.start_span(name="retrieve") as span:
            span.set_inputs({
                "original_question": ev.original_question, 
                "num_verification_queries": len(ev.verification_queries)
            })
            logger.info(f"🔍 Starting iterative retrieval...")
            
            all_nodes = []
            seen_node_ids = set()
            
            # STEP A: Retrieve context for Main Question (higher top_k)
            logger.info(f"  [STEP A] Main Question: '{ev.original_question}'")
            main_nodes = self.retriever.retrieve(ev.original_question)
            
            for node in main_nodes:
                # Tag with source
                if not hasattr(node.node, 'metadata') or node.node.metadata is None:
                    node.node.metadata = {}
                node.node.metadata["retrieval_source"] = "Main Question"
                all_nodes.append(node)
                seen_node_ids.add(node.node.node_id)
            
            logger.info(f"      ✓ Retrieved {len(main_nodes)} nodes for main question")
            
            # STEP B: Iterative Option-Specific Retrieval
            logger.info(f"  [STEP B] Verifying {len(ev.verification_queries)} options...")
            
            for i, (query_str, opt_key) in enumerate(ev.verification_queries, 1):
                logger.info(f"      [{i}/{len(ev.verification_queries)}] Option {opt_key}: '{query_str}'")
                
                option_nodes = self.retriever.retrieve(query_str)
                
                new_nodes_count = 0
                for node in option_nodes:
                    node_id = node.node.node_id
                    if node_id not in seen_node_ids:
                        # Tag with source
                        if not hasattr(node.node, 'metadata') or node.node.metadata is None:
                            node.node.metadata = {}
                        node.node.metadata["retrieval_source"] = f"Option {opt_key}"
                        node.node.metadata["option_key"] = opt_key
                        node.node.metadata["verification_query"] = query_str
                        
                        all_nodes.append(node)
                        seen_node_ids.add(node_id)
                        new_nodes_count += 1
                
                logger.info(f"          ✓ Retrieved {len(option_nodes)} nodes ({new_nodes_count} new)")
            
            # STEP C: Combine and organize
            logger.info(f"  [STEP C] Total unique nodes: {len(all_nodes)}")
            
            # Sort by score within each source group
            # Keep main question nodes first, then option-specific nodes
            main_ctx_nodes = [n for n in all_nodes if n.node.metadata.get("retrieval_source") == "Main Question"]
            option_ctx_nodes = [n for n in all_nodes if n.node.metadata.get("retrieval_source") != "Main Question"]
            
            main_ctx_nodes.sort(key=lambda n: n.score or 0, reverse=True)
            option_ctx_nodes.sort(key=lambda n: n.score or 0, reverse=True)
            
            # Combine: Top-5 main + all option nodes
            final_nodes = main_ctx_nodes[:5] + option_ctx_nodes
            
            logger.info(f"  → Final: {len(final_nodes)} nodes ({len(main_ctx_nodes[:5])} main + {len(option_ctx_nodes)} option-specific)")
            
            span.set_outputs({"num_nodes": len(final_nodes)})
            span.set_attribute("num_main_nodes", len(main_ctx_nodes[:5]))
            span.set_attribute("num_option_nodes", len(option_ctx_nodes))
        
        return RetrievalEvent(
            original_question=ev.original_question,
            queries=[ev.original_question] + [q for q, _ in ev.verification_queries],
            nodes=final_nodes,
            options=ev.options,
            options_formatted=ev.options_formatted
        )
    
    @step
    async def reason_and_answer(self, ev: RetrievalEvent) -> StopEvent:
        """
        Reason about the answer using Sherlock Holmes exclusion principle.
        
        Args:
            ev: RetrievalEvent with context and options
            
        Returns:
            StopEvent with the final answer
        """
        with mlflow.start_span(name="reason_and_answer") as span:
            span.set_inputs({
                "question": ev.original_question, 
                "num_nodes": len(ev.nodes)
            })
            
            # Prepare context organized by source
            # Group nodes by retrieval source
            main_nodes = [n for n in ev.nodes if n.node.metadata.get("retrieval_source") == "Main Question"]
            option_nodes = [n for n in ev.nodes if n.node.metadata.get("retrieval_source") != "Main Question"]
            
            context_str = "="*60 + "\n"
            context_str += "CONTEXT ORGANIZED BY SOURCE\n"
            context_str += "="*60 + "\n\n"
            
            # Main Question Context
            context_str += "[A] Context regarding Main Question:\n"
            context_str += "-"*60 + "\n"
            for i, node in enumerate(main_nodes[:5], 1):
                text = node.node.get_content()[:350]  # Truncate each node
                context_str += f"  [{i}] {text}\n\n"
            
            # Option-Specific Context (grouped by option)
            if option_nodes:
                context_str += "\n[B] Context regarding Specific Options:\n"
                context_str += "-"*60 + "\n"
                
                # Group by option key
                from collections import defaultdict
                option_groups = defaultdict(list)
                for node in option_nodes:
                    opt_key = node.node.metadata.get("option_key", "Unknown")
                    option_groups[opt_key].append(node)
                
                # Output each option's context
                for opt_key in sorted(option_groups.keys()):
                    opt_text = ev.options.get(opt_key, "Unknown")
                    nodes_for_option = option_groups[opt_key][:2]  # Top 2 per option
                    
                    context_str += f"\n  Option {opt_key}: {opt_text}\n"
                    for i, node in enumerate(nodes_for_option, 1):
                        text = node.node.get_content()[:300]
                        context_str += f"    [{i}] {text}\n"
            
            context_str += "\n" + "="*60 + "\n\n"
            
            # Build Sherlock Holmes reasoning prompt with exclusion principle
            prompt = (
                f"Context Information:\n{context_str}\n"
                "="*60 + "\n"
                "Task: Identify the correct option using the Exclusion Principle.\n\n"
                "Instructions:\n"
                "1. If an option is a SPECIFIC cultural term (e.g., 'Hanbok', 'Kimono'), check if it belongs to the target country/culture in the Question.\n"
                "2. If ALL specific cultural terms belong to WRONG countries/cultures, choose the GENERIC option (e.g., 'as usual', 'none', 'unknown').\n"
                "3. If NO specific terms exist, choose the most reasonable generic option.\n\n"
                "Example Reasoning Trace:\n"
                "Question: What is the traditional clothing in China?\n"
                "Options:\n"
                "A. Hanbok\n"
                "B. Kebaya\n"
                "C. As usual\n\n"
                "Analysis:\n"
                "- Option A: Hanbok → Context says: Korean traditional dress → Incorrect (Korea ≠ China)\n"
                "- Option B: Kebaya → Context says: Indonesian traditional dress → Incorrect (Indonesia ≠ China)\n"
                "- Option C: As usual → Generic option → Correct by EXCLUSION (all specific terms eliminated)\n\n"
                "Answer: C\n\n"
                "="*60 + "\n\n"
                f"Question: {ev.original_question}\n"
                f"Options:\n{ev.options_formatted}\n\n"
                "Analysis:\n"
            )
            
            logger.info(f"🕵️ Reasoning with Sherlock Holmes exclusion principle...")
            logger.debug(f"Prompt length: {len(prompt)} chars")
            
            # Generate reasoning and answer
            response = self.llm.complete(prompt)
            full_response = response.text.strip()
            
            logger.info(f"  🧠 Reasoning:\n{full_response}")
            
            # Extract the final answer (look for "Answer:" pattern)
            answer = None
            lines = full_response.split("\n")
            
            # Strategy 1: Look for explicit "Answer: X" pattern
            for line in lines:
                line_clean = line.strip()
                if line_clean.lower().startswith("answer:"):
                    # Extract letter after "Answer:"
                    answer_part = line_clean.split(":", 1)[1].strip()
                    # Get first character that's a valid option
                    for char in answer_part.upper():
                        if char in ev.options:
                            answer = char
                            break
                    if answer:
                        break
            
            # Strategy 2: Find last line with a single letter option
            if not answer:
                for line in reversed(lines):
                    line_clean = line.strip().upper()
                    if len(line_clean) == 1 and line_clean in ev.options:
                        answer = line_clean
                        break
            
            # Strategy 3: Find any option letter in the last few lines
            if not answer:
                for line in reversed(lines[-5:]):
                    for opt_key in ev.options.keys():
                        if opt_key.upper() in line.upper():
                            answer = opt_key.upper()
                            break
                    if answer:
                        break
            
            # Fallback: Use first option as default
            if not answer:
                answer = list(ev.options.keys())[0].upper()
                logger.warning(f"  ⚠️ Could not extract answer, defaulting to: {answer}")
            
            logger.info(f"  ✓ Final Answer: {answer}")
            
            span.set_outputs({"answer": answer, "reasoning": full_response})
            span.set_attribute("prompt_length", len(prompt))
        
        return StopEvent(result=answer)
    
