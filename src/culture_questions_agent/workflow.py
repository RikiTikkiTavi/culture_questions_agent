"""LlamaIndex Workflow for Cultural QA using Base Model Completion."""
import os
from pathlib import Path
from typing import List

import torch
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


class RetrievalEvent(Event):
    """Event carrying retrieved nodes."""
    query: str
    nodes: List[NodeWithScore]


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
        
        print("="*80)
        print("Initializing Cultural QA Workflow...")
        print("="*80)
        
        # Set cache directory
        os.environ["HF_HOME"] = cfg.model.cache_dir
        
        # [1] Load LLM (Base Model - Completion Mode)
        print(f"\n[1/3] Loading LLM: {cfg.model.llm_name}")
        print(f"  - Mode: Completion (Base Model)")
        print(f"  - Dtype: bfloat16")
        print(f"  - Device: auto (H100)")
        
        self.llm = HuggingFaceLLM(
            model_name=cfg.model.llm_name,
            tokenizer_name=cfg.model.llm_name,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "device_map": "auto",
                "cache_dir": cfg.model.cache_dir,
            },
            tokenizer_kwargs={
                "cache_dir": cfg.model.cache_dir,
            },
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False,
        )
        
        # [2] Load Embedding Model
        print(f"\n[2/3] Loading embedding model: {cfg.model.embed_name}")
        embed_model = HuggingFaceEmbedding(
            model_name=cfg.model.embed_name,
            cache_folder=cfg.model.cache_dir,
        )
        Settings.embed_model = embed_model
        
        # [3] Load Index and Setup Hybrid Retrieval
        print(f"\n[3/3] Loading index from: {cfg.storage.persist_dir}")
        storage_context = StorageContext.from_defaults(
            persist_dir=cfg.storage.persist_dir
        )
        self.index = load_index_from_storage(storage_context)
        
        # Get all nodes for BM25
        nodes = list(self.index.docstore.docs.values())
        
        # Setup retrievers
        vector_retriever = self.index.as_retriever(
            similarity_top_k=cfg.retrieval.top_k
        )
        
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=cfg.retrieval.top_k,
        )
        
        # Hybrid retrieval via QueryFusionRetriever
        self.retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            similarity_top_k=cfg.retrieval.top_k,
            mode=cfg.retrieval.mode,
        )
        
        print("\n" + "="*80)
        print("✓ Workflow initialized successfully!")
        print("="*80 + "\n")
    
    @step
    async def retrieve(self, ctx, ev: StartEvent) -> RetrievalEvent:
        """
        Perform hybrid retrieval.
        
        Args:
            ctx: Workflow context
            ev: StartEvent containing the query
            
        Returns:
            RetrievalEvent with retrieved nodes
        """
        query = ev.get("query")
        if not query:
            raise ValueError("No query provided in StartEvent")
        
        print(f"\n🔍 Retrieving context for: '{query}'")
        nodes = await self.retriever.aretrieve(query)
        print(f"  ✓ Retrieved {len(nodes)} nodes")
        
        return RetrievalEvent(query=query, nodes=nodes)
    
    @step
    async def generate(self, ctx, ev: RetrievalEvent) -> StopEvent:
        """
        Generate answer using base model completion.
        
        Args:
            ctx: Workflow context
            ev: RetrievalEvent with query and nodes
            
        Returns:
            StopEvent with the final answer
        """
        # Prepare context from retrieved nodes
        context_parts = []
        for i, node in enumerate(ev.nodes[:3], 1):  # Use top 3 nodes
            text = node.node.get_content()[:500]  # Truncate for prompt size
            context_parts.append(f"[{i}] {text}")
        
        context_text = "\n\n".join(context_parts)
        
        # Few-shot completion prompt for Base Model
        prompt = f"""Context:
{context_text}

Question: What is the capital of France?
Answer: Paris

Question: What is the traditional dress of Japan called?
Answer: Kimono

Question: {ev.query}
Answer:"""
        
        print(f"\n🤖 Generating answer with base model completion...")
        
        # Use complete() for base model
        response = self.llm.complete(prompt)
        answer = response.text.strip()
        
        # Clean up the answer (take only first line if multiple)
        if "\n" in answer:
            answer = answer.split("\n")[0].strip()
        
        print(f"  ✓ Answer: {answer}")
        
        return StopEvent(result=answer)
