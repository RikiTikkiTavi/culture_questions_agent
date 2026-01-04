"""Main entry point for Cultural QA system with Hydra configuration."""
import asyncio
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from culture_questions_agent import builder
from culture_questions_agent.workflow import CulturalQAWorkflow


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for Cultural QA system.
    
    Args:
        cfg: Hydra configuration
    """
    print("\n" + "="*80)
    print("CULTURAL QA SYSTEM - Offline Hybrid RAG")
    print("="*80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("="*80 + "\n")
    
    # Check if index exists
    persist_dir = Path(cfg.storage.persist_dir)
    
    if not persist_dir.exists() or not list(persist_dir.glob("*")):
        print("📦 Index not found. Building atlas...")
        builder.build_atlas(cfg)
    else:
        print(f"✓ Using existing index from: {persist_dir}\n")
    
    # Initialize workflow
    workflow = CulturalQAWorkflow(
        cfg=cfg,
        timeout=120,
        verbose=True
    )
    
    # Example queries
    queries = [
        "What is the traditional instrument of Japan?",
        "Describe the main features of Chinese culture",
        "What are the key elements of Brazilian music?",
    ]
    
    print("\n" + "="*80)
    print("Running Example Queries")
    print("="*80)
    
    async def run_queries():
        """Run all example queries."""
        for i, query in enumerate(queries, 1):
            print(f"\n{'─'*80}")
            print(f"Query {i}/{len(queries)}")
            print(f"{'─'*80}")
            
            result = await workflow.run(query=query)
            
            print(f"\n📝 Final Answer: {result}")
            print(f"{'─'*80}")
    
    # Run async queries
    asyncio.run(run_queries())
    
    print("\n" + "="*80)
    print("✓ All queries completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
