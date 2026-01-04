"""Main entry point for Cultural QA system with Hydra configuration."""
import asyncio
import logging
from pathlib import Path

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf

from culture_questions_agent import builder
from culture_questions_agent.workflow import CulturalQAWorkflow
from culture_questions_agent.data import read_mcq_data

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for Cultural QA system.
    
    Args:
        cfg: Hydra configuration
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment("cultural_qa_system")
    mlflow.llama_index.autolog()

    logger.info("="*80)
    logger.info("CULTURAL QA SYSTEM - Offline Hybrid RAG")
    logger.info("="*80)
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("="*80)
    
    # Check if index exists
    persist_dir = Path(cfg.storage.persist_dir)
    
    if not persist_dir.exists() or not list(persist_dir.glob("*")):
        logger.info("📦 Index not found. Building atlas...")
        builder.build_atlas(cfg)
    else:
        logger.info(f"✓ Using existing index from: {persist_dir}")
    
    # Initialize workflow
    workflow = CulturalQAWorkflow(
        cfg=cfg,
        timeout=120,
        verbose=True
    )
    
    mcq_questions = read_mcq_data(Path("data/train_dataset_mcq.csv"))[:3]
    

    logger.info("="*80)
    logger.info("Running Example Queries")
    logger.info("="*80)
    
    async def run_queries():
        """Run all example queries."""
        results = []
        
        with mlflow.start_run(run_name="cultural_qa_inference"):
            mlflow.log_params({
                "llm_name": cfg.model.llm_name,
                "embed_name": cfg.model.embed_name,
                "top_k": cfg.retrieval.top_k,
                "mode": cfg.retrieval.mode,
            })
            
            for i, question in enumerate(mcq_questions, 1):
                logger.info(f"{'─'*80}")
                logger.info(f"Query {i}/{len(mcq_questions)}")
                logger.info(f"{'─'*80}")
                
                result = await workflow.run(mcq_question=question)
                results.append({"query": question.question, "answer": result})
                
                logger.info(f"📝 Final Answer: {result}")
                logger.info(f"{'─'*80}")
            
            # Log results
            mlflow.log_dict({"results": results}, "results.json")
        
        return results
    
    # Run async queries
    asyncio.run(run_queries())
    
    logger.info("="*80)
    logger.info("✓ All queries completed!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
