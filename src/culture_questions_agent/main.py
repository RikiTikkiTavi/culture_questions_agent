"""Main entry point for Cultural QA system with Hydra configuration."""
import asyncio
import logging
from pathlib import Path

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf

from culture_questions_agent.workflow import CulturalQAWorkflow
from culture_questions_agent.data import read_mcq_data, read_mcq_data_train

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
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup MLflow
    # Ensure tracking directory exists
    tracking_dir = Path("tracking")
    tracking_dir.mkdir(exist_ok=True)
    
    # Set tracking URI with absolute path
    tracking_uri = f"sqlite:///{tracking_dir.absolute()}/mlruns.sqlite"
    mlflow.set_tracking_uri(tracking_uri)
    
    # Set experiment with artifact location
    experiment_name = "cultural_qa_system"
    artifact_location = str((tracking_dir / "artifacts").absolute())
    
    try:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=artifact_location
        )
    except Exception:
        # Experiment already exists
        pass
    
    mlflow.set_experiment(experiment_name)
    mlflow.autolog()

    logger.info("="*80)
    logger.info("CULTURAL QA SYSTEM - NLL-based Approach")
    logger.info("="*80)
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("="*80)
    
    # Initialize workflow (no index building needed)
    workflow = CulturalQAWorkflow(
        cfg=cfg,
        timeout=120,
        verbose=True
    )
    
    mcq_questions = read_mcq_data_train(Path("data/train_dataset_mcq.csv"))[:3]
    

    logger.info("="*80)
    logger.info("Running Example Queries")
    logger.info("="*80)
    
    async def run_queries():
        """Run all example queries."""
        results = []
        
        with mlflow.start_run(run_name="cultural_qa_nll_inference"):
            mlflow.log_params({
                "llm_name": cfg.model.llm_name,
                "num_queries": cfg.retrieval.get("num_queries", 10),
                "use_question_context": cfg.retrieval.get("use_question_context", True),
                "use_option_context": cfg.retrieval.get("use_option_context", True),
            })
            
            for i, question in enumerate(mcq_questions, 1):
                logger.info(f"{'─'*80}")
                logger.info(f"Query {i}/{len(mcq_questions)}")
                logger.info(f"{'─'*80}")
                
                result = await workflow.run(mcq_question=question)
                results.append({"query": question.question, "answer": result})
                
                logger.info(f"📝 Final Answer: {result}")
                logger.info(f"Correct Answer: {question.answer}")
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
