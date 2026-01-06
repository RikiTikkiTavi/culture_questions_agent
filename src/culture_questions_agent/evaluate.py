"""Main entry point for Cultural QA system with Hydra configuration."""
import asyncio
from http import client
import logging
from pathlib import Path

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf

from culture_questions_agent.utils import flatten
from culture_questions_agent.workflow import CulturalQAWorkflow
from culture_questions_agent.data import read_mcq_data, read_mcq_data_train
from culture_questions_agent.structures import MCQQuestionTrain, MCQQuestion

from mlflow.genai import scorer

logger = logging.getLogger(__name__)

def build_predict_fn(wf: CulturalQAWorkflow):
    async def predict_fn(question: str, options: dict[str, str]) -> str:
        """Predict function for MLflow evaluation."""
        mcq_question = MCQQuestion(
            question=question,
            options=options,
        )
        return await wf.run(mcq_question=mcq_question)
            
    return predict_fn

@scorer
def exact_match(outputs, expectations) -> bool:
    return outputs == expectations["answer"]

def build_eval_dataset_from_mcq_questions(mcq_questions: list[MCQQuestionTrain]) -> list[dict]:
    """Build evaluation dataset from MCQ questions."""
    eval_data = []
    for question in mcq_questions:
        eval_data.append({
            "inputs": {"question": question.question, "options": question.options},
            "expectations": {"answer": question.answer}
        })
    return eval_data

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

    # Log hydra config to mlflow
    mlflow.log_params(flatten(OmegaConf.to_container(cfg, resolve=True)))

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
    
    mcq_questions = read_mcq_data_train(Path("data/train_dataset_mcq.csv"))[:50]


    logger.info("="*80)
    logger.info("Running Example Queries")
    logger.info("="*80)


    mlflow.genai.evaluate( # type: ignore
        data=build_eval_dataset_from_mcq_questions(mcq_questions),
        predict_fn=build_predict_fn(workflow),
        scorers=[
            exact_match, # type: ignore
        ],
    )


if __name__ == "__main__":
    main()
