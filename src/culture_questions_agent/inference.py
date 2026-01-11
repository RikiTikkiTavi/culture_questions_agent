"""Competition submission generation for MCQ task."""
import logging
import asyncio
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
import mlflow
import pandas as pd
from tqdm.asyncio import tqdm as atqdm

import hydra
from omegaconf import DictConfig, OmegaConf

from culture_questions_agent.data import read_mcq_data_test
from culture_questions_agent.structures import MCQQuestion
from culture_questions_agent.utils import flatten
from culture_questions_agent.workflow import CulturalQAWorkflow

logger = logging.getLogger(__name__)


async def predict_single_question(
    workflow: CulturalQAWorkflow, 
    question: MCQQuestion,
    semaphore: asyncio.Semaphore
) -> Dict:
    """
    Predict answer for a single MCQ question with concurrency control.
    
    Args:
        workflow: Initialized CulturalQAWorkflow
        question: MCQ question to answer
        semaphore: Semaphore to limit concurrent GPU operations
        
    Returns:
        Prediction dictionary with MCQID and answer choices
    """
    async with semaphore:
        try:
            predicted_answer = await workflow.run(mcq_question=question)
            
            # Convert to boolean format for submission
            prediction = {
                "MCQID": question.msq_id,
                "A": predicted_answer == "A",
                "B": predicted_answer == "B",
                "C": predicted_answer == "C",
                "D": predicted_answer == "D",
            }
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting for {question.msq_id}: {e}")
            # Return A as default fallback
            return {
                "MCQID": question.msq_id,
                "A": True,
                "B": False,
                "C": False,
                "D": False,
            }


async def predict_batch(
    workflow: CulturalQAWorkflow, 
    questions: List[MCQQuestion],
    max_concurrent: int = 8
) -> List[Dict]:
    """
    Predict answers for a batch of questions with optimized concurrency.
    
    Uses asyncio to parallelize GPU-heavy operations (retrieval + generation)
    with a semaphore to prevent GPU memory overflow.
    
    Args:
        workflow: Initialized CulturalQAWorkflow
        questions: List of MCQ questions
        max_concurrent: Maximum concurrent predictions (default: 8 for H100)
        
    Returns:
        List of prediction dictionaries with MCQID and answer choices
    """
    # Create semaphore to limit concurrent GPU operations
    semaphore = asyncio.Semaphore(max_concurrent)
    
    logger.info(f"Processing {len(questions)} questions with max {max_concurrent} concurrent tasks")
    
    # Create all tasks
    tasks = [
        predict_single_question(workflow, question, semaphore)
        for question in questions
    ]
    
    # Process all questions concurrently with progress bar
    results = []
    for coro in atqdm.as_completed(tasks, total=len(tasks), desc="Predicting answers"):
        result = await coro
        results.append(result)
    
    # Sort results by MCQID to maintain order
    results.sort(key=lambda x: x["MCQID"])
    
    return results


async def save_mcq_submission_async(predictions: List[Dict], output_path: Path):
    """
    Save predictions in competition TSV format (async I/O).
    
    Args:
        predictions: List of prediction dictionaries
        output_path: Path to save the TSV file
    """
    # Use thread pool for I/O-bound operation
    loop = asyncio.get_event_loop()
    
    def _save():
        # Create DataFrame
        df = pd.DataFrame(predictions)
        
        # Ensure column order
        df = df[["MCQID", "A", "B", "C", "D"]]
        
        # Save as TSV
        df.to_csv(output_path, sep="\t", index=False)
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        await loop.run_in_executor(executor, _save)
    
    logger.info(f"✓ Saved {len(predictions)} predictions to {output_path}")


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    """
    Generate competition submission file for MCQ task.
    
    Reads test dataset, runs workflow for each question with optimized
    concurrency, and generates mcq_prediction.tsv in the required format.
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
    experiment_name = "inference"
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
    logger.info("MCQ COMPETITION SUBMISSION GENERATION (OPTIMIZED)")
    logger.info("="*80)
    
    # Get paths from config
    test_data_path = Path(cfg.get("test_mcq_path", "data/test_dataset_mcq.csv"))
    output_path = Path(cfg.get("submission_output_path", "mcq_prediction.tsv"))
    
    # Get concurrency settings from config (default: 8 for H100 GPU)
    max_concurrent = cfg.get("inference", {}).get("max_concurrent", 8)
    
    logger.info(f"Test data: {test_data_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Max concurrent predictions: {max_concurrent}")
    
    # Load test dataset
    logger.info(f"\n[1/3] Loading test dataset...")
    questions = read_mcq_data_test(test_data_path)[:16]
    logger.info(f"  ✓ Loaded {len(questions)} questions")
    
    # Initialize workflow
    logger.info(f"\n[2/3] Initializing workflow...")
    workflow = CulturalQAWorkflow(cfg)
    logger.info(f"  ✓ Workflow initialized")
    
    # Generate predictions with optimized concurrency
    logger.info(f"\n[3/3] Generating predictions (async with {max_concurrent} concurrent tasks)...")
    predictions = asyncio.run(predict_batch(workflow, questions, max_concurrent=max_concurrent))
    logger.info(f"  ✓ Generated {len(predictions)} predictions")
    
    # Save submission file
    logger.info(f"\nSaving submission file...")
    asyncio.run(save_mcq_submission_async(predictions, output_path))

    # Log submission to mlflow as an artifact
    mlflow.log_artifact(str(output_path), artifact_path="submissions")

    logger.info("="*80)
    logger.info("✓ SUBMISSION GENERATION COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nSubmission file: {output_path}")
    logger.info(f"Total predictions: {len(predictions)}")
    
    # Show sample predictions
    logger.info(f"\nSample predictions (first 5):")
    for pred in predictions[:5]:
        answer = [k for k, v in pred.items() if k != "MCQID" and v][0]
        logger.info(f"  {pred['MCQID']}: {answer}")


if __name__ == "__main__":
    main()
