"""Competition submission generation for MCQ task."""
import logging
import asyncio
from pathlib import Path
from typing import List, Dict
import pandas as pd
from tqdm.asyncio import tqdm as atqdm

import hydra
from omegaconf import DictConfig

from culture_questions_agent.data import read_mcq_data_test
from culture_questions_agent.structures import MCQQuestion
from culture_questions_agent.workflow import CulturalQAWorkflow

logger = logging.getLogger(__name__)


async def predict_single_question(workflow: CulturalQAWorkflow, question: MCQQuestion) -> str:
    """
    Predict answer for a single MCQ question.
    
    Args:
        workflow: Initialized CulturalQAWorkflow
        question: MCQ question to answer
        
    Returns:
        Predicted answer choice (A, B, C, or D)
    """
    try:
        result = await workflow.run(mcq_question=question)
        return result
    except Exception as e:
        logger.error(f"Error predicting for {question.msq_id}: {e}")
        # Return A as default fallback
        return "A"


async def predict_batch(workflow: CulturalQAWorkflow, questions: List[MCQQuestion]) -> List[Dict]:
    """
    Predict answers for a batch of questions.
    
    Args:
        workflow: Initialized CulturalQAWorkflow
        questions: List of MCQ questions
        
    Returns:
        List of prediction dictionaries with MCQID and answer choices
    """
    results = []
    
    # Process questions with progress bar
    for question in atqdm(questions, desc="Predicting answers"):
        predicted_answer = await predict_single_question(workflow, question)
        
        # Convert to boolean format for submission
        prediction = {
            "MCQID": question.msq_id,
            "A": predicted_answer == "A",
            "B": predicted_answer == "B",
            "C": predicted_answer == "C",
            "D": predicted_answer == "D",
        }
        
        results.append(prediction)
    
    return results


def save_mcq_submission(predictions: List[Dict], output_path: Path):
    """
    Save predictions in competition TSV format.
    
    Args:
        predictions: List of prediction dictionaries
        output_path: Path to save the TSV file
    """
    # Create DataFrame
    df = pd.DataFrame(predictions)
    
    # Ensure column order
    df = df[["MCQID", "A", "B", "C", "D"]]
    
    # Save as TSV
    df.to_csv(output_path, sep="\t", index=False)
    
    logger.info(f"✓ Saved {len(predictions)} predictions to {output_path}")


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    """
    Generate competition submission file for MCQ task.
    
    Reads test dataset, runs workflow for each question, and generates
    mcq_prediction.tsv in the required competition format.
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("="*80)
    logger.info("MCQ COMPETITION SUBMISSION GENERATION")
    logger.info("="*80)
    
    # Get paths from config
    test_data_path = Path(cfg.get("test_mcq_path", "data/test_dataset_mcq.csv"))
    output_path = Path(cfg.get("submission_output_path", "mcq_prediction.tsv"))
    
    logger.info(f"Test data: {test_data_path}")
    logger.info(f"Output: {output_path}")
    
    # Load test dataset
    logger.info(f"\n[1/3] Loading test dataset...")
    questions = read_mcq_data_test(test_data_path)
    logger.info(f"  ✓ Loaded {len(questions)} questions")
    
    # Initialize workflow
    logger.info(f"\n[2/3] Initializing workflow...")
    workflow = CulturalQAWorkflow(cfg)
    logger.info(f"  ✓ Workflow initialized")
    
    # Generate predictions
    logger.info(f"\n[3/3] Generating predictions...")
    predictions = asyncio.run(predict_batch(workflow, questions))
    logger.info(f"  ✓ Generated {len(predictions)} predictions")
    
    # Save submission file
    logger.info(f"\nSaving submission file...")
    save_mcq_submission(predictions, output_path)
    
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
