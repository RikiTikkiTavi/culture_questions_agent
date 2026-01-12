"""Competition submission generation for MCQ and SAQ tasks."""
import logging
import asyncio
from pathlib import Path
import tempfile
from typing import List, Dict, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import mlflow
import pandas as pd
from tqdm.asyncio import tqdm as atqdm
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf

from culture_questions_agent.data import read_mcq_data_test, read_saq_data_test
from culture_questions_agent.structures import MCQQuestion, SAQQuestion
from culture_questions_agent.utils import flatten
from culture_questions_agent.workflow import CulturalQAWorkflow

logger = logging.getLogger(__name__)


async def predict_single_mcq_question(
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
            predicted_answer = await workflow.run(question=question)

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


async def predict_single_saq_question(
    workflow: CulturalQAWorkflow,
    question: SAQQuestion,
    semaphore: asyncio.Semaphore
) -> Dict:
    """
    Generate answer for a single SAQ question with concurrency control.
    
    Args:
        workflow: Initialized CulturalQAWorkflow
        question: SAQ question to answer
        question_id: Question identifier
        semaphore: Semaphore to limit concurrent GPU operations
        
    Returns:
        Prediction dictionary with question_id and generated answer
    """
    async with semaphore:
        try:
            generated_answer = await workflow.run(question=question)
            
            prediction = {
                "pos": question.pos_id,
                "ID": question.saq_id,
                "answer": generated_answer,
            }
            return prediction
            
        except Exception as e:
            logger.error(f"Error generating answer for {question.saq_id}: {e}")
            # Return empty answer as fallback
            return {
                "pos": question.pos_id,
                "ID": question.saq_id,
                "answer": "",
            }


def process_mcq_chunk(args: tuple) -> List[Dict]:
    """
    Process a chunk of MCQ questions in a separate process.
    
    Args:
        args: Tuple of (cfg, questions, max_concurrent, chunk_id)
        
    Returns:
        List of prediction dictionaries
    """
    cfg, questions, max_concurrent, chunk_id = args
    
    # Initialize workflow in this process
    workflow = CulturalQAWorkflow(cfg, timeout=120)
    
    # Run async batch prediction
    async def _process():
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = [
            predict_single_mcq_question(workflow, question, semaphore)
            for question in questions
        ]
        
        results = []
        for coro in atqdm.as_completed(tasks, total=len(tasks), desc=f"Process {chunk_id}"):
            result = await coro
            results.append(result)
        
        return results
    
    return asyncio.run(_process())


async def predict_mcq_batch(
    cfg: DictConfig,
    questions: List[MCQQuestion],
    max_concurrent: int = 8,
    num_processes: int = 1
) -> List[Dict]:
    """
    Predict answers for a batch of MCQ questions with multi-process parallelism.
    
    Uses ProcessPoolExecutor to distribute questions across multiple processes,
    with asyncio concurrency within each process for GPU operations.
    
    Args:
        cfg: Hydra configuration
        questions: List of MCQ questions
        max_concurrent: Maximum concurrent predictions per process (default: 8)
        num_processes: Number of parallel processes (default: 1)
        
    Returns:
        List of prediction dictionaries with MCQID and answer choices
    """
    if num_processes <= 1:
        # Single process mode - use original async implementation
        workflow = CulturalQAWorkflow(cfg, timeout=120)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        logger.info(f"Processing {len(questions)} MCQ questions with max {max_concurrent} concurrent tasks")
        
        tasks = [
            predict_single_mcq_question(workflow, question, semaphore)
            for question in questions
        ]
        
        results = []
        for coro in atqdm.as_completed(tasks, total=len(tasks), desc="Predicting MCQ answers"):
            result = await coro
            results.append(result)
        
        results.sort(key=lambda x: x["MCQID"])
        return results
    
    # Multi-process mode
    logger.info(f"Processing {len(questions)} MCQ questions with {num_processes} processes, {max_concurrent} concurrent tasks per process")
    
    # Split questions into chunks for each process
    chunk_size = (len(questions) + num_processes - 1) // num_processes
    chunks = [questions[i:i + chunk_size] for i in range(0, len(questions), chunk_size)]
    
    # Prepare arguments for each process
    process_args = [
        (cfg, chunk, max_concurrent, i + 1)
        for i, chunk in enumerate(chunks)
    ]
    
    # Process chunks in parallel
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all tasks
        futures = [executor.submit(process_mcq_chunk, args) for args in process_args]
        
        # Collect results with progress tracking
        all_results = []
        for future in tqdm(futures, desc="Processing chunks", total=len(futures)):
            chunk_results = future.result()
            all_results.extend(chunk_results)
    
    # Sort results by MCQID to maintain order
    all_results.sort(key=lambda x: x["MCQID"])
    
    return all_results


def process_saq_chunk(args: tuple) -> List[Dict]:
    """
    Process a chunk of SAQ questions in a separate process.
    
    Args:
        args: Tuple of (cfg, questions, max_concurrent, chunk_id)
        
    Returns:
        List of prediction dictionaries
    """
    cfg, questions, max_concurrent, chunk_id = args
    
    # Initialize workflow in this process
    workflow = CulturalQAWorkflow(cfg, timeout=120)
    
    # Run async batch prediction
    async def _process():
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = [
            predict_single_saq_question(workflow, question, semaphore)
            for question in questions
        ]
        
        results = []
        for coro in atqdm.as_completed(tasks, total=len(tasks), desc=f"Process {chunk_id}"):
            result = await coro
            results.append(result)
        
        return results
    
    return asyncio.run(_process())


async def predict_saq_batch(
    cfg: DictConfig,
    questions: List[SAQQuestion],
    question_ids: List[str],
    max_concurrent: int = 8,
    num_processes: int = 1
) -> List[Dict]:
    """
    Generate answers for a batch of SAQ questions with multi-process parallelism.
    
    Uses ProcessPoolExecutor to distribute questions across multiple processes,
    with asyncio concurrency within each process for GPU operations.
    
    Args:
        cfg: Hydra configuration
        questions: List of SAQ questions
        question_ids: List of question identifiers
        max_concurrent: Maximum concurrent predictions per process (default: 8)
        num_processes: Number of parallel processes (default: 1)
        
    Returns:
        List of prediction dictionaries with ID and generated answer
    """
    if num_processes <= 1:
        # Single process mode - use original async implementation
        workflow = CulturalQAWorkflow(cfg, timeout=120)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        logger.info(f"Processing {len(questions)} SAQ questions with max {max_concurrent} concurrent tasks")
        
        tasks = [
            predict_single_saq_question(workflow, question, question_id, semaphore)
            for question, question_id in zip(questions, question_ids)
        ]
        
        results = []
        for coro in atqdm.as_completed(tasks, total=len(tasks), desc="Generating SAQ answers"):
            result = await coro
            results.append(result)
        
        results.sort(key=lambda x: x["ID"])
        return results
    
    # Multi-process mode
    logger.info(f"Processing {len(questions)} SAQ questions with {num_processes} processes, {max_concurrent} concurrent tasks per process")
    
    # Split questions into chunks for each process
    chunk_size = (len(questions) + num_processes - 1) // num_processes
    question_chunks = [questions[i:i + chunk_size] for i in range(0, len(questions), chunk_size)]
    id_chunks = [question_ids[i:i + chunk_size] for i in range(0, len(question_ids), chunk_size)]
    
    # Prepare arguments for each process
    process_args = [
        (cfg, q_chunk, max_concurrent, i + 1)
        for i, q_chunk in enumerate(question_chunks)
    ]
    
    # Process chunks in parallel
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all tasks
        futures = [executor.submit(process_saq_chunk, args) for args in process_args]
        
        # Collect results with progress tracking
        all_results = []
        for future in tqdm(futures, desc="Processing chunks", total=len(futures)):
            chunk_results = future.result()
            all_results.extend(chunk_results)
    
    return all_results


async def save_mcq_submission_async(predictions: List[Dict], output_path: Path, id_order: list[str]):
    """
    Save MCQ predictions in competition TSV format (async I/O).
    
    Args:
        predictions: List of MCQ prediction dictionaries
        output_path: Path to save the TSV file
    """
    # Use thread pool for I/O-bound operation
    loop = asyncio.get_event_loop()
    
    def _save():
        # Create DataFrame
        df = pd.DataFrame(predictions)
        
        # Sort by provided id_order (list of all ids), without modifying dataframe
        df = df.set_index("MCQID").loc[id_order].reset_index()
        
        # Ensure column order
        df = df[["MCQID", "A", "B", "C", "D"]]
        
        # Save as TSV
        df.to_csv(output_path, sep="\t", index=False)
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        await loop.run_in_executor(executor, _save)
    
    logger.info(f"✓ Saved {len(predictions)} MCQ predictions to {output_path}")


async def save_saq_submission_async(predictions: List[Dict], output_path: Path):
    """
    Save SAQ predictions in competition TSV format (async I/O).
    
    Args:
        predictions: List of SAQ prediction dictionaries
        output_path: Path to save the TSV file
        id_order: List of all SAQ IDs in the desired order
    """
    # Use thread pool for I/O-bound operation
    loop = asyncio.get_event_loop()
    
    def _save():
        # Create DataFrame
        df = pd.DataFrame(predictions)
        
        df = df.sort_values(by="pos", ascending=True)

        # Ensure column order and sort by provided id_order (list of all ids), without modifying dataframe
        df = df[["ID", "answer"]]
        
        # Save as TSV
        df.to_csv(output_path, sep="\t", index=False)
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        await loop.run_in_executor(executor, _save)
    
    logger.info(f"✓ Saved {len(predictions)} SAQ predictions to {output_path}")


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    """
    Generate competition submission file for MCQ or SAQ task based on configuration.
    
    Reads test dataset, runs workflow for each question with optimized
    concurrency, and generates prediction file in the required format.
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
    
    # Determine task type from config (default: MCQ)
    task_type = cfg.get("task_type", "mcq").lower()
    
    if task_type == "mcq":
        run_mcq_inference(cfg)
    elif task_type == "saq":
        run_saq_inference(cfg)
    else:
        raise ValueError(f"Unknown task_type: {task_type}. Must be 'mcq' or 'saq'")


def run_mcq_inference(cfg: DictConfig):
    """
    Run MCQ inference and generate submission file.
    
    Args:
        cfg: Hydra configuration
    """
    logger.info("="*80)
    logger.info("MCQ COMPETITION SUBMISSION GENERATION (MULTI-PROCESS)")
    logger.info("="*80)
    
    # Get paths from config
    test_data_path = Path(cfg.get("test_mcq_path", "data/test_dataset_mcq.csv"))
    output_path = Path(cfg.get("submission_output_path", "data/mcq_prediction.tsv"))
    
    # Get concurrency settings from config
    max_concurrent = cfg.get("inference", {}).get("max_concurrent", 8)
    num_processes = cfg.get("inference", {}).get("num_processes", mp.cpu_count())
    
    logger.info(f"Test data: {test_data_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Number of processes: {num_processes}")
    logger.info(f"Max concurrent predictions per process: {max_concurrent}")
    
    # Load test dataset
    logger.info(f"\n[1/2] Loading MCQ test dataset...")
    questions = read_mcq_data_test(test_data_path)
    logger.info(f"  ✓ Loaded {len(questions)} questions")
    
    # Generate predictions with multi-process parallelism
    logger.info(f"\n[2/2] Generating MCQ predictions ({num_processes} processes, {max_concurrent} concurrent per process)...")
    predictions = asyncio.run(predict_mcq_batch(cfg, questions, max_concurrent=max_concurrent, num_processes=num_processes))
    logger.info(f"  ✓ Generated {len(predictions)} predictions")
    
    # Save submission file
    logger.info(f"\nSaving MCQ submission file...")
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_output_path = Path(tmpdir) / "mcq_prediction.tsv"
        asyncio.run(save_mcq_submission_async(predictions, temp_output_path, id_order=[q.msq_id for q in questions]))
        # Log submission to mlflow as an artifact
        mlflow.log_artifact(str(temp_output_path), artifact_path="submissions")
    
    logger.info("="*80)
    logger.info("✓ MCQ SUBMISSION GENERATION COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nSubmission file: {output_path}")
    logger.info(f"Total predictions: {len(predictions)}")
    
    # Show sample predictions
    logger.info(f"\nSample predictions (first 5):")
    for pred in predictions[:5]:
        answer = [k for k, v in pred.items() if k != "MCQID" and v][0]
        logger.info(f"  {pred['MCQID']}: {answer}")


def run_saq_inference(cfg: DictConfig):
    """
    Run SAQ inference and generate submission file.
    
    Args:
        cfg: Hydra configuration
    """
    logger.info("="*80)
    logger.info("SAQ COMPETITION SUBMISSION GENERATION (MULTI-PROCESS)")
    logger.info("="*80)
    
    # Get paths from config
    test_data_path = Path(cfg.get("test_saq_path", "data/test_dataset_saq.csv"))
    output_path = Path(cfg.get("saq_submission_output_path", "data/saq_prediction.tsv"))
    
    # Get concurrency settings from config
    max_concurrent = cfg.get("inference", {}).get("max_concurrent", 8)
    num_processes = cfg.get("inference", {}).get("num_processes", mp.cpu_count())
    
    logger.info(f"Test data: {test_data_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Number of processes: {num_processes}")
    logger.info(f"Max concurrent predictions per process: {max_concurrent}")
    
    # Load test dataset
    logger.info(f"\n[1/2] Loading SAQ test dataset...")
    # Read CSV to get both questions and IDs
    df = pd.read_csv(test_data_path)
    index = df.index.tolist()
    questions = read_saq_data_test(test_data_path)
    question_ids = df["ID"].tolist()
    logger.info(f"  ✓ Loaded {len(questions)} questions")
    
    # Generate predictions with multi-process parallelism
    logger.info(f"\n[2/2] Generating SAQ answers ({num_processes} processes, {max_concurrent} concurrent per process)...")
    predictions = asyncio.run(predict_saq_batch(cfg, questions, question_ids, max_concurrent=max_concurrent, num_processes=num_processes))
    logger.info(f"  ✓ Generated {len(predictions)} answers")
    
    # Save submission file
    logger.info(f"\nSaving SAQ submission file...")
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_output_path = Path(tmpdir) / "saq_prediction.tsv"
        asyncio.run(save_saq_submission_async(predictions, temp_output_path))
        # Log submission to mlflow as an artifact
        mlflow.log_artifact(str(temp_output_path), artifact_path="submissions")

    logger.info("="*80)
    logger.info("✓ SAQ SUBMISSION GENERATION COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nSubmission file: {output_path}")
    logger.info(f"Total predictions: {len(predictions)}")
    
    # Show sample predictions
    logger.info(f"\nSample predictions (first 5):")
    for pred in predictions[:5]:
        logger.info(f"  {pred['ID']}: {pred['answer'][:100]}...")


if __name__ == "__main__":
    main()
