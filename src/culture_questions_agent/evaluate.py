"""Main entry point for Cultural QA system with Hydra configuration."""
import asyncio
from http import client
import logging
from pathlib import Path
from collections import defaultdict

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from culture_questions_agent.utils import flatten
from culture_questions_agent.workflow import CulturalQAWorkflow
from culture_questions_agent.data import read_mcq_data_train, read_saq_data_train
from culture_questions_agent.structures import MCQQuestionTrain, MCQQuestion, SAQQuestionTrain, SAQQuestion

from mlflow.genai import scorer

logger = logging.getLogger(__name__)

def build_mcq_predict_fn(wf: CulturalQAWorkflow):
    async def predict_fn(question: str, options: dict[str, str]) -> str:
        """Predict function for MCQ evaluation."""
        mcq_question = MCQQuestion(
            msq_id="",
            question=question,
            options=options,
        )
        return await wf.run(question=mcq_question)
            
    return predict_fn

def build_saq_predict_fn(wf: CulturalQAWorkflow):
    async def predict_fn(question: str) -> str:
        """Predict function for SAQ evaluation."""
        saq_question = SAQQuestion(question=question, saq_id="", pos_id=0)
        return await wf.run(question=saq_question)
            
    return predict_fn

@scorer
def mcq_exact_match(outputs, expectations) -> bool:
    """Exact match scorer for MCQ questions."""
    return outputs == expectations["answer"]

@scorer
def saq_exact_match(outputs, expectations) -> bool:
    """Exact match scorer for SAQ questions - checks if output matches any acceptable answer."""
    output_normalized = outputs.strip().lower()
    acceptable_answers = expectations["answers"]
    return any(output_normalized == answer.strip().lower() for answer in acceptable_answers)

@scorer
def saq_partial_match(outputs, expectations) -> bool:
    """Partial match scorer for SAQ questions - checks if output contains any acceptable answer."""
    output_normalized = outputs.strip().lower()
    acceptable_answers = expectations["answers"]
    return any(answer.strip().lower() in output_normalized for answer in acceptable_answers)

def compute_country_metrics(eval_results, mcq_questions: list[MCQQuestionTrain]) -> dict:
    """
    Compute accuracy metrics by country for MCQ questions.
    
    Args:
        eval_results: MLflow evaluation results
        mcq_questions: Original MCQ questions with country information
        
    Returns:
        Dictionary with country-level metrics
    """
    # Track correct/total by country
    country_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    #print(eval_results.tables)
    #print(eval_results.tables["eval_results"].columns)

    # Get predictions from eval results
    predictions = eval_results.tables["eval_results"]["mcq_exact_match/value"]
    
    # Process each question
    for idx, (question, prediction) in enumerate(zip(mcq_questions, predictions)):
        correct_answer = question.answer
        is_correct = bool(prediction)
        
        # Get country for the correct answer
        country = question.countries.get(correct_answer, "Unknown")
        
        # Update stats for this country
        country_stats[country]["total"] += 1
        if is_correct:
            country_stats[country]["correct"] += 1
    
    # Calculate accuracy for each country
    country_metrics = {}
    for country, stats in country_stats.items():
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        country_metrics[country] = {
            "accuracy": accuracy,
            "correct": stats["correct"],
            "total": stats["total"],
        }
    
    return country_metrics

def build_eval_dataset_from_mcq_questions(mcq_questions: list[MCQQuestionTrain]) -> list[dict]:
    """Build evaluation dataset from MCQ questions."""
    eval_data = []
    for question in mcq_questions:
        eval_data.append({
            "inputs": {"question": question.question, "options": question.options},
            "expectations": {"answer": question.answer, "countries": question.countries}
        })
    return eval_data

def build_eval_dataset_from_saq_questions(saq_questions: list[SAQQuestionTrain]) -> list[dict]:
    """Build evaluation dataset from SAQ questions."""
    eval_data = []
    for question in saq_questions:
        eval_data.append({
            "inputs": {"question": question.question},
            "expectations": {"answers": question.answers}
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

    # Get task type from config
    task_type = cfg.get("task_type", "mcq").lower()
    
    logger.info("="*80)
    logger.info(f"CULTURAL QA SYSTEM - {task_type.upper()} Evaluation")
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
    
    max_questions = cfg.get("evaluation", {}).get("max_questions", None)
    
    if task_type == "mcq":
        # Load MCQ questions
        all_questions = read_mcq_data_train(Path("data/train_dataset_mcq.csv"))
        questions = all_questions[:max_questions] if max_questions else all_questions

        # print set of countries
        countries = set()
        for question in questions:
            countries.update(question.countries.values())
        logger.info(f"Evaluating on {len(questions)} MCQ questions from {len(countries)} countries.")
        logger.info(f"Countries: {', '.join(sorted(countries))}")

        logger.info("="*80)
        logger.info("Running MCQ Evaluation")
        logger.info("="*80)

        eval_results = mlflow.genai.evaluate( # type: ignore
            data=build_eval_dataset_from_mcq_questions(questions),
            predict_fn=build_mcq_predict_fn(workflow),
            scorers=[
                mcq_exact_match, # type: ignore
            ],
        )
        
        # Compute country-level metrics
        logger.info("="*80)
        logger.info("Computing Country-Level Metrics")
        logger.info("="*80)
        
        country_metrics = compute_country_metrics(eval_results, questions)
        
        # Sort countries by accuracy (descending)
        sorted_countries = sorted(
            country_metrics.items(),
            key=lambda x: x[1]["accuracy"],
            reverse=True
        )
        
        # Display country metrics
        logger.info("\nAccuracy by Country:")
        logger.info("-" * 60)
        for country, metrics in sorted_countries:
            logger.info(
                f"{country:30s}: {metrics['accuracy']:.2%} "
                f"({metrics['correct']}/{metrics['total']})"
            )
        
        # Log country metrics to MLflow
        for country, metrics in country_metrics.items():
            mlflow.log_metric(f"accuracy_{country.replace(' ', '_')}", metrics["accuracy"])
            mlflow.log_metric(f"correct_{country.replace(' ', '_')}", metrics["correct"])
            mlflow.log_metric(f"total_{country.replace(' ', '_')}", metrics["total"])
        
        # Create and log country metrics table
        country_df = pd.DataFrame([
            {
                "Country": country,
                "Accuracy": f"{metrics['accuracy']:.2%}",
                "Correct": metrics['correct'],
                "Total": metrics['total'],
            }
            for country, metrics in sorted_countries
        ])
        
        country_table_path = "country_metrics.csv"
        country_df.to_csv(country_table_path, index=False)
        mlflow.log_artifact(country_table_path)
        
        logger.info("="*80)
        logger.info("Evaluation Complete")
        logger.info("="*80)
        logger.info(f"Overall Accuracy: {eval_results.metrics['mcq_exact_match/mean']:.2%}")
        logger.info(f"Country Metrics saved to: {country_table_path}")
        logger.info("="*80)
    
    elif task_type == "saq":
        # Load SAQ questions
        all_questions = read_saq_data_train(Path("data/train_dataset_saq.csv"))
        questions = all_questions[:max_questions] if max_questions else all_questions
        
        logger.info(f"Evaluating on {len(questions)} SAQ questions.")
        
        logger.info("="*80)
        logger.info("Running SAQ Evaluation")
        logger.info("="*80)

        eval_results = mlflow.genai.evaluate( # type: ignore
            data=build_eval_dataset_from_saq_questions(questions),
            predict_fn=build_saq_predict_fn(workflow),
            scorers=[
                saq_exact_match, # type: ignore
                saq_partial_match, # type: ignore
            ],
        )
        
        logger.info("="*80)
        logger.info("Evaluation Complete")
        logger.info("="*80)
        logger.info(f"Exact Match Accuracy: {eval_results.metrics['saq_exact_match/mean']:.2%}")
        logger.info(f"Partial Match Accuracy: {eval_results.metrics['saq_partial_match/mean']:.2%}")
        logger.info("="*80)
    
    else:
        raise ValueError(f"Unknown task_type: {task_type}. Must be 'mcq' or 'saq'.")


if __name__ == "__main__":
    main()
