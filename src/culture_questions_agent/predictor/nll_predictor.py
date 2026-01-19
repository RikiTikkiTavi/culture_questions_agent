"""NLL-based LLM predictor using HuggingFace transformers."""

import logging
from typing import Dict, Tuple

import mlflow
import numpy as np
import torch
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from culture_questions_agent.base_predictor import BasePredictor

logger = logging.getLogger(__name__)


class NLLPredictor(BasePredictor):
    """Predict answer options using Negative Log-Likelihood (NLL) loss."""
    
    def __init__(
        self, 
        model_name: str,
        cache_dir: str = ".cache",
        device: str = "auto"
    ):
        """
        Initialize the NLL-based predictor.
        
        Args:
            model_name: HuggingFace model name
            cache_dir: Cache directory for model files
            device: Device to use ("auto", "cuda", "cpu")
        """
        logger.info(f"Initializing NLL Predictor with model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Loading model to device: {device}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            device_map=device,
            dtype=torch.bfloat16
        )
        self.model.eval()

        # Load Jinja2 prompt templates
        template_dir = Path(__file__).parent.parent.parent / "prompts"
        self.jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))
        self.mcq_yesno_template = self.jinja_env.get_template("mcq_yesno_nll_prompt.jinja")
        
        logger.info("✓ NLL Predictor initialized successfully")
    
    def compute_nll(self, prompt: str) -> float:
        """
        Compute negative log-likelihood for a prompt.
        
        Args:
            prompt: Text prompt to evaluate
            
        Returns:
            NLL loss value
        """
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # Compute loss
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
        
        return loss

    def compute_conditional_nll(self, prompt: str, completion: str) -> float:
        """Compute conditional negative log-likelihood (NLL) of `completion` given `prompt`.

        Only tokens belonging to `completion` contribute to the loss; prompt tokens are
        masked out. This is the preferred scoring method for comparing candidates that
        share the same prompt.

        Args:
            prompt: Prompt text (conditioning context)
            completion: Completion text to score

        Returns:
            Conditional NLL value (mean over completion tokens).
        """
        full_text = prompt + completion

        enc_full = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        enc_prompt = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        input_ids = enc_full["input_ids"].to(self.model.device)
        attention_mask = enc_full.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)

        labels = input_ids.clone()
        prompt_len = enc_prompt["input_ids"].shape[1]

        if prompt_len >= labels.shape[1]:
            logger.warning(
                "Prompt consumed full context window; conditional NLL is undefined. "
                "Returning +inf to avoid selecting this candidate."
            )
            return float("inf")

        labels[:, :prompt_len] = -100

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss.item()

        return loss
    
    def normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """ Each score is log odds of Yes vs No; convert to [0,1] probability scale. """
        # Use log softmax
        log_sum_exp = torch.logsumexp(torch.tensor(list(scores.values())), dim=0).item()
        probs = {k: float(np.exp(v - log_sum_exp)) for k, v in scores.items()}
        return probs

    def choose_between_two_options(
        self,
        question: str,
        option1_key: str,
        option2_key: str,
        options: Dict[str, str],
        option_contexts: Dict[str, str],
        question_contexts: list[str]
    ) -> Tuple[str, Dict[str, float]]:
        """
        Ask the model to choose between two specific options using conditional NLL.
        
        Args:
            question: The question text
            option1_key: First option key
            option2_key: Second option key
            options: Full dictionary of options (for context)
            option_contexts: Dictionary mapping option keys to context texts
            question_contexts: List of context texts for the question
            
        Returns:
            Tuple of (chosen_option_key, dict of NLL scores for the two options)
        """
        logger.info(f"Comparing top 2 options: {option1_key} vs {option2_key}")
        
        # Build context
        context_parts = []
        if question_contexts:
            for i, ctx in enumerate(question_contexts, 1):
                if ctx.strip():
                    context_parts.append(f"[{i}] {ctx.strip()}")
        
        # Add contexts for both options
        for opt_key in [option1_key, option2_key]:
            option_context = option_contexts.get(opt_key, "")
            if option_context.strip():
                context_parts.append(f"[{opt_key}] {option_context.strip()}")
        
        # Build prompt for binary choice
        context_text = "\n".join(context_parts) if context_parts else ""
        
        prompt = f"""Given the following context:
{context_text}

Question: {question}

Choose the best answer from these two options:
{option1_key}. {options[option1_key]}
{option2_key}. {options[option2_key]}

Answer:"""
        
        # Compute conditional NLL for each option key
        nll_scores = {}
        for opt_key in [option1_key, option2_key]:
            # Add space prefix for proper tokenization
            nll = self.compute_conditional_nll(prompt, f" {opt_key}")
            nll_scores[opt_key] = -nll  # Negate so lower NLL = higher score
        
        # Select option with lower NLL (higher score)
        chosen_option = min(nll_scores.items(), key=lambda x: -x[1])[0]
        
        logger.info(f"✓ Final choice: {chosen_option} (NLL scores: {nll_scores})")
        
        return chosen_option, nll_scores

    def predict_best_option(
        self,
        question: str,
        options: Dict[str, str],
        option_contexts: Dict[str, str],
        question_contexts: list[str]
    ) -> Tuple[str, Dict[str, float]]:
        """
        Predict the best option using a two-stage process:
        1. Score all options using Yes/No verification with conditional NLL
        2. Select top 2 options and ask model to choose between them

        Stage 1: For each option, we ask the model:
            Question: ...
            Option: ...
            Is this the correct answer? Yes or No.
            Answer:

        And compute:
            score = NLL("No") - NLL("Yes")

        Stage 2: Present only the top 2 options and compute conditional NLL
        for each option key to make the final choice.
        
        Args:
            question: The question text
            options: Dictionary mapping option keys to option texts
            option_contexts: Dictionary mapping option keys to context texts
            question_contexts: List of context texts for the question
            
        Returns:
            Tuple of (best_option_key, dict of scores from stage 1)
        """
        logger.info(f"Evaluating {len(options)} options using Yes/No NLL scoring...")
        
        scores: Dict[str, float] = {}
        
        with mlflow.start_span(name="predict_best_option") as span:
            span.set_inputs({
                "question": question,
                "options": options,
                "option_contexts": option_contexts,
                "question_contexts": question_contexts,
            })

            # Stage 1: Score all options
            for opt_key, opt_text in options.items():
                # Get option-specific context
                option_context = option_contexts.get(opt_key, "")
                
                # Build formatted context sections
                context_parts = []
                
                # Add question contexts (numbered)
                if question_contexts:
                    for i, ctx in enumerate(question_contexts, 1):
                        if ctx.strip():
                            context_parts.append(f"[{i}] {ctx.strip()}")
                
                # Add option-specific context (if exists)
                if option_context.strip():
                    context_parts.append(f"[Definition] {option_context.strip()}")
                
                # Render prompt from template (stored under ./prompts)
                prompt_prefix = self.mcq_yesno_template.render(
                    context_parts=context_parts,
                    question=question,
                    options=options,
                    proposed_answer_key=opt_key,
                )

                # Prefix with a space so most tokenizers treat it as a new word.
                nll_yes = self.compute_conditional_nll(prompt_prefix, " Yes")
                nll_no = self.compute_conditional_nll(prompt_prefix, " No")

                score = nll_no - nll_yes
                scores[opt_key] = score

            # Normalize scores
            probs = self.normalize_scores(scores) 
            
            # Stage 2: Choose between top 2 options
            if len(options) > 2:
                # Get top 2 scoring options
                sorted_options = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                top1_key, top1_score = sorted_options[0]
                top2_key, top2_score = sorted_options[1]
                
                logger.info(f"Stage 1 top 2: {top1_key} ({top1_score:.4f}), {top2_key} ({top2_score:.4f})")
                
                # Ask model to choose between them
                best_option, final_scores = self.choose_between_two_options(
                    question=question,
                    option1_key=top1_key,
                    option2_key=top2_key,
                    options=options,
                    option_contexts=option_contexts,
                    question_contexts=question_contexts
                )
                
                span.set_outputs({
                    "stage1_probabilities": probs,
                    "stage1_top2": [top1_key, top2_key],
                    "final_choice": best_option,
                    "stage2_scores": final_scores
                })
            else:
                # If 2 or fewer options, just use the best from stage 1
                best_option = max(scores.items(), key=lambda x: x[1])[0]
                span.set_outputs({"option_probabilities": probs})
            
            logger.info(f"✓ Final answer: {best_option}")
        
        return best_option, scores
