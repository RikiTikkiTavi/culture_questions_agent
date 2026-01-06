"""NLL-based LLM predictor using HuggingFace transformers."""
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class NLLPredictor:
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
    
    def predict_best_option(
        self,
        question: str,
        options: Dict[str, str],
        option_contexts: Dict[str, str],
        question_contexts: list[str]
    ) -> Tuple[str, Dict[str, float]]:
        """
        Predict the best option using NLL loss.
        
        Args:
            question: The question text
            options: Dictionary mapping option keys to option texts
            option_contexts: Dictionary mapping option keys to context texts
            question_contexts: List of context texts for the question
            
        Returns:
            Tuple of (best_option_key, dict of losses)
        """
        logger.info(f"Evaluating {len(options)} options using NLL...")
        
        losses = {}
        
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
            
            # Build prompt
            if context_parts:
                combined_context = "\n".join(context_parts)
                prompt = f"Context:\n{combined_context}\n\nQuestion: {question}\nAnswer: {opt_text}"
            else:
                prompt = f"Question: {question}\nAnswer: {opt_text}"
            
            # Compute NLL
            loss = self.compute_nll(prompt)
            losses[opt_key] = loss
            
            logger.info(f"  Option {opt_key} ({opt_text[:30]}...): NLL = {loss:.4f}")
        
        # Select option with minimal loss
        best_option = min(losses.items(), key=lambda x: x[1])[0]
        
        logger.info(f"✓ Selected option {best_option} with minimal NLL = {losses[best_option]:.4f}")
        
        return best_option, losses
