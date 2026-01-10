"""Factory for creating predictor instances based on configuration."""

import logging
from typing import Optional

from culture_questions_agent.base_predictor import BasePredictor
from culture_questions_agent.nll_predictor import NLLPredictor
from culture_questions_agent.generative_predictor import GenerativePredictor

logger = logging.getLogger(__name__)


class PredictorFactory:
    """Factory for creating predictor instances."""
    
    @staticmethod
    def create_predictor(
        predictor_type: str,
        model_name: str,
        cache_dir: str = ".cache",
        device: str = "auto",
        max_new_tokens: int = 10,
        temperature: float = 0.1,
        model: Optional[object] = None,
        tokenizer: Optional[object] = None
    ) -> BasePredictor:
        """
        Create a predictor instance based on type.
        
        Args:
            predictor_type: Type of predictor ("discriminative" or "generative")
            model_name: HuggingFace model name
            cache_dir: Cache directory for model files
            device: Device to use ("auto", "cuda", "cpu")
            max_new_tokens: Maximum tokens to generate (for generative)
            temperature: Sampling temperature (for generative)
            model: Pre-loaded model (optional, for reuse)
            tokenizer: Pre-loaded tokenizer (optional, for reuse)
            
        Returns:
            Predictor instance
        """
        predictor_type = predictor_type.lower()
        
        if predictor_type == "discriminative":
            logger.info("Creating discriminative (NLL) predictor")
            return NLLPredictor(
                model_name=model_name,
                cache_dir=cache_dir,
                device=device
            )
        elif predictor_type == "generative":
            logger.info("Creating generative predictor")
            return GenerativePredictor(
                model_name=model_name,
                cache_dir=cache_dir,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
        else:
            raise ValueError(
                f"Unknown predictor type: {predictor_type}. "
                f"Supported types: 'discriminative', 'generative'"
            )
