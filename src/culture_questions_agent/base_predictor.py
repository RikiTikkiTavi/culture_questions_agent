"""Base predictor interface for cultural QA answer prediction strategies."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple


class BasePredictor(ABC):
    """Abstract base class for answer prediction strategies."""
    
    @abstractmethod
    def predict_best_option(
        self,
        question: str,
        options: Dict[str, str],
        option_contexts: Dict[str, str],
        question_contexts: list[str]
    ) -> Tuple[str, Dict[str, float]]:
        """
        Predict the best option given question and contexts.
        
        Args:
            question: The question text
            options: Dictionary mapping option keys to option texts
            option_contexts: Dictionary mapping option keys to context texts
            question_contexts: List of context texts for the question
            
        Returns:
            Tuple of (best_option_key, dict of scores/probabilities)
        """
        pass
