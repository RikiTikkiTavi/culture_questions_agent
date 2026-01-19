from abc import ABC, abstractmethod

class BasePredictor(ABC):
    
    @abstractmethod
    def predict_best_option(
        self,
        question: str,
        options: dict[str, str],
        option_contexts: dict[str, str],
        question_contexts: list[str]
    ) -> str: ...

    @abstractmethod
    def predict_short_answer(
        self,
        question: str,
        question_contexts: list[str]
    ) -> str: ...