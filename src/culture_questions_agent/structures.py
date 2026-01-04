from dataclasses import dataclass


@dataclass
class MCQQuestion:
    question: str
    options: dict[str, str]
    answer: str