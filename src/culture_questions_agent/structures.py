from dataclasses import dataclass

from pydantic import BaseModel

@dataclass
class MCQQuestion:
    msq_id: str
    question: str
    options: dict[str, str]


@dataclass
class MCQQuestionTrain(MCQQuestion):
    countries: dict[str, str]
    answer: str

@dataclass
class SAQQuestion:
    question: str

@dataclass
class SAQQuestionTrain:
    question: str
    answers: list[str]