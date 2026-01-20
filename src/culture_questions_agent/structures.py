from dataclasses import dataclass

from pydantic import BaseModel

@dataclass
class MCQQuestion:
    msq_id: str
    question: str
    country: str
    options: dict[str, str]


@dataclass
class MCQQuestionTrain(MCQQuestion):
    answer: str

@dataclass
class SAQQuestion:
    saq_id: str
    pos_id: int
    question: str
    country: str

@dataclass
class SAQQuestionTrain(SAQQuestion):
    answers: list[str]