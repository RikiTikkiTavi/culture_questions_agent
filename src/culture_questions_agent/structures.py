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
    saq_id: str
    pos_id: int
    question: str

@dataclass
class SAQQuestionTrain(SAQQuestion):
    answers: list[str]