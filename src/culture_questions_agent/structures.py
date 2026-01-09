from dataclasses import dataclass

from pydantic import BaseModel

@dataclass
class MCQQuestion:
    question: str
    options: dict[str, str]


@dataclass
class MCQQuestionTrain:
    question: str
    options: dict[str, str]
    countries: dict[str, str]
    answer: str

@dataclass
class SAQQuestion:
    question: str

@dataclass
class SAQQuestionTrain:
    question: str
    answer: str