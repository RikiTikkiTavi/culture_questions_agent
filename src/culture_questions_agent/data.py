from pathlib import Path
import json
import pandas as pd

from culture_questions_agent.structures import MCQQuestionTrain, MCQQuestion, SAQQuestionTrain, SAQQuestion


def read_mcq_data_train(file_path: Path) -> list[MCQQuestionTrain]:
    """Reads the multiple-choice question dataset from a CSV file."""
    df = pd.read_csv(file_path)
    r = []
    for row in df.itertuples():
        question = row.prompt.split("?")[0] + "?"  # type: ignore
        options = json.loads(row.choices)  # type: ignore
        countries = json.loads(row.choice_countries)  # type: ignore
        answer = row.answer_idx
        msq_id = row.MCQID  # type: ignore
        country = extract_country_from_row(row)
        r.append(MCQQuestionTrain(msq_id=msq_id, question=question, options=options, country=country, answer=answer)) # type: ignore
    return r

def extract_country_from_row(row) -> str:
    country = row.country
    if country == "US":
        country = "United States"
    elif country == "UK":
        country = "United Kingdom"
    return country

def read_mcq_data_test(file_path: Path) -> list[MCQQuestion]:
    """Reads the multiple-choice question dataset (without labels) from a CSV file."""
    df = pd.read_csv(file_path)
    r = []
    for row in df.itertuples():
        question = row.prompt.split("?")[0] + "?"  # type: ignore
        country = extract_country_from_row(row)
        options = json.loads(row.choices)  # type: ignore
        msq_id = row.MCQID  # type: ignore
        r.append(MCQQuestion(msq_id=msq_id, question=question, options=options, country=country)) # type: ignore
    return r


def read_saq_data_train(file_path: Path) -> list[SAQQuestionTrain]:
    """Reads the short-answer question dataset from a CSV file."""
    df = pd.read_csv(file_path)
    r = []
    for row in df.itertuples():
        question = row.en_question  # type: ignore
        saq_id = row.ID  # type: ignore
        pos_id = row.Index  # type: ignore
        country = extract_country_from_row(row)
        answers = []
        for annotation in eval(row.annotations): # type: ignore
            answers.extend(annotation["en_answers"])
        r.append(SAQQuestionTrain(saq_id=saq_id, pos_id=pos_id, question=question, answers=list(set(answers)), country=country)) # type: ignore
    return r


def read_saq_data_test(file_path: Path) -> list[SAQQuestion]:
    """Reads the short-answer question dataset (without labels) from a CSV file."""
    df = pd.read_csv(file_path)
    r = []
    for row in df.itertuples():
        question = row.en_question  # type: ignore
        saq_id = row.ID  # type: ignore
        pos_id = row.Index  # type: ignore
        country = extract_country_from_row(row)
        r.append(SAQQuestion(saq_id=saq_id, pos_id=pos_id, question=question, country=country)) # type: ignore
    return r

