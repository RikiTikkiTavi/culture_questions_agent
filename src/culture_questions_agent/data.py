from pathlib import Path
import json
import pandas as pd

from culture_questions_agent.structures import MCQQuestion


def read_mcq_data(file_path: Path) -> list[MCQQuestion]:
    """Reads the multiple-choice question dataset from a CSV file."""
    df = pd.read_csv(file_path)
    r = []
    for row in df.itertuples():
        question = row.prompt.split("?")[0] + "?"  # type: ignore
        options = json.loads(row.choices)  # type: ignore
        answer = row.answer_idx
        r.append(MCQQuestion(question=question, options=options, answer=answer)) # type: ignore
    return r
