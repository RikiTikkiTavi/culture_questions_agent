from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Dict, Dict, List

from tqdm import tqdm

from culture_questions_agent.ingestion.metadata_extractor import MetadataExtractor
from culture_questions_agent.ingestion.wikipedia import Section, WikipediaSectionParser

from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document

from llama_index.readers.wikipedia import WikipediaReader
import pandas as pd
import json

class TrainingDataReader(BasePydanticReader):
    """Reader to ingest training data"""

    def _load_saq(self, saq_path: Path) -> list[Document]:
        """Load SAQ training data from TSV file."""
        df_saq = pd.read_csv(saq_path)
        documents = []
        
        for row in df_saq.itertuples():
            question_en = row.en_question  # type: ignore
            question_local = row.question
            answers_en = []
            answers_local = []
            for annotation in eval(row.annotations): # type: ignore
                answers_en.extend(annotation["en_answers"])
                answers_local.extend(annotation["answers"])
            country = {
                "US": "United States",
                "GB": "United Kingdom",
                "IR": "Iran",
                "CN": "China"
            }.get(row.country, "unknown") # type: ignore
            documents.extend(
                [
                    Document(
                        text=f"Valid answers to '{question_en}' are: {', '.join(answers_en)}",
                        metadata={
                            "question_type": "saq",
                            "country": country,
                            "question": question_en,
                            "answers": answers_en
                        }
                    ),
                    Document(
                        text=f"Valid answers to '{question_local}' are: {', '.join(answers_local)}",
                        metadata={
                            "question_type": "saq",
                            "country": country,
                            "question": question_local,
                            "answers": answers_local
                        }
                    )
                ]
            )
        
        return documents

    def _load_mcq(self, mcq_path: Path) -> list[Document]:
        """Load MCQ training data from TSV file."""
        df_mcq = pd.read_csv(mcq_path)
        documents = []
        
        for row in df_mcq.itertuples():
            question = row.prompt.split("?")[0] + "?"  # type: ignore
            options = json.loads(row.choices)  # type: ignore
            countries = json.loads(row.choice_countries)  # type: ignore            
            answer_idx = row.answer_idx  # type: ignore
            country = countries.get("answer_idx", "unknown")  # type: ignore

            if country == "US":
                country = "United States"
            elif country == "UK":
                country = "United Kingdom"

            documents.append(
                Document(
                    text=f"The correct answer to '{question}' is: {options[answer_idx]}",
                    metadata={
                        "country": country,
                        "options": options,
                        "answer": answer_idx,
                        "question_type": "mcq",
                        "question": question
                    }
                )
            )
        
        return documents

    def lazy_load_data(self, saq_path: Path, mcq_path: Path) -> list[Document]:
        """Load training data from TSV file."""
        return self._load_mcq(mcq_path) + self._load_saq(saq_path)