import asyncio
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import logging
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterable, Iterator, List, Optional
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from culture_questions_agent.ingestion.questions import TrainingDataReader
from culture_questions_agent.query_generator import QueryGenerator

from ddgs.ddgs import DDGS
from llama_index.readers.web import TrafilaturaWebReader

import aiostream

logger = logging.getLogger(__name__)


class WebSearchReader(BasePydanticReader):
    """Reader that performs web searches based on training questions and loads web documents."""
    
    is_remote: bool = True
    llm_name: str
    cache_dir: Optional[str] = None
    saq_path: str = "data/saq_training_data.tsv"
    mcq_path: str = "data/mcq_training_data.tsv"
    ddgs_backend: str = "yandex,yahoo,wikipedia,grokipedia"
    num_queries: int = 3
    max_results_per_query: int = 3
    max_workers: int = 10
    limit_questions_type: Optional[str] = "mcq"

    def _initialize(self) -> None:
        self._loop = asyncio.get_event_loop()
        logger.info(f"Loading model {self.llm_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.llm_name,
            cache_dir=self.cache_dir
        )    
        self._model = AutoModelForCausalLM.from_pretrained(
            self.llm_name,
            cache_dir=self.cache_dir,
            device_map="auto",
        )
        self._model.eval()
        self._generator = QueryGenerator(
            model=self._model,
            tokenizer=self._tokenizer
        )
        self._ddgs = DDGS()
        self._web_reader = TrafilaturaWebReader()
        self._semaphore = asyncio.Semaphore(self.max_workers)
    
    def _get_regions_for_country(self, country: str) -> List[str]:
        """Get search regions based on country."""
        # country_region_map = {
        #     "United States": ["us-en"],
        #     "unknown": ["us-en"],
        #     "China": ["cn-zh", "cn-en"],
        #     "Iran": ["ir-fa", "ir-en"],
        #     "United Kingdom": ["gb-en"],
        # }
        # return country_region_map.get(country, ["us-en"])
        return ["us-en"]

    def _generate_queries_sync(self, question_text: str) -> List[str]:
        """Generate search queries for a question (CPU-bound operation)."""
        queries = self._generator.generate_queries(question_text, num_queries=self.num_queries)
        # Use first generated query and the original question
        return [queries[0], question_text]
    
    async def _perform_single_search_and_load(self, query: str, region: str) -> List[Document]:
        """Perform a single web search (IO-bound operation)."""
        
        # Wait on semaphore to limit concurrency
        async with self._semaphore:
            try:
                web_results = await asyncio.to_thread(
                    self._ddgs.text,
                    query=query,
                    max_results=self.max_results_per_query, 
                    region=region, 
                    backend=self.ddgs_backend
                )
                docs = await self._web_reader.aload_data(
                    urls=[res["href"] for res in web_results],
                    include_comments=False,
                )
                for doc in docs:
                    doc.metadata.update({
                        "source": "web_search", 
                        "query": query,
                    })
                return docs
            except Exception as e:
                logger.warning(
                    "Failed to perform web search for query '%s' in region '%s': %s", 
                    query, region, str(e)
                )
                return []
    
    async def _perform_web_searches(self, questions: Iterable[Document]) -> AsyncIterator[Document]:
        """Generate web search results one at a time from questions."""

        tasks = []

        pbar = tqdm(desc="Performing web searches", total=len(list(questions)))

        for question in questions:
            country = question.metadata.get("country", "unknown")
            question_text = question.metadata.get("question", "")
            
            # Generate queries (CPU-bound)
            queries = [question_text]
            regions = self._get_regions_for_country(country)


            for region in regions:
                for query in queries:
                    task = asyncio.create_task(
                        self._perform_single_search_and_load(query, region)
                    )
                    tasks.append(task)
            
            for task in asyncio.as_completed(tasks):
                docs = await task
                for doc in docs:
                    pbar.update(1)
                    yield doc


    def lazy_load_data(self, **kwargs) -> Iterable[Document]:
        """
        Load web documents based on training data questions.

        Args:
            **kwargs: Additional arguments (unused, for compatibility)
            
        Returns:
            Document objects from web searches
        """
        if getattr(self, "_executor", None) is None:
            self._initialize()

        # Load training questions (generator)
        reader = TrainingDataReader()
        questions = reader.lazy_load_data(
            saq_path=Path(self.saq_path),
            mcq_path=Path(self.mcq_path)
        )
        if self.limit_questions_type:
            questions = list(
                q for q in questions 
                if q.metadata.get("question_type") == self.limit_questions_type
            )
        
        logger.info(f"Loaded {len(list(questions))} training questions for web search")

        search_results_generator = self._perform_web_searches(questions)
        
        async def collect(gen):
            return [doc async for doc in gen]

        docs = self._loop.run_until_complete(collect(search_results_generator))
        return docs


        
   