"""Generative LLM predictor using standard text generation."""

import logging
import re
from typing import Dict, Tuple

import mlflow
import torch
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, StopStringCriteria, AutoModel

from culture_questions_agent.predictor.base import BasePredictor

logger = logging.getLogger(__name__)


class GenerativePredictor(BasePredictor):
    """Predict answer options using standard text generation."""
    
    def __init__(
        self, 
        model_name: str,
        cache_dir: str = ".cache",
        device: str = "auto",
        max_new_tokens: int = 10,
        temperature: float = 0.0
    ):
        """
        Initialize the generative predictor.
        
        Args:
            model_name: HuggingFace model name
            cache_dir: Cache directory for model files
            device: Device to use ("auto", "cuda", "cpu")
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        logger.info(f"Initializing Generative Predictor with model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Loading model to device: {device}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            device_map=device,
            dtype=torch.bfloat16
        ) # type: ignore
        self.model.eval()
        
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Load Jinja2 prompt template
        template_dir = Path(__file__).parent.parent.parent.parent / "prompts"
        self.jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))
        self.mcq_template = self.jinja_env.get_template("mcq_prompt.jinja")
        self.saq_template = self.jinja_env.get_template("saq_prompt.jinja")

        logger.info("✓ Generative Predictor initialized successfully")
    
    def generate_answer(self, prompt: str) -> str:
        """
        Generate answer for a prompt.
        
        Args:
            prompt: Text prompt
            
        Returns:
            Generated text
        """
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        ).to(self.model.device) # type: ignore
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate( # type: ignore
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id, # type: ignore
                eos_token_id=self.tokenizer.eos_token_id, # type: ignore
                stop_strings=["\nQuestion:"],
                tokenizer=self.tokenizer,
            )
        
        # Decode only the generated part (skip the input prompt)
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True) # type: ignore
        generated_text = generated_text.replace("\nQuestion:", "")

        return generated_text.strip()
    
    def get_option_token_ids(self, letter) -> list[int]:
        variants = [
            letter,          # "A"
            f" {letter}",    # " A"
            f"\n{letter}",   # "\nA"
        ]

        token_ids = []
        for v in variants:
            ids = self.tokenizer.encode(v, add_special_tokens=False)
            assert len(ids) == 1, f"Option is not a single token: {v}"
            token_ids.append(ids[0])

        assert len(token_ids) > 0, "No token IDs found for option"
        return token_ids

    def predict_best_option(
        self,
        question: str,
        options: Dict[str, str],
        option_contexts: Dict[str, str],
        question_contexts: list[str]
    ) -> str:
        """
        Predict the best option using standard text generation.
        
        The model is prompted to directly answer with the option key (A/B/C/D).
        
        Args:
            question: The question text
            options: Dictionary mapping option keys to option texts
            option_contexts: Dictionary mapping option keys to context texts
            question_contexts: List of context texts for the question
            
        Returns:
            Tuple of (best_option_key, dict of scores)
        """
        logger.info(f"Evaluating {len(options)} options using generative prediction...")
        
        with mlflow.start_span(name="predict_best_option") as span:
            span.set_inputs({
                "question": question,
                "options": options,
                "option_contexts": option_contexts,
                "question_contexts": question_contexts,
            })
            
            # Build formatted context sections
            context_parts = []
            
            # Add question contexts (numbered)
            if question_contexts:
                for i, ctx in enumerate(question_contexts, 1):
                    if ctx.strip():
                        context_parts.append(f"[{i}] {ctx.strip()}")
            
            # Add option-specific contexts (if exist)
            for opt_key in sorted(options.keys()):
                option_context = option_contexts.get(opt_key, "")
                if option_context.strip():
                    context_parts.append(f"[{opt_key} - Definition] {option_context.strip()}")
            
            with mlflow.start_span(name="generate_answer", span_type="LLM") as gen_span:
                # Render prompt from template
                prompt = self.mcq_template.render(
                    context_parts=context_parts,
                    question=question,
                    options=options,
                )
                gen_span.set_inputs({"prompt": prompt})
                
                # Tokenize
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=16384
                ).to(self.model.device) # type: ignore
                
                # Forward pass
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits  # [1, seq_len, vocab]

                # Get next-token logits
                next_token_logits = logits[0, -1]

                # Force leading-space options (LLaMA-3 correct)
                option_scores = {}
                for opt_key in options.keys():
                    token_ids = self.get_option_token_ids(opt_key)
                    option_scores[opt_key] = torch.logsumexp(
                        torch.tensor([next_token_logits[i] for i in token_ids]),
                        dim=0
                    ).item()

                # Pick highest logit
                best_option = max(
                    option_scores.keys(),
                    key=lambda opt: option_scores[opt]
                )

                span.set_outputs({"predicted_option": best_option})
            
            logger.info(f"✓ Selected option {best_option}")
        
        return best_option

    def predict_short_answer(
        self,
        question: str,
        question_contexts: list[str]
    ) -> str:
        """
        Predict short answer using standard text generation.
        
        Args:
            question: The question text
            question_contexts: List of context texts for the question
            
        Returns:
            Generated short answer
        """
        logger.info("Generating short answer using generative prediction...")
        
        with mlflow.start_span(name="predict_short_answer") as span:
            span.set_inputs({
                "question": question,
                "question_contexts": question_contexts,
            })
            
            # Build formatted context sections
            context_parts = []
            
            # Add question contexts (numbered)
            if question_contexts:
                for i, ctx in enumerate(question_contexts, 1):
                    if ctx.strip():
                        context_parts.append(f"[{i}] {ctx.strip()}")
            
            with mlflow.start_span(name="generate_answer", span_type="LLM") as gen_span:
                # Render prompt from template
                prompt = self.saq_template.render(
                    context_parts=context_parts,
                    question=question,
                )
                gen_span.set_inputs({"prompt": prompt})
                
                # Generate answer
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=16384,
                ).to(self.model.device) # type: ignore
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate( # type: ignore
                        **inputs,
                        max_new_tokens=10,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id, # type: ignore
                        eos_token_id=self.tokenizer.eos_token_id, # type: ignore
                        tokenizer=self.tokenizer,
                    )
                
                # Decode only the generated part (skip the input prompt)
                generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True) # type: ignore
                # Find the text in "" using regex
                
                match = re.search(r'"(.*?)"', generated_text)
                if match:
                    generated_text = match.group(1)
                else:
                    generated_text = generated_text.strip()
                
                generated_text = generated_text.replace('"', '').replace("“", "").replace("”", "").replace("'", "")

                gen_span.set_outputs({"generated_answer": generated_text})
        
        return generated_text