"""Generative LLM predictor using standard text generation."""

import logging
from typing import Dict, Tuple

import mlflow
import torch
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, StopStringCriteria

from culture_questions_agent.base_predictor import BasePredictor

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
        )
        self.model.eval()
        
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Load Jinja2 prompt template
        template_dir = Path(__file__).parent.parent.parent / "prompts"
        self.jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))
        self.mcq_template = self.jinja_env.get_template("mcq_prompt.jinja")
        
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
            max_length=2048
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stop_strings=["\nQuestion:"],
                tokenizer=self.tokenizer,
            )
        
        # Decode only the generated part (skip the input prompt)
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_text = generated_text.replace("\nQuestion:", "")

        return generated_text.strip()
    
    def predict_best_option(
        self,
        question: str,
        options: Dict[str, str],
        option_contexts: Dict[str, str],
        question_contexts: list[str]
    ) -> Tuple[str, Dict[str, float]]:
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
                
                # Generate answer
                generated = self.generate_answer(prompt)

                gen_span.set_outputs({"generated_text": generated})

            
            # Parse the generated answer to extract option key
            # Look for single letter A/B/C/D in the response
            generated_upper = generated.upper()
            best_option = None
            
            # First try to find exact option key
            for opt_key in options.keys():
                if opt_key in generated_upper:
                    best_option = opt_key
                    break
            
            # If not found, default to first option
            if best_option is None:
                logger.warning(f"Could not parse option from generated text: '{generated}', defaulting to first option")
                best_option = list(options.keys())[0]
            
            # Create dummy scores (1.0 for selected, 0.0 for others)
            scores = {opt_key: 1.0 if opt_key == best_option else 0.0 for opt_key in options.keys()}
            
            span.set_outputs({"predicted_option": best_option, "generated_text": generated})
            
            logger.info(f"✓ Selected option {best_option}")
        
        return best_option, scores
