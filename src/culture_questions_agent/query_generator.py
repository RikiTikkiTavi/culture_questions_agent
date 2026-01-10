"""Query generation module using LLM for semantic search query expansion."""
import logging
import ast
import torch
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)


class QueryGenerator:
    """Generate search queries from questions using LLM."""
    
    def __init__(self, model, tokenizer):
        """
        Initialize query generator with pre-loaded model.
        
        Args:
            model: HuggingFace CausalLM model
            tokenizer: HuggingFace tokenizer
        """
        logger.info("Initializing LLM-based Query Generator")
        self.model = model
        self.tokenizer = tokenizer
        
        # Load Jinja2 templates
        template_dir = Path(__file__).parent.parent.parent / "prompts"
        self.jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))
        self.query_template = self.jinja_env.get_template("query_generation_prompt.jinja")
    
    def _parse_output(self, text: str, original_question: str) -> list[str]:
        """
        Input Text Example:
        "The user is asking about China. \nQueries: ['China clothing', 'Hanfu']"
        """
        logger.debug(f"Parsing generated output: '{text}'")
        try:
            # Step A: Isolate the part after "Queries:"
            if "Queries:" in text:
                list_part = text.split("Queries:")[-1].strip()
            else:
                # Fallback: Just look for the first bracket if the label is missing
                list_part = text

            # Step B: Find the brackets
            start_index = list_part.find("[")
            end_index = list_part.rfind("]") + 1
            
            if start_index == -1 or end_index == 0:
                raise ValueError("No brackets found")

            clean_list_str = list_part[start_index:end_index]

            # Step C: Convert string to List safely
            queries = ast.literal_eval(clean_list_str)
            
            # Validation: Ensure it's actually a list of strings
            if isinstance(queries, list) and all(isinstance(x, str) for x in queries):
                return queries
            
        except (ValueError, SyntaxError) as e:
            print(f"Parsing failed: {e}. Output was: {text}")
        
        # Fail-safe: If anything breaks, search the raw question
        return [original_question]

    def generate_queries(self, question: str, num_queries: int = 3) -> list[str]:
        """
        Generate search queries from a question using LLM.
        
        Args:
            question: Input question
            num_queries: Number of search queries to generate
            
        Returns:
            List of generated search queries
        """
        logger.debug(f"Generating {num_queries} search queries for: '{question}'")
        
        # Render prompt from Jinja2 template
        prompt = self.query_template.render(
            question=question,
            num_queries=num_queries
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # 2. Generate (Stop exactly when the list finishes)
        # We stop at "]" to prevent the model from starting a new example
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=128,      # Give enough space for Reasoning
                stop_strings=["]", "\n\n"], 
                tokenizer=self.tokenizer,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        # 3. Decode output
        # Skip the prompt itself to process only the new text
        full_output = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # 4. Parse the output
        return self._parse_output(full_output, question)
    
    def get_context_string(self, question: str, num_queries: int = 3) -> str:
        """
        Generate queries and join them into a context string.
        
        Args:
            question: Input question
            num_queries: Number of queries to generate
            
        Returns:
            Space-joined query string
        """
        queries = self.generate_queries(question, num_queries=num_queries)
        context_str = " ".join(queries)
        logger.info(f"Context string: '{context_str}'")
        return context_str
    
    def generate_option_query(self, option_text: str) -> str:
        """
        Generate a verification query for an answer option.
        
        Args:
            option_text: The option text to verify
            
        Returns:
            Search query string
        """
        # Strict validation query: Define it and find its origin
        query = f"{option_text}"
        logger.debug(f"Option query: '{query}'")
        return query
    
    def is_generic_option(self, option_text: str) -> bool:
        """
        Check if an option is generic and shouldn't be searched.
        
        Args:
            option_text: Option text to check
            
        Returns:
            True if generic, False otherwise
        """
        bad_phrases = [
            "none of the above", "all of the above", "as usual", 
            "not mentioned", "unknown", "none", "various",
            "it varies", "depends", "n/a", "not applicable"
        ]
        text_lower = option_text.lower().strip()
        return any(phrase in text_lower for phrase in bad_phrases)

