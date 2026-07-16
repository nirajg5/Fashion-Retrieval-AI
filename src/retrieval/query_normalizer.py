from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

import torch

from config import DEVICE


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


class QueryNormalizer:

    def __init__(self):

        print("Loading Query Normalizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        ).to(DEVICE)

        self.model.eval()

        print("✓ Query Normalizer Loaded")

    def normalize(self, query):

        prompt = f"""
Rewrite the following fashion search query for semantic retrieval.

Rules:
- Keep clothing type.
- Keep colors.
- Keep gender if mentioned.
- Keep accessories.
- Keep scene or environment.
- Remove unnecessary words.
- Return only the rewritten query.

Query:
{query}
"""

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()


query_normalizer = QueryNormalizer()