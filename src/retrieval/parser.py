"""
Query Parser

Lightweight preprocessing before embedding generation.
"""

import re


class QueryParser:

    def __init__(self):
        pass

    def clean(self, text: str) -> str:

        text = text.lower().strip()

        text = re.sub(r"\s+", " ", text)

        return text

    def parse(self, query):

        query = query.lower().strip()

        query = " ".join(query.split())

        return {
            "original": query,
            "cleaned": query
        }


parser = QueryParser()