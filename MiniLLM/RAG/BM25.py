"""Okapi BM25 from scratch, in pure Python: ranks passages by how well their words match a query.

score(passage, query) = sum over query terms t of
    idf(t) * f(t) * (k1 + 1) / (f(t) + k1 * (1 - b + b * length / average_length))
where f(t) is how often t occurs in the passage and idf(t) = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5)),
with N passages of which n(t) contain t. k1 caps the reward for repeating a word; b penalises long passages.
"""

from __future__ import annotations

import heapq
import math
import re
from collections import Counter
from typing import Sequence

WORD_PATTERN = re.compile(r"\w+")
STOPWORDS = frozenset("""
    a about after all also an and any are as at be been but by can could did do does for from had has have he her
    his how i if in into is it its may me more most my no not of on one or other our out she so some such than that
    the their them then there these they this those to up us was we were what when where which while who whom whose
    why will with would you your
""".split())


def normalize_word(word: str) -> str:
    """Tiny plural folding so "planets" matches "planet"; a full stemmer is overkill for passage ranking."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    return [normalize_word(word) for word in WORD_PATTERN.findall(text.lower()) if word not in STOPWORDS]


class BM25:
    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.document_count = len(documents)
        self.document_lengths: list[int] = []
        # Inverted index: term -> [(document index, term count)], so a query only visits matching documents.
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for document_index, document in enumerate(documents):
            term_counts = Counter(tokenize(document))
            self.document_lengths.append(sum(term_counts.values()))
            for term, count in term_counts.items():
                self.postings.setdefault(term, []).append((document_index, count))
        self.average_length = sum(self.document_lengths) / self.document_count if self.document_count else 0.0

    def idf(self, term: str) -> float:
        containing = len(self.postings.get(term, ()))
        return math.log(1 + (self.document_count - containing + 0.5) / (containing + 0.5))

    def scores(self, query: str) -> list[float]:
        scores = [0.0] * self.document_count
        for term in set(tokenize(query)):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for document_index, count in postings:
                length_ratio = self.document_lengths[document_index] / max(self.average_length, 1e-9)
                saturation = count + self.k1 * (1 - self.b + self.b * length_ratio)
                scores[document_index] += idf * count * (self.k1 + 1) / saturation
        return scores

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        """(document index, score) for the k best documents that share at least one word with the query."""
        scored = ((score, -index) for index, score in enumerate(self.scores(query)) if score > 0)
        return [(-negative_index, score) for score, negative_index in heapq.nlargest(k, scored)]
