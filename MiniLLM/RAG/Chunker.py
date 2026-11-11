"""Split Wikipedia pages into small overlapping passages, measured in tokens of the project tokenizer.

Pages are first cut at their `== Section ==` headings (reference-style sections are dropped), then each section
is split at the coarsest boundary that fits: paragraphs, lines, sentences, words, and finally characters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from MiniLLM.RAG.WikiSource import WikiPage

DEFAULT_PASSAGE_TOKENS = 160
DEFAULT_OVERLAP_TOKENS = 32
SECTION_HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)
# Matched as substrings, so "General and cited references" or "Notes and sources" are dropped too.
SKIPPED_SECTION_WORDS = ("see also", "reference", "external link", "notes", "further reading", "bibliography",
                         "sources", "citation", "footnote", "works cited")
SPLIT_SEPARATORS = ["\n\n", "\n", ". ", " "]

TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class Passage:
    title: str
    section: str
    url: str
    text: str

    @property
    def label(self) -> str:
        return f"{self.title} - {self.section}" if self.section else self.title


def token_counter(tokenizer) -> TokenCounter:
    # verbose=False: whole sections are counted before splitting and may exceed the model's context.
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False, verbose=False))


def split_sections(text: str) -> list[tuple[str, str]]:
    """[(heading, body)], the lead section having heading "". Bodies of skipped sections are dropped."""
    sections: list[tuple[str, str]] = []
    heading, body_start = "", 0
    for match in SECTION_HEADING.finditer(text):
        sections.append((heading, text[body_start:match.start()]))
        heading, body_start = match.group(2), match.end()
    sections.append((heading, text[body_start:]))
    return [(heading, body.strip()) for heading, body in sections
            if body.strip() and not any(word in heading.lower() for word in SKIPPED_SECTION_WORDS)]


def _split_keeping_separator(text: str, separator: str) -> list[str]:
    parts = text.split(separator)
    return [part + separator for part in parts[:-1]] + [parts[-1]]


def _atomic_pieces(text: str, count_tokens: TokenCounter, max_tokens: int, separators: list[str]) -> list[str]:
    """Pieces of at most max_tokens that keep their separators, so "".join(pieces) == text."""
    token_count = count_tokens(text)
    if token_count <= max_tokens:
        return [text]
    if not separators:
        characters_per_piece = max(1, len(text) * max_tokens // token_count)
        return [text[start:start + characters_per_piece] for start in range(0, len(text), characters_per_piece)]
    separator, *finer_separators = separators
    pieces = []
    for part in _split_keeping_separator(text, separator):
        if part:
            pieces.extend(_atomic_pieces(part, count_tokens, max_tokens, finer_separators))
    return pieces


def split_text(text: str, count_tokens: TokenCounter, max_tokens: int = DEFAULT_PASSAGE_TOKENS,
               overlap_tokens: int = DEFAULT_OVERLAP_TOKENS) -> list[str]:
    """Greedily pack pieces into passages of at most ~max_tokens. Each passage starts with up to overlap_tokens
    from the end of the previous one, so a fact that straddles a boundary is still whole in one passage."""
    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError("need 0 <= overlap_tokens < max_tokens")
    pieces = _atomic_pieces(text, count_tokens, max_tokens, SPLIT_SEPARATORS)
    piece_tokens = [count_tokens(piece) for piece in pieces]

    passages: list[str] = []
    window: list[int] = []
    window_tokens = 0
    for index, tokens in enumerate(piece_tokens):
        if window and window_tokens + tokens > max_tokens:
            passages.append("".join(pieces[i] for i in window).strip())
            while window and (window_tokens > overlap_tokens or window_tokens + tokens > max_tokens):
                window_tokens -= piece_tokens[window.pop(0)]
        window.append(index)
        window_tokens += tokens
    if window:
        passages.append("".join(pieces[i] for i in window).strip())
    return [passage for passage in passages if passage]


def chunk_page(page: WikiPage, count_tokens: TokenCounter, max_tokens: int = DEFAULT_PASSAGE_TOKENS,
               overlap_tokens: int = DEFAULT_OVERLAP_TOKENS) -> list[Passage]:
    return [Passage(page.title, heading, page.url, text)
            for heading, body in split_sections(page.text)
            for text in split_text(body, count_tokens, max_tokens, overlap_tokens)]
