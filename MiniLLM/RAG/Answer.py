"""Answer questions with MiniLLM, grounded in Wikipedia passages (retrieval-augmented generation).

    python -m MiniLLM.RAG.Answer "What is the capital of Australia?"
    python -m MiniLLM.RAG.Answer --compare "Who wrote Pride and Prejudice?"     # with vs without retrieval
    python -m MiniLLM.RAG.Answer --show-passages --top-k 2 "Who created Python?"
    python -m MiniLLM.RAG.Answer --offline "..."                                 # cached pages only
    python -m MiniLLM.RAG.Answer --model checkpoints/sft/latest "..."            # default: newest local checkpoint

Steps: Wikipedia search -> fetch + cache pages -> passages (project tokenizer) -> BM25 top 2-3 -> prompt that fits
the context budget (1024 tokens, answer included) -> generate_text. SFT/DPO checkpoints get the ChatML prompt;
base checkpoints get a plain "Question: ... Answer:" prompt, since they never saw ChatML.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch

from MiniLLM.Evaluation.Common import HF_PREFIX, load_eval_model, load_eval_tokenizer, pick_device
from MiniLLM.Evaluation.TextGeneration import format_prompt, generate_text
from MiniLLM.RAG.BM25 import BM25
from MiniLLM.RAG.Chunker import (
    DEFAULT_OVERLAP_TOKENS,
    DEFAULT_PASSAGE_TOKENS,
    Passage,
    chunk_page,
    token_counter,
)
from MiniLLM.RAG.WikiSource import DEFAULT_CACHE_DIR, WikiPage, WikiSource
from MiniLLM.Training.TrainingUtils import resolve_checkpoint

DEFAULT_TOKEN_BUDGET = 1024
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_TOP_K = 3
DEFAULT_PAGES = 4
DEFAULT_REPETITION_PENALTY = 1.15
DEFAULT_MODEL_CANDIDATES = ("checkpoints/sft/latest", "checkpoints_local/sft/latest",
                            "checkpoints/pretrain/latest", "checkpoints_local/pretrain/latest")
CHAT_MODEL_MARKERS = ("sft", "dpo", "instruct", "chat")
INSTRUCTION = ("Answer the question using the Wikipedia passages. Reply in one or two sentences. "
               "If the passages do not contain the answer, say you don't know.")
# Base models keep going after the answer (often with a new "Question:"); the answer is one line.
BASE_STOP_STRINGS = ["\n"]
# At most this many passages per article, so one long off-topic article cannot fill every slot.
MAX_PASSAGES_PER_PAGE = 2
# Wikipedia leads summarise an article's key facts (who, what, where, when), so lead passages get a bonus over
# body passages that merely repeat the query words, e.g. the "Pride and Prejudice" lead naming Jane Austen.
LEAD_SECTION_BOOST = 1.5


@dataclass
class RAGAnswer:
    question: str
    answer: str
    passages: list[Passage]               # passages that fit into the prompt
    prompt_tokens: int
    retrieved: list[Passage] = field(default_factory=list)  # BM25 top-k before the budget cut


def format_passages(passages: list[Passage]) -> str:
    return "\n".join(f"[{number}] {passage.label}: {' '.join(passage.text.split())}"
                     for number, passage in enumerate(passages, start=1))


def build_prompt(question: str, passages: list[Passage], chat: bool) -> str:
    """The user turn (chat) or the whole prompt (base). generate_text wraps the chat turn in ChatML."""
    if chat:
        if not passages:
            return question
        return f"{INSTRUCTION}\n\nPassages:\n{format_passages(passages)}\n\nQuestion: {question}"
    if not passages:
        return f"Question: {question}\nAnswer:"
    return f"Wikipedia passages:\n{format_passages(passages)}\n\nQuestion: {question}\nAnswer:"


def prompt_token_count(tokenizer, prompt: str, chat: bool) -> int:
    return len(tokenizer.encode(format_prompt(tokenizer, prompt, chat), add_special_tokens=False))


def fit_prompt(question: str, passages: list[Passage], tokenizer, prompt_budget: int,
               chat: bool) -> tuple[str, list[Passage], int]:
    """Largest prompt within prompt_budget tokens: drop the lowest-ranked passages first, then trim the best one.

    Returns (prompt, passages used, prompt tokens). Without room for any passage it falls back to no retrieval.
    """
    for count in range(len(passages), 0, -1):
        prompt = build_prompt(question, passages[:count], chat)
        tokens = prompt_token_count(tokenizer, prompt, chat)
        if tokens <= prompt_budget:
            return prompt, passages[:count], tokens

    if passages:
        best = passages[0]
        overhead = prompt_token_count(tokenizer, build_prompt(question, [replace(best, text="")], chat), chat)
        passage_ids = tokenizer.encode(best.text, add_special_tokens=False, verbose=False)
        room = prompt_budget - overhead
        # decode -> encode is not exactly length-preserving, so shrink until the rebuilt prompt really fits.
        while room > 0:
            trimmed = replace(best, text=tokenizer.decode(passage_ids[:room]).strip())
            prompt = build_prompt(question, [trimmed], chat)
            tokens = prompt_token_count(tokenizer, prompt, chat)
            if tokens <= prompt_budget:
                return prompt, [trimmed], tokens
            room -= max(1, tokens - prompt_budget)

    prompt = build_prompt(question, [], chat)
    return prompt, [], prompt_token_count(tokenizer, prompt, chat)


class WikiRAG:
    def __init__(self, model, tokenizer, device: torch.device, wiki: WikiSource, chat: bool,
                 top_k: int = DEFAULT_TOP_K, pages: int = DEFAULT_PAGES, token_budget: int = DEFAULT_TOKEN_BUDGET,
                 max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, passage_tokens: int = DEFAULT_PASSAGE_TOKENS,
                 overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
                 repetition_penalty: float = DEFAULT_REPETITION_PENALTY):
        self.model, self.tokenizer, self.device, self.wiki, self.chat = model, tokenizer, device, wiki, chat
        self.top_k, self.pages = top_k, pages
        context = model.config.max_seq_len
        self.token_budget = min(token_budget, context)
        # generate_texts caps new tokens at half the context; mirror that so the budget arithmetic matches.
        self.max_new_tokens = min(max_new_tokens, context // 2)
        self.prompt_budget = self.token_budget - self.max_new_tokens
        self.passage_tokens, self.overlap_tokens = passage_tokens, overlap_tokens
        self.repetition_penalty = repetition_penalty
        self.count_tokens = token_counter(tokenizer)
        self._passages_by_url: dict[str, list[Passage]] = {}

    def passages_for(self, page: WikiPage) -> list[Passage]:
        if page.url not in self._passages_by_url:
            self._passages_by_url[page.url] = chunk_page(page, self.count_tokens, self.passage_tokens,
                                                         self.overlap_tokens)
        return self._passages_by_url[page.url]

    def retrieve(self, question: str) -> list[Passage]:
        passages = [passage for page in self.wiki.pages_for_question(question, self.pages)
                    for passage in self.passages_for(page)]
        # Title and section words count too: "Canberra" should match the lead of the Canberra article.
        scores = BM25([f"{passage.label} {passage.text}" for passage in passages]).scores(question)
        boosted = [score * (LEAD_SECTION_BOOST if passage.section == "" else 1.0)
                   for score, passage in zip(scores, passages)]
        ranked = sorted((index for index, score in enumerate(boosted) if score > 0), key=lambda index: -boosted[index])
        selected: list[Passage] = []
        per_page: dict[str, int] = {}
        for index in ranked:
            passage = passages[index]
            if per_page.get(passage.url, 0) < MAX_PASSAGES_PER_PAGE:
                per_page[passage.url] = per_page.get(passage.url, 0) + 1
                selected.append(passage)
                if len(selected) == self.top_k:
                    break
        return selected

    def answer(self, question: str, use_retrieval: bool = True) -> RAGAnswer:
        retrieved = self.retrieve(question) if use_retrieval else []
        prompt, used, prompt_tokens = fit_prompt(question, retrieved, self.tokenizer, self.prompt_budget, self.chat)
        text = generate_text(self.model, self.tokenizer, prompt, self.device, self.chat,
                             max_new_tokens=self.max_new_tokens, temperature=0.0,
                             repetition_penalty=self.repetition_penalty,
                             stop_strings=None if self.chat else BASE_STOP_STRINGS)
        return RAGAnswer(question, text.strip(), used, prompt_tokens, retrieved)


def resolve_model_path(model_argument: str | None) -> str:
    if model_argument:
        return model_argument
    for candidate in DEFAULT_MODEL_CANDIDATES:
        folder = Path(candidate).parent
        if folder.is_dir() and any(folder.glob("step-*")):
            return candidate
    raise FileNotFoundError(f"No MiniLLM checkpoint in {', '.join(DEFAULT_MODEL_CANDIDATES)}; pass --model")


def default_tokenizer_dir(model_path: str) -> str | None:
    """MiniLLM checkpoints record the tokenizer they were trained with; HF models bring their own."""
    if model_path.startswith(HF_PREFIX):
        return None
    saved_config = json.loads((resolve_checkpoint(model_path) / "config.json").read_text())
    return saved_config.get("data", {}).get("tokenizer_dir")


def add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=None, help="checkpoint dir, hub:<repo>:<path> or hf:<repo> "
                        "(default: newest of " + ", ".join(DEFAULT_MODEL_CANDIDATES) + ")")
    parser.add_argument("--tokenizer", default=None, help="default: the tokenizer recorded in the checkpoint")
    prompt_style = parser.add_mutually_exclusive_group()
    prompt_style.add_argument("--chat", dest="chat", action="store_true", default=None,
                              help="ChatML prompt (default for sft/dpo/instruct/chat models)")
    prompt_style.add_argument("--base", dest="chat", action="store_false", help="plain Question/Answer prompt")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="passages put into the prompt")
    parser.add_argument("--pages", type=int, default=DEFAULT_PAGES, help="Wikipedia articles fetched per question")
    parser.add_argument("--token-budget", type=int, default=DEFAULT_TOKEN_BUDGET,
                        help="prompt + answer tokens (capped at the model's context)")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--passage-tokens", type=int, default=DEFAULT_PASSAGE_TOKENS)
    parser.add_argument("--overlap-tokens", type=int, default=DEFAULT_OVERLAP_TOKENS)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--offline", action="store_true", help="use cached Wikipedia pages only")


def load_pipeline(args: argparse.Namespace) -> WikiRAG:
    model_path = resolve_model_path(args.model)
    device = pick_device()
    model = load_eval_model(model_path, device, max_seq_len=args.token_budget)
    tokenizer = load_eval_tokenizer(model_path, args.tokenizer or default_tokenizer_dir(model_path))
    chat = args.chat if args.chat is not None else any(marker in model_path.lower() for marker in CHAT_MODEL_MARKERS)
    pipeline = WikiRAG(model, tokenizer, device, WikiSource(args.cache_dir, offline=args.offline), chat,
                       top_k=args.top_k, pages=args.pages, token_budget=args.token_budget,
                       max_new_tokens=args.max_new_tokens, passage_tokens=args.passage_tokens,
                       overlap_tokens=args.overlap_tokens)
    print(f"model {model_path} on {device} | {'ChatML' if chat else 'base'} prompt | budget "
          f"{pipeline.token_budget} tokens = {pipeline.prompt_budget} prompt + {pipeline.max_new_tokens} answer",
          file=sys.stderr)
    return pipeline


def print_passages(result: RAGAnswer, show_text: bool) -> None:
    if not result.retrieved:
        print("No Wikipedia passages found; answered without retrieval.")
        return
    print(f"\nPassages in prompt ({len(result.passages)} of {len(result.retrieved)} retrieved, "
          f"{result.prompt_tokens} prompt tokens):")
    for number, passage in enumerate(result.passages, start=1):
        print(f"  [{number}] {passage.label}  {passage.url}")
        if show_text:
            print(f"      {' '.join(passage.text.split())}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--compare", action="store_true", help="also answer without retrieval")
    parser.add_argument("--show-passages", action="store_true", help="print the passage texts")
    add_pipeline_arguments(parser)
    args = parser.parse_args()

    pipeline = load_pipeline(args)
    result = pipeline.answer(args.question)
    if args.compare:
        print(f"Without retrieval: {pipeline.answer(args.question, use_retrieval=False).answer}")
        print(f"With retrieval:    {result.answer}")
    else:
        print(result.answer)
    print_passages(result, args.show_passages)


if __name__ == "__main__":
    main()
