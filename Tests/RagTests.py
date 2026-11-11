"""RAG tests: fully offline, using saved Wikipedia pages, a fake API and a tiny random model."""

import io
import time
import urllib.error
import urllib.request

import pytest
import torch

from MiniLLM.Config import ModelConfig
from MiniLLM.Data.Tokenizer import load_tokenizer
from MiniLLM.Model import Transformer
from MiniLLM.RAG.Answer import WikiRAG, build_prompt, fit_prompt, prompt_token_count
from MiniLLM.RAG.BM25 import BM25, tokenize
from MiniLLM.RAG.Chunker import Passage, chunk_page, split_sections, split_text, token_counter
from MiniLLM.RAG.Compare import build_report, contains_answer
from MiniLLM.RAG.WikiSource import (
    OfflineError,
    WikiPage,
    WikiSource,
    fetch_json_over_http,
    retry_delay_seconds,
    search_query_from_question,
)

CANBERRA = WikiPage("Canberra", "https://en.wikipedia.org/wiki/Canberra", (
    "Canberra is the capital city of Australia. It was founded in 1913 as a planned city.\n\n"
    "== Geography ==\nThe city lies at the northern end of the Australian Capital Territory.\n\n"
    "== Climate ==\nCanberra has warm summers and cool winters with frequent frosts.\n\n"
    "== See also ==\nList of cities in Australia\n\n== References ==\n"))
SYDNEY = WikiPage("Sydney", "https://en.wikipedia.org/wiki/Sydney", (
    "Sydney is the most populous city in Australia. It is not the capital.\n\n"
    "== Harbour ==\nThe Sydney Opera House stands on the harbour."))
PYTHON = WikiPage("Python (programming language)", "https://en.wikipedia.org/wiki/Python_(programming_language)",
                  "Python is a programming language created by Guido van Rossum and first released in 1991.")


def count_words(text: str) -> int:
    return len(text.split())


@pytest.fixture()
def project_tokenizer(tokenizer_dir):
    return load_tokenizer(tokenizer_dir)


@pytest.fixture()
def saved_pages(tmp_path):
    """A Wikipedia cache as WikiSource writes it, read back offline."""
    writer = WikiSource(tmp_path / "wiki_cache")
    for page in (CANBERRA, SYDNEY, PYTHON):
        writer.save_page(page)
    return WikiSource(tmp_path / "wiki_cache", offline=True)


def test_search_query_keeps_only_content_words():
    assert search_query_from_question("What is the capital of Australia?") == "capital Australia"
    assert search_query_from_question("Which language did Guido van Rossum create?") == \
        "language Guido van Rossum create"
    assert search_query_from_question("Who?") == "Who?"  # nothing left: fall back to the question


def test_wiki_source_fetches_once_then_serves_from_cache(tmp_path):
    calls = []

    def fake_api(url, params):
        calls.append(params)
        if params.get("list") == "search":
            return {"query": {"search": [{"title": "Canberra"}, {"title": "Missing page"}]}}
        if params["titles"] == "Canberra":
            return {"query": {"pages": [{"title": CANBERRA.title, "fullurl": CANBERRA.url,
                                         "extract": CANBERRA.text}]}}
        return {"query": {"pages": [{"title": params["titles"], "missing": True}]}}

    online = WikiSource(tmp_path / "cache", fetch_json=fake_api, min_seconds_between_requests=0)
    assert online.pages_for_question("What is the capital of Australia?") == [CANBERRA]
    assert calls[0]["srsearch"] == "capital Australia"
    assert len(calls) == 3  # one search + two page lookups

    offline = WikiSource(tmp_path / "cache", offline=True)
    assert offline.pages_for_question("What is the capital of Australia?") == [CANBERRA]
    assert offline.page("Never fetched") is None
    with pytest.raises(OfflineError):
        offline._call_api({"action": "query"})


def test_fetch_retries_rate_limits_honouring_retry_after(monkeypatch):
    rate_limited = urllib.error.HTTPError("https://x", 429, "Too Many Requests", {"Retry-After": "7"}, None)
    responses = [rate_limited, io.BytesIO(b'{"ok": true}')]
    sleeps = []

    def fake_urlopen(request, timeout):
        assert request.get_header("User-agent") == "test-agent"
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", sleeps.append)
    assert fetch_json_over_http("https://x", {"action": "query"}, "test-agent") == {"ok": True}
    assert sleeps == [7.0]
    assert retry_delay_seconds(urllib.error.URLError("down"), attempt=2) == 8.0

    not_found = urllib.error.HTTPError("https://x", 404, "Not Found", {}, None)
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: (_ for _ in ()).throw(not_found))
    with pytest.raises(urllib.error.HTTPError):
        fetch_json_over_http("https://x", {}, "test-agent")


def test_offline_search_ranks_saved_pages(saved_pages):
    assert saved_pages.search("Guido van Rossum programming", limit=1) == ["Python (programming language)"]
    assert saved_pages.search("capital Canberra", limit=2)[0] == "Canberra"


def test_split_sections_drops_reference_sections():
    sections = split_sections(CANBERRA.text)
    assert [heading for heading, _ in sections] == ["", "Geography", "Climate"]
    assert sections[0][1].startswith("Canberra is the capital")


def test_split_text_respects_limit_and_overlaps():
    paragraphs = [" ".join(f"p{paragraph}w{word}" for word in range(12)) + "." for paragraph in range(10)]
    passages = split_text("\n\n".join(paragraphs), count_words, max_tokens=30, overlap_tokens=12)
    assert len(passages) > 1
    assert all(count_words(passage) <= 30 for passage in passages)
    for previous, following in zip(passages, passages[1:]):
        assert previous.split()[-1] in following.split()
    assert set(" ".join(passages).split()) == set(" ".join(paragraphs).split())
    with pytest.raises(ValueError):
        split_text("text", count_words, max_tokens=10, overlap_tokens=10)


def test_chunk_page_measures_with_project_tokenizer(project_tokenizer):
    count_tokens = token_counter(project_tokenizer)
    passages = chunk_page(CANBERRA, count_tokens, max_tokens=24, overlap_tokens=6)
    assert {passage.section for passage in passages} == {"", "Geography", "Climate"}
    assert all(count_tokens(passage.text) <= 24 for passage in passages)
    assert all(passage.url == CANBERRA.url for passage in passages)


def test_bm25_ranks_matching_passage_first():
    ranker = BM25(["the cat sat on the mat", "planets orbit the sun; Jupiter is the largest planet",
                   "dogs bark at cats"])
    assert tokenize("The Planets") == ["planet"]
    top = ranker.top_k("Which is the largest planet?", k=3)
    assert top[0][0] == 1
    assert all(score > 0 for _, score in top)
    assert len(top) == 1  # documents sharing no word are not returned
    assert ranker.idf("jupiter") > ranker.idf("cat")  # rarer word ("cat" + "cats" in 2 of 3 documents)


def test_fit_prompt_drops_then_trims_passages(project_tokenizer):
    passages = [Passage("Canberra", "", CANBERRA.url, CANBERRA.text.split("\n\n")[0] * 4),
                Passage("Sydney", "", SYDNEY.url, SYDNEY.text * 3)]
    question = "What is the capital of Australia?"
    full_tokens = prompt_token_count(project_tokenizer, build_prompt(question, passages, chat=True), chat=True)

    prompt, used, tokens = fit_prompt(question, passages, project_tokenizer, full_tokens, chat=True)
    assert used == passages and tokens == full_tokens

    one_passage_tokens = prompt_token_count(project_tokenizer, build_prompt(question, passages[:1], True), True)
    _, used, tokens = fit_prompt(question, passages, project_tokenizer, one_passage_tokens, chat=True)
    assert used == passages[:1] and tokens <= one_passage_tokens

    _, used, tokens = fit_prompt(question, passages, project_tokenizer, one_passage_tokens - 15, chat=True)
    assert len(used) == 1 and used[0].text != passages[0].text and tokens <= one_passage_tokens - 15

    prompt, used, _ = fit_prompt(question, passages, project_tokenizer, 5, chat=False)
    assert used == [] and prompt == f"Question: {question}\nAnswer:"


def test_answer_end_to_end_with_tiny_model(saved_pages, project_tokenizer):
    torch.manual_seed(0)
    config = ModelConfig(vocab_size=len(project_tokenizer), dim=64, n_layers=2, n_heads=4, n_kv_heads=2,
                         ffn_hidden_dim=128, max_seq_len=256, gradient_checkpointing=False)
    model = Transformer(config).eval()
    pipeline = WikiRAG(model, project_tokenizer, torch.device("cpu"), saved_pages, chat=True, top_k=2, pages=2,
                       token_budget=1024, max_new_tokens=16, passage_tokens=40, overlap_tokens=8)
    assert pipeline.token_budget == 256  # capped at the model's context
    assert pipeline.prompt_budget == 256 - 16

    result = pipeline.answer("What is the capital of Australia?")
    assert isinstance(result.answer, str)
    assert result.passages and result.passages[0].title == "Canberra"
    assert result.prompt_tokens <= pipeline.prompt_budget

    pipeline.top_k = 5
    sources = [passage.url for passage in pipeline.retrieve("Canberra capital city Australia climate geography")]
    assert max(sources.count(url) for url in sources) <= 2  # one article cannot fill every slot

    without = pipeline.answer("What is the capital of Australia?", use_retrieval=False)
    assert without.passages == [] and without.retrieved == []


def test_compare_scoring_and_report():
    assert contains_answer("The capital is Canberra.", ["Canberra"])
    assert not contains_answer("Australia is large.", ["Au"])  # whole words only
    assert contains_answer("about 299,792 km/s", ["299,792"])
    report = build_report([{"question": "Q?", "answers": ["A"], "answer_with": "A", "answer_without": "B",
                            "correct_with": True, "correct_without": False, "retrieval_hit": True}], "tiny")
    assert "| With retrieval | 1/1 (100%) |" in report and "| Without retrieval | 0/1 (0%) |" in report
