"""Retrieval-augmented generation for MiniLLM: Wikipedia + BM25, pure Python, no fine-tuning.

    WikiSource.py  search the Wikipedia API, download plain-text pages, cache them on disk
    Chunker.py     split pages into small overlapping passages, measured with the project tokenizer
    BM25.py        from-scratch BM25 ranking of the passages
    Answer.py      fit the top passages into the ChatML prompt within the context budget; CLI
    Compare.py     accuracy on a question set with and without retrieval
"""
