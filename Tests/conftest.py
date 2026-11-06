import pytest

from MiniLLM.Data.Tokenizer import save_tokenizer, train_tokenizer

SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog. Students learn about photosynthesis in biology class.",
    "def add(a, b):\n    if a:\n        return a + b\n    return b\n",
    "class Stack:\n    def __init__(self):\n        self.items = []\n\n    def push(self, item):\n        self.items.append(item)\n",
    "function greet(name) {\n  if (name) {\n    return `Hello ${name}`;\n  }\n}\n",
    "SELECT name, COUNT(*) FROM orders\n  WHERE amount > 10\n  GROUP BY name;\n",
    "Water evaporates, condenses into clouds, and falls as rain.\n\tIndented with a tab.\n",
] * 50


@pytest.fixture(scope="session")
def tokenizer_dir(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("tokenizer")
    tokenizer = train_tokenizer(iter(SAMPLE_TEXTS), vocab_size=600)
    save_tokenizer(tokenizer, out_dir, model_max_length=128)
    return out_dir
