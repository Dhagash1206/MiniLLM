"""Build a tiny offline dataset (Python stdlib source as "code", its docstrings as "english")
so the whole pipeline can be smoke-tested without network access or gated datasets.

Usage:
    python Scripts/MakeDebugData.py --out data/tokens_debug
"""

import argparse
import ast
import json
import sysconfig
from pathlib import Path

import numpy as np

from MiniLLM.Data.DocumentFilters import code_quality_ok, contains_secret
from MiniLLM.Data.Tokenizer import EOS_TOKEN, save_tokenizer, train_tokenizer


def collect_documents(max_files: int) -> tuple[list[str], list[str]]:
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    code_docs, english_docs = [], []
    for path in sorted(stdlib.glob("*.py"))[:max_files]:
        source = path.read_text(errors="ignore")
        if not code_quality_ok(source) or contains_secret(source):
            continue
        code_docs.append(source)
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        docstrings = [ast.get_docstring(node) for node in ast.walk(tree)
                      if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef))]
        text = "\n\n".join(doc for doc in docstrings if doc and len(doc.split()) > 8)
        if text:
            english_docs.append(text)
    return english_docs, code_docs


def write_source(out_dir: Path, tokenizer, name: str, domain: str, docs: list[str]) -> None:
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    encodings = tokenizer.encode_batch(docs, add_special_tokens=False)
    arrays = [np.asarray(encoding.ids + [eos_id], dtype=np.uint16) for encoding in encodings]
    val_count = max(1, len(arrays) // 10)
    np.concatenate(arrays[:val_count]).tofile(out_dir / f"{name}_val_0000.bin")
    train = np.concatenate(arrays[val_count:])
    train.tofile(out_dir / f"{name}_train_0000.bin")
    meta = {"source": name, "domain": domain, "train_tokens": int(len(train)),
            "train_files": [f"{name}_train_0000.bin"], "val_files": [f"{name}_val_0000.bin"], "dtype": "uint16"}
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2))
    print(f"{name}: {len(train):,} train tokens")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/tokens_debug")
    parser.add_argument("--max-files", type=int, default=400)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    english_docs, code_docs = collect_documents(args.max_files)
    tokenizer = train_tokenizer(iter(english_docs + code_docs), vocab_size=32768)
    save_tokenizer(tokenizer, out_dir / "tokenizer", model_max_length=256)
    write_source(out_dir, tokenizer, "fineweb_edu", "english", english_docs)
    write_source(out_dir, tokenizer, "python", "code", code_docs)


if __name__ == "__main__":
    main()
