"""Train and load the 32k byte-level BPE tokenizer.

Usage:
    python -m MiniLLM.Data.Tokenizer --out artifacts/tokenizer --docs-per-source 150000
"""

from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

EOS_TOKEN = "<|endoftext|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
PAD_TOKEN = "<|pad|>"
SPECIAL_TOKENS = [EOS_TOKEN, IM_START, IM_END, PAD_TOKEN]

# GPT-4-style split with one addition at the front: whitespace that follows a newline
# is isolated, so code indentation (2/4/8 spaces, tabs) becomes its own token instead of
# leaking one space into the next word.
SPLIT_PATTERN = (
    r"(?<=\n)[ \t]+"
    r"|'(?i:[sdmt]|ll|ve|re)"
    r"|[^\r\n\p{L}\p{N}]?+\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]++[\r\n]*"
    r"|\s*[\r\n]"
    r"|\s+(?!\S)"
    r"|\s+"
)

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def build_empty_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(byte_fallback=False))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(SPLIT_PATTERN), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def train_tokenizer(text_iterator, vocab_size: int = 32768, total_docs: int | None = None) -> Tokenizer:
    tokenizer = build_empty_tokenizer()
    # Checked after training: common indents should each be one token.
    indent_tokens = [" " * width for width in (2, 4, 6, 8, 12, 16, 20, 24)] + ["\t" * depth for depth in (1, 2, 3, 4)]
    byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(text_iterator, trainer=trainer, length=total_docs)

    vocab = tokenizer.get_vocab()
    missing = [
        indent for indent in indent_tokens
        if "".join(piece for piece, _ in byte_level.pre_tokenize_str(indent)) not in vocab
    ]
    if missing:
        print(f"Warning: indentation strings not learned as single tokens: {[repr(m) for m in missing]}")
    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, out_dir: str | Path, model_max_length: int = 1024) -> None:
    from transformers import PreTrainedTokenizerFast

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=EOS_TOKEN,
        eos_token=EOS_TOKEN,
        unk_token=None,
        pad_token=PAD_TOKEN,
        additional_special_tokens=[IM_START, IM_END],
        model_max_length=model_max_length,
        clean_up_tokenization_spaces=False,
    )
    hf_tokenizer.chat_template = CHAT_TEMPLATE
    hf_tokenizer.save_pretrained(out_dir)


def load_tokenizer(tokenizer_dir: str | Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(tokenizer_dir))


def mixed_sample_iterator(docs_per_source: int, seed: int):
    from MiniLLM.Data.PretrainSources import PRETRAIN_SOURCES, iter_documents

    # Weight sources by their pretraining mix so merges reflect the 70/30 split.
    max_fraction = max(spec.mix_fraction for spec in PRETRAIN_SOURCES)
    streams = []
    for spec in PRETRAIN_SOURCES:
        doc_count = max(1, int(docs_per_source * spec.mix_fraction / max_fraction))
        # A different shuffle seed than pretraining gives a different shard order (skip() would download everything).
        stream = iter_documents(spec, seed=seed + 7)
        streams.append(itertools.islice(stream, doc_count))

    rng = random.Random(seed)
    active_streams = list(streams)
    while active_streams:
        stream = rng.choice(active_streams)
        try:
            yield next(stream)
        except StopIteration:
            active_streams.remove(stream)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="artifacts/tokenizer")
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--docs-per-source", type=int, default=150_000, help="Docs for the largest source")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--push-to-hub", default=None, help="Optional HF repo id to upload the tokenizer to")
    args = parser.parse_args()

    tokenizer = train_tokenizer(mixed_sample_iterator(args.docs_per_source, args.seed), args.vocab_size)
    save_tokenizer(tokenizer, args.out)
    print(f"Saved tokenizer ({tokenizer.get_vocab_size()} tokens) to {args.out}")

    sample = "def add(a, b):\n    if a:\n        return a + b\n\tpass\n"
    encoding = tokenizer.encode(sample)
    print("Indentation check:", encoding.tokens)

    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push_to_hub, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(folder_path=args.out, repo_id=args.push_to_hub, path_in_repo="tokenizer")
        print(f"Uploaded tokenizer to {args.push_to_hub}/tokenizer")


if __name__ == "__main__":
    main()
