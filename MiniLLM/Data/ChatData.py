"""ChatML conversation encoding plus SFT and DPO dataset builders."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from MiniLLM.Data.Tokenizer import EOS_TOKEN, IM_END, IM_START, PAD_TOKEN

ROLE_ALIASES = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "system": "system"}
SPECIAL_STRINGS = (IM_START, IM_END, EOS_TOKEN, PAD_TOKEN)
IGNORE_INDEX = -100


def load_raw_tokenizer(tokenizer_dir: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(Path(tokenizer_dir) / "tokenizer.json"))


def render_conversation(messages: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    """Render with the same layout as CHAT_TEMPLATE; return text and assistant char spans to train on."""
    pieces: list[str] = []
    assistant_spans: list[tuple[int, int]] = []
    cursor = 0
    for message in messages:
        header = f"{IM_START}{message['role']}\n"
        body = f"{message['content']}{IM_END}"
        pieces.extend([header, body, "\n"])
        body_start = cursor + len(header)
        if message["role"] == "assistant":
            assistant_spans.append((body_start, body_start + len(body)))
        cursor = body_start + len(body) + 1
    return "".join(pieces), assistant_spans


def encode_conversations(
    tokenizer: Tokenizer, conversations: list[list[dict]], only_last_assistant: bool = False
) -> list[tuple[np.ndarray, np.ndarray]]:
    rendered = [render_conversation(messages) for messages in conversations]
    encodings = tokenizer.encode_batch([text for text, _ in rendered], add_special_tokens=False)
    examples = []
    for (_, spans), encoding in zip(rendered, encodings):
        if only_last_assistant:
            spans = spans[-1:]
        input_ids = np.asarray(encoding.ids, dtype=np.int32)
        labels = np.full(len(input_ids), IGNORE_INDEX, dtype=np.int32)
        # Offsets map tokens back to characters, so masks stay exact regardless of BPE merges.
        for token_index, (start, _end) in enumerate(encoding.offsets):
            if any(span_start <= start < span_end for span_start, span_end in spans):
                labels[token_index] = input_ids[token_index]
        examples.append((input_ids, labels))
    return examples


def _clean(messages: list[dict]) -> list[dict] | None:
    cleaned = []
    for message in messages:
        role = ROLE_ALIASES.get(message["role"])
        content = (message.get("content") or "").strip()
        if role is None or not content or any(special in content for special in SPECIAL_STRINGS):
            return None
        cleaned.append({"role": role, "content": content})
    if not any(message["role"] == "assistant" for message in cleaned):
        return None
    return cleaned


def load_sft_conversations(openhermes_samples: int, seed: int) -> list[list[dict]]:
    from datasets import load_dataset

    conversations: list[list[dict]] = []

    code_alpaca = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    for row in code_alpaca:
        prompt = row["instruction"] + (f"\n\n{row['input']}" if row.get("input") else "")
        conversations.append([{"role": "user", "content": prompt}, {"role": "assistant", "content": row["output"]}])

    magicoder = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
    for row in magicoder:
        conversations.append(
            [{"role": "user", "content": row["problem"]}, {"role": "assistant", "content": row["solution"]}]
        )

    if openhermes_samples > 0:
        openhermes = load_dataset("teknium/OpenHermes-2.5", split="train").shuffle(seed=seed)
        openhermes = openhermes.select(range(min(openhermes_samples, len(openhermes))))
        for row in openhermes:
            conversations.append(
                [{"role": turn["from"], "content": turn["value"]} for turn in row["conversations"]]
            )

    cleaned = [_clean(messages) for messages in conversations]
    return [messages for messages in cleaned if messages is not None]


def build_sft_dataset(
    tokenizer_dir: str, max_seq_len: int, openhermes_samples: int, seed: int, max_examples: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    cache_path = Path(f"data/cache/sft_len{max_seq_len}_oh{openhermes_samples}_max{max_examples}_seed{seed}.pt")
    if cache_path.exists():
        return torch.load(cache_path, weights_only=False)
    tokenizer = load_raw_tokenizer(tokenizer_dir)
    conversations = load_sft_conversations(openhermes_samples, seed)
    if max_examples:
        random.Random(seed).shuffle(conversations)
        conversations = conversations[:max_examples]
    examples = []
    for start in range(0, len(conversations), 2000):
        examples.extend(encode_conversations(tokenizer, conversations[start : start + 2000]))
    # Drop (not truncate) long examples so no answer is cut mid-way.
    examples = [example for example in examples if len(example[0]) <= max_seq_len and (example[1] >= 0).any()]
    random.Random(seed).shuffle(examples)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(examples, cache_path)
    return examples


def load_preference_pairs(dataset_name: str, split: str, max_pairs: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split).shuffle(seed=seed)
    pairs = []
    for row in dataset:
        chosen = _clean(row["chosen"])
        rejected = _clean(row["rejected"])
        if not chosen or not rejected or chosen[-1]["role"] != "assistant" or rejected[-1]["role"] != "assistant":
            continue
        if chosen[-1]["content"] == rejected[-1]["content"]:
            continue
        pairs.append({"prompt": chosen[:-1], "chosen": chosen[-1]["content"], "rejected": rejected[-1]["content"]})
        if len(pairs) >= max_pairs * 3:  # Headroom: many pairs get dropped by the length filter.
            break
    return pairs


def build_dpo_dataset(
    tokenizer_dir: str, dataset_name: str, split: str, max_pairs: int, max_seq_len: int, seed: int,
) -> list[dict]:
    safe_name = dataset_name.replace("/", "__")
    cache_path = Path(f"data/cache/dpo_{safe_name}_{split}_n{max_pairs}_len{max_seq_len}_seed{seed}.pt")
    if cache_path.exists():
        return torch.load(cache_path, weights_only=False)
    tokenizer = load_raw_tokenizer(tokenizer_dir)
    pairs = load_preference_pairs(dataset_name, split, max_pairs, seed)
    examples = []
    for start in range(0, len(pairs), 1000):
        batch = pairs[start : start + 1000]
        chosen = encode_conversations(
            tokenizer, [pair["prompt"] + [{"role": "assistant", "content": pair["chosen"]}] for pair in batch], True
        )
        rejected = encode_conversations(
            tokenizer, [pair["prompt"] + [{"role": "assistant", "content": pair["rejected"]}] for pair in batch], True
        )
        for chosen_example, rejected_example in zip(chosen, rejected):
            if max(len(chosen_example[0]), len(rejected_example[0])) <= max_seq_len:
                examples.append({"chosen": chosen_example, "rejected": rejected_example})
        if len(examples) >= max_pairs:
            break
    examples = examples[:max_pairs]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(examples, cache_path)
    return examples


def pad_batch(examples: list[tuple[np.ndarray, np.ndarray]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad; with causal attention, trailing pads never influence real tokens."""
    max_len = max(len(input_ids) for input_ids, _ in examples)
    input_batch = torch.full((len(examples), max_len), pad_id, dtype=torch.long)
    label_batch = torch.full((len(examples), max_len), IGNORE_INDEX, dtype=torch.long)
    for row, (input_ids, labels) in enumerate(examples):
        input_batch[row, : len(input_ids)] = torch.from_numpy(input_ids.astype(np.int64))
        label_batch[row, : len(labels)] = torch.from_numpy(labels.astype(np.int64))
    # Shift so position t predicts token t+1.
    return input_batch[:, :-1], label_batch[:, 1:]


def length_grouped_batches(lengths: list[int], batch_size: int, seed: int, world_size: int) -> list[list[int]]:
    """Shuffle, then sort within mega-batches so each batch has similar lengths (less padding)."""
    rng = random.Random(seed)
    indices = list(range(len(lengths)))
    rng.shuffle(indices)
    mega_size = batch_size * 64
    batches = []
    for start in range(0, len(indices), mega_size):
        mega = sorted(indices[start : start + mega_size], key=lambda index: lengths[index], reverse=True)
        batches.extend(mega[offset : offset + batch_size] for offset in range(0, len(mega), batch_size))
    rng.shuffle(batches)
    # Every rank must run the same number of steps.
    usable = len(batches) - len(batches) % world_size
    return batches[:usable]
