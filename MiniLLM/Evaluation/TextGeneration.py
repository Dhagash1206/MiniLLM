"""Prompt formatting + batched, KV-cached generation helpers shared by the generation-based evals."""

from __future__ import annotations

from collections.abc import Iterator

import torch

from MiniLLM.Data.Tokenizer import IM_END
from MiniLLM.Evaluation.Common import autocast_context


def format_prompt(tokenizer, prompt: str, chat: bool) -> str:
    if not chat:
        return prompt
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                         add_generation_prompt=True)


def stop_token_ids(tokenizer) -> list[int]:
    # Baseline tokenizers may lack <|im_end|> (None, or mapped to unk).
    im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
    return sorted(token_id for token_id in {tokenizer.eos_token_id, im_end_id}
                  if token_id is not None and token_id != tokenizer.unk_token_id)


def cut_at_stop_strings(text: str, stop_strings: list[str] | None) -> str:
    cut = min((text.find(stop) for stop in stop_strings or [] if stop in text), default=len(text))
    return text[:cut]


def generate_texts(
    model,
    tokenizer,
    prompts: list[str],
    device: torch.device,
    chat: bool,
    max_new_tokens: int = 384,
    temperature: float = 0.0,
    top_p: float = 0.95,
    repetition_penalty: float = 1.0,
    stop_strings: list[str] | None = None,
    batch_size: int = 8,
) -> list[str]:
    """Completions in prompt order. Prompts are sorted by length and left-padded, so batches waste little."""
    # Leave room for the answer inside the context window (keeps at least half for the prompt).
    max_new_tokens = min(max_new_tokens, model.config.max_seq_len // 2)
    prompt_budget = model.config.max_seq_len - max_new_tokens
    encoded = [tokenizer.encode(format_prompt(tokenizer, prompt, chat), add_special_tokens=False)[-prompt_budget:]
               for prompt in prompts]
    stop_ids = stop_token_ids(tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (stop_ids[0] if stop_ids else 0)

    completions = [""] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda index: len(encoded[index]))
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        width = max(len(encoded[index]) for index in indices)
        input_ids = torch.full((len(indices), width), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(indices), width), dtype=torch.long)
        for row, index in enumerate(indices):
            input_ids[row, width - len(encoded[index]) :] = torch.tensor(encoded[index])
            attention_mask[row, width - len(encoded[index]) :] = 1
        with autocast_context(device):
            output = model.generate(
                input_ids.to(device), max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
                repetition_penalty=repetition_penalty, eos_token_ids=stop_ids,
                attention_mask=attention_mask.to(device), stop_strings=stop_strings, tokenizer=tokenizer,
                pad_token_id=pad_id,
            )
        for row, index in enumerate(indices):
            new_ids = output[row, width:].tolist()
            end = next((position for position, token_id in enumerate(new_ids) if token_id in stop_ids), len(new_ids))
            text = tokenizer.decode(new_ids[:end], skip_special_tokens=True)
            completions[index] = cut_at_stop_strings(text, stop_strings)
    return completions


def generate_text(model, tokenizer, prompt: str, device: torch.device, chat: bool, **kwargs) -> str:
    return generate_texts(model, tokenizer, [prompt], device, chat, batch_size=1, **kwargs)[0]


def stream_text(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    chat: bool,
    max_new_tokens: int = 384,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stop_strings: list[str] | None = None,
) -> Iterator[str]:
    """Yields text pieces as they are generated (MiniLLM models only), holding back a possible stop-string prefix."""
    max_new_tokens = min(max_new_tokens, model.config.max_seq_len // 2)
    input_ids = tokenizer.encode(format_prompt(tokenizer, prompt, chat), add_special_tokens=False)
    input_ids = input_ids[-(model.config.max_seq_len - max_new_tokens) :]
    stop_ids = stop_token_ids(tokenizer)
    generated: list[int] = []
    emitted = ""
    holdback = max(map(len, stop_strings)) - 1 if stop_strings else 0
    with autocast_context(device):
        for next_token in model.stream_generate(
            torch.tensor([input_ids], device=device), max_new_tokens, temperature=temperature, top_p=top_p,
            eos_token_ids=stop_ids, stop_strings=stop_strings, tokenizer=tokenizer,
        ):
            token_id = int(next_token)
            if token_id in stop_ids:
                break
            generated.append(token_id)
            text = tokenizer.decode(generated, skip_special_tokens=True)
            stopped = cut_at_stop_strings(text, stop_strings)
            # Unfinished UTF-8 sequences decode to U+FFFD until their last byte arrives.
            safe_end = len(stopped) if len(stopped) < len(text) else max(len(emitted), len(text) - holdback)
            safe_end = min(safe_end, len(text.rstrip("�")))
            if safe_end > len(emitted):
                yield stopped[len(emitted) : safe_end]
                emitted = stopped[:safe_end]
            if len(stopped) < len(text):
                return
    text = cut_at_stop_strings(tokenizer.decode(generated, skip_special_tokens=True), stop_strings)
    if len(text) > len(emitted):
        yield text[len(emitted) :]
