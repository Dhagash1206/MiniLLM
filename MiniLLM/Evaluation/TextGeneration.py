"""Prompt formatting + KV-cached generation helpers shared by the generation-based evals."""

from __future__ import annotations

import torch

from MiniLLM.Data.Tokenizer import IM_END


def generate_text(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    chat: bool,
    max_new_tokens: int = 384,
    temperature: float = 0.0,
    top_p: float = 0.95,
    repetition_penalty: float = 1.0,
) -> str:
    if chat:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    # Leave room for the answer inside the context window (keeps at least half for the prompt).
    max_new_tokens = min(max_new_tokens, model.config.max_seq_len // 2)
    input_ids = input_ids[-(model.config.max_seq_len - max_new_tokens) :]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    stop_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(IM_END)]
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        output = model.generate(
            input_tensor, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
            repetition_penalty=repetition_penalty, eos_token_ids=stop_ids,
        )
    return tokenizer.decode(output[0, len(input_ids) :].tolist(), skip_special_tokens=True)
