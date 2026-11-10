"""Llama-style decoder: RoPE, RMSNorm, SwiGLU, SDPA attention.

Module names mirror Hugging Face `LlamaForCausalLM` so export is a pure key rename
(see `MiniLLM/HFExport.py`).
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from MiniLLM.Config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # Normalize in fp32 for stability under fp16 autocast (matches HF Llama).
        input_dtype = hidden.dtype
        hidden_fp32 = hidden.float()
        hidden_fp32 = hidden_fp32 * torch.rsqrt(hidden_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * hidden_fp32.to(input_dtype)


def build_rope_cache(head_dim: int, max_seq_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    # "rotate_half" layout (first half / second half), same as HF Llama, so no weight permutation on export.
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    first_half, second_half = tensor.chunk(2, dim=-1)
    return torch.cat([-second_half, first_half], dim=-1)


def apply_rope(tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return tensor * cos + rotate_half(tensor) * sin


class KVCache:
    """Per-layer key/value buffers preallocated to `max_length`; decoding writes in place (no torch.cat per token)."""

    def __init__(self, n_layers: int, max_length: int):
        self.max_length = max_length
        self.keys: list[torch.Tensor | None] = [None] * n_layers
        self.values: list[torch.Tensor | None] = [None] * n_layers
        self.lengths = [0] * n_layers

    @property
    def length(self) -> int:
        return self.lengths[0]

    def layer_length(self, layer_index: int) -> int:
        return self.lengths[layer_index]

    def update(self, layer_index: int, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.lengths[layer_index]
        end = start + key.shape[2]
        if end > self.max_length:
            raise ValueError(f"KV cache overflow: {end} > {self.max_length}")
        if self.keys[layer_index] is None:
            batch_size, n_kv_heads, _, head_dim = key.shape
            self.keys[layer_index] = key.new_empty(batch_size, n_kv_heads, self.max_length, head_dim)
            self.values[layer_index] = value.new_empty(batch_size, n_kv_heads, self.max_length, head_dim)
        self.keys[layer_index][:, :, start:end] = key
        self.values[layer_index][:, :, start:end] = value
        self.lengths[layer_index] = end
        return self.keys[layer_index][:, :, :end], self.values[layer_index][:, :, :end]


def padded_attention_mask(attention_mask: torch.Tensor, past_length: int, seq_len: int) -> torch.Tensor:
    """(batch, total_len) 1/0 token mask -> (batch, 1, seq_len, total_len) bool mask: causal and no padding keys."""
    key_valid = attention_mask.bool()
    query_positions = torch.arange(past_length, past_length + seq_len, device=attention_mask.device)
    key_positions = torch.arange(key_valid.shape[1], device=attention_mask.device)
    causal = key_positions[None, :] <= query_positions[:, None]
    allowed = causal[None] & key_valid[:, None, :]
    # A padding query would see no key at all, and SDPA returns NaN for fully masked rows.
    allowed |= (key_positions[None, :] == query_positions[:, None])[None]
    return allowed[:, None]


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        layer_index: int = 0,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden.shape
        query = self.q_proj(hidden).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        query = apply_rope(query, cos, sin).to(value.dtype)
        key = apply_rope(key, cos, sin).to(value.dtype)

        past_length = 0
        if kv_cache is not None:
            # Per-layer length: layer 0 has already appended this step's keys when later layers run.
            past_length = kv_cache.layer_length(layer_index)
            key, value = kv_cache.update(layer_index, key, value)

        if self.n_kv_heads != self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)

        # T4 (sm75) has no FlashAttention; SDPA falls back to the memory-efficient kernel.
        if attn_mask is not None:
            attn_out = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
        elif past_length == 0:
            attn_out = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        elif seq_len == 1:
            attn_out = F.scaled_dot_product_attention(query, key, value)
        else:
            # Chunked prefill on top of a cache: new query i sees all past keys plus new keys <= i.
            total_length = key.shape[2]
            query_positions = torch.arange(past_length, past_length + seq_len, device=hidden.device)
            key_positions = torch.arange(total_length, device=hidden.device)
            attn_mask = key_positions[None, :] <= query_positions[:, None]
            attn_out = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.o_proj(attn_out)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.dim, config.ffn_hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.dim, config.ffn_hidden_dim, bias=False)
        self.down_proj = nn.Linear(config.ffn_hidden_dim, config.dim, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.dim, config.norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.dim, config.norm_eps)
        self.mlp = SwiGLU(config)

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        layer_index: int = 0,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), cos, sin, kv_cache, layer_index, attn_mask)
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        return hidden


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(DecoderLayer(config) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        rope_cos, rope_sin = build_rope_cache(config.head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)
        self.apply(self._init_weights)
        # GPT-2 style: shrink residual-branch output projections by depth.
        residual_std = 0.02 / math.sqrt(2 * config.n_layers)
        for layer in self.layers:
            nn.init.normal_(layer.self_attn.o_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(layer.mlp.down_proj.weight, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """`attention_mask` (batch, past + seq_len), 0 at padding, is only needed for left-padded batches."""
        seq_len = input_ids.shape[1]
        start_position = kv_cache.length if kv_cache is not None else 0
        if start_position + seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {start_position + seq_len} exceeds max_seq_len {self.config.max_seq_len}"
            )
        attn_mask = None
        if attention_mask is None:
            cos = self.rope_cos[start_position : start_position + seq_len]
            sin = self.rope_sin[start_position : start_position + seq_len]
        else:
            # Each row counts positions from its first real token, so left padding does not shift RoPE.
            positions = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)[:, -seq_len:]
            cos, sin = self.rope_cos[positions].unsqueeze(1), self.rope_sin[positions].unsqueeze(1)
            attn_mask = padded_attention_mask(attention_mask, start_position, seq_len)

        hidden = self.embed_tokens(input_ids)
        for layer_index, layer in enumerate(self.layers):
            if self.config.gradient_checkpointing and self.training and kv_cache is None:
                hidden = checkpoint(layer, hidden, cos, sin, None, layer_index, attn_mask, use_reentrant=False)
            else:
                hidden = layer(hidden, cos, sin, kv_cache, layer_index, attn_mask)
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_p: float | None = 0.95,
        top_k: int | None = None,
        repetition_penalty: float = 1.0,
        eos_token_ids: list[int] | None = None,
        use_cache: bool = True,
        attention_mask: torch.Tensor | None = None,
        stop_strings: list[str] | None = None,
        tokenizer=None,
        pad_token_id: int | None = None,
    ) -> torch.Tensor:
        """Prompt (cut to max_seq_len) + sampled tokens; see `stream_generate` for the arguments."""
        input_ids = input_ids[:, -self.config.max_seq_len :]
        if attention_mask is not None:
            attention_mask = attention_mask[:, -self.config.max_seq_len :]
        new_tokens = list(self.stream_generate(
            input_ids, max_new_tokens, temperature, top_p, top_k, repetition_penalty, eos_token_ids, use_cache,
            attention_mask, stop_strings, tokenizer, pad_token_id,
        ))
        return torch.cat([input_ids, *new_tokens], dim=1)

    @torch.no_grad()
    def stream_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_p: float | None = 0.95,
        top_k: int | None = None,
        repetition_penalty: float = 1.0,
        eos_token_ids: list[int] | None = None,
        use_cache: bool = True,
        attention_mask: torch.Tensor | None = None,
        stop_strings: list[str] | None = None,
        tokenizer=None,
        pad_token_id: int | None = None,
    ) -> Iterator[torch.Tensor]:
        """Yields each step's next token, shape (batch, 1), until every row has stopped.

        temperature <= 0 is greedy. Batches of different-length prompts are left-padded, with
        `attention_mask` 0 at the padding. A row stops at an EOS id or once its generated text contains
        one of `stop_strings` (decoded with `tokenizer`); the stopping token is still yielded, and stopped
        rows then receive `pad_token_id` (default: the first EOS id) so the batch stays rectangular.
        """
        if stop_strings and tokenizer is None:
            raise ValueError("stop_strings needs a tokenizer to decode the generated tokens")
        max_len = self.config.max_seq_len
        input_ids = input_ids[:, -max_len:]
        batch_size, device = input_ids.shape[0], input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        attention_mask = attention_mask[:, -max_len:].bool()
        # Generated tokens are never padding, so an unpadded prompt can keep SDPA's plain causal path.
        has_padding = not bool(attention_mask.all())
        eos_tensor = torch.tensor(eos_token_ids or [], dtype=torch.long, device=device)
        filler_id = pad_token_id if pad_token_id is not None else (eos_token_ids[0] if eos_token_ids else 0)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        generated: list[list[int]] = [[] for _ in range(batch_size)]
        # Every token decodes to at least one character, so a just-completed stop string fits in this tail.
        stop_window = max(map(len, stop_strings)) + 2 if stop_strings else 0

        def forward(ids: torch.Tensor, mask: torch.Tensor, cache: KVCache | None) -> torch.Tensor:
            return self(ids, kv_cache=cache, attention_mask=mask if has_padding else None)[0]

        cache_length = min(max_len, input_ids.shape[1] + max_new_tokens)
        kv_cache = KVCache(self.config.n_layers, cache_length) if use_cache else None
        logits = forward(input_ids, attention_mask, kv_cache)
        all_ids = input_ids
        for step in range(max_new_tokens):
            next_token = sample_next_token(
                logits[:, -1, :].float(), all_ids, temperature, top_p, top_k, repetition_penalty
            )
            next_token = torch.where(finished[:, None], filler_id, next_token)
            if eos_tensor.numel():
                finished |= torch.isin(next_token.squeeze(-1), eos_tensor)
            if stop_strings:
                stopped_rows = []
                done = finished.tolist()
                for row, token_id in enumerate(next_token.squeeze(-1).tolist()):
                    if done[row]:
                        continue
                    generated[row].append(token_id)
                    tail = tokenizer.decode(generated[row][-stop_window:])
                    if any(stop in tail for stop in stop_strings):
                        stopped_rows.append(row)
                finished[stopped_rows] = True
            yield next_token
            if step == max_new_tokens - 1 or bool(finished.all()):
                return

            all_ids = torch.cat([all_ids, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(batch_size, 1)], dim=1)
            if kv_cache is None:
                logits = forward(all_ids[:, -max_len:], attention_mask[:, -max_len:], None)
            elif kv_cache.length >= kv_cache.max_length:
                # Sliding the window would change every RoPE position, so recompute the cache.
                kv_cache = KVCache(self.config.n_layers, max_len)
                logits = forward(all_ids[:, -(max_len - 1) :], attention_mask[:, -(max_len - 1) :], kv_cache)
            else:
                logits = forward(next_token, attention_mask, kv_cache)


def sample_next_token(
    logits: torch.Tensor,
    previous_ids: torch.Tensor,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    repetition_penalty: float,
) -> torch.Tensor:
    if repetition_penalty != 1.0:
        previous_logits = logits.gather(1, previous_ids)
        penalized = torch.where(
            previous_logits > 0, previous_logits / repetition_penalty, previous_logits * repetition_penalty
        )
        logits = logits.scatter(1, previous_ids, penalized)

    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature
    if top_k is not None and top_k > 0:
        kth_value = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
        logits = logits.masked_fill(logits < kth_value, -float("inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        # Drop tokens once the mass *before* them already exceeds top_p (always keeps the top token).
        remove_sorted = cumulative_probs - sorted_logits.softmax(dim=-1) > top_p
        remove_mask = remove_sorted.scatter(1, sorted_indices, remove_sorted)
        logits = logits.masked_fill(remove_mask, -float("inf"))
    return torch.multinomial(logits.softmax(dim=-1), num_samples=1)


def configure_optimizer(
    model: nn.Module, weight_decay: float, learning_rate: float, betas: tuple[float, float], device_type: str
) -> torch.optim.Optimizer:
    params = [param for param in model.parameters() if param.requires_grad]
    # Weight-decay matrices only; norms (1-D) are excluded.
    decay_params = [param for param in params if param.dim() >= 2]
    no_decay_params = [param for param in params if param.dim() < 2]
    groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, fused=device_type == "cuda")
