"""Export a MiniLLM checkpoint to GGUF for llama.cpp / Ollama / LM Studio.

Writes the llama architecture directly with the `gguf` package (pip install -e ".[gguf]"). llama.cpp's own
converter rejects custom BPE tokenizers it has no hash for; this exporter declares the closest built-in
pre-tokenizer ("llama-bpe"). It splits text exactly like MiniLLM except that MiniLLM isolates indentation
after a newline, so llama.cpp may tokenize leading code indentation slightly differently (still valid tokens).

Usage:
    python -m MiniLLM.GGUFExport --model checkpoints/sft/latest --tokenizer data/tokens/tokenizer \
        --out artifacts/minillm-sft-f16.gguf
    llama-cli -m artifacts/minillm-sft-f16.gguf -cnv          # chat (uses the embedded chat template)
    llama-quantize artifacts/minillm-sft-f16.gguf artifacts/minillm-sft-q8_0.gguf Q8_0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

SPECIAL_TOKEN_TYPE = 3  # gguf TokenType.CONTROL
NORMAL_TOKEN_TYPE = 1


def permute_for_llama_cpp(weights: np.ndarray, num_heads: int) -> np.ndarray:
    """HF "rotate_half" RoPE layout -> llama.cpp's interleaved layout (same as convert_hf_to_gguf.py)."""
    return (weights.reshape(num_heads, 2, weights.shape[0] // num_heads // 2, *weights.shape[1:])
            .swapaxes(1, 2).reshape(weights.shape))


def tensor_name(name: str) -> str | None:
    """MiniLLM state-dict name -> GGUF tensor name (None = skip)."""
    fixed = {"embed_tokens.weight": "token_embd.weight", "norm.weight": "output_norm.weight",
             "lm_head.weight": "output.weight"}
    if name in fixed:
        return fixed[name]
    parts = name.split(".")  # layers.<i>.<module>.<proj>.weight
    layer, module = parts[1], ".".join(parts[2:])
    per_layer = {
        "input_layernorm.weight": "attn_norm.weight",
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.o_proj.weight": "attn_output.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        "mlp.gate_proj.weight": "ffn_gate.weight",
        "mlp.up_proj.weight": "ffn_up.weight",
        "mlp.down_proj.weight": "ffn_down.weight",
    }
    return f"blk.{layer}.{per_layer[module]}" if module in per_layer else None


def load_vocab(tokenizer_dir: Path) -> tuple[list[str], list[int], list[str], dict]:
    tokenizer_json = json.loads((tokenizer_dir / "tokenizer.json").read_text())
    vocab: dict[str, int] = dict(tokenizer_json["model"]["vocab"])
    special_ids = set()
    for added in tokenizer_json.get("added_tokens", []):
        vocab[added["content"]] = added["id"]
        if added.get("special"):
            special_ids.add(added["id"])
    tokens = [""] * (max(vocab.values()) + 1)
    for token, token_id in vocab.items():
        tokens[token_id] = token
    token_types = [SPECIAL_TOKEN_TYPE if token_id in special_ids else NORMAL_TOKEN_TYPE
                   for token_id in range(len(tokens))]
    merges = [merge if isinstance(merge, str) else " ".join(merge) for merge in tokenizer_json["model"]["merges"]]
    config_path = tokenizer_dir / "tokenizer_config.json"
    tokenizer_config = json.loads(config_path.read_text()) if config_path.exists() else {}
    template_path = tokenizer_dir / "chat_template.jinja"
    if template_path.exists():
        tokenizer_config["chat_template"] = template_path.read_text()
    return tokens, token_types, merges, tokenizer_config


def export_gguf(model_path: str, tokenizer_dir: str, out_path: str, dtype: str = "f16") -> Path:
    from MiniLLM.HFExport import load_any_model

    return write_gguf(load_any_model(model_path, "cpu"), tokenizer_dir, out_path, dtype)


def write_gguf(model, tokenizer_dir: str | Path, out_path: str | Path, dtype: str = "f16") -> Path:
    import gguf

    config = model.config
    tokens, token_types, merges, tokenizer_config = load_vocab(Path(tokenizer_dir))
    if len(tokens) > config.vocab_size:
        raise ValueError(f"Tokenizer has {len(tokens)} tokens but the model only {config.vocab_size}")
    # Pad unused embedding rows so the vocab matches the embedding matrix.
    tokens += [f"[PAD{index}]" for index in range(len(tokens), config.vocab_size)]
    token_types += [gguf.TokenType.UNUSED] * (config.vocab_size - len(token_types))
    token_ids = {token: index for index, token in enumerate(tokens)}

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(out), "llama")
    writer.add_name(out.stem)
    writer.add_context_length(config.max_seq_len)
    writer.add_embedding_length(config.dim)
    writer.add_block_count(config.n_layers)
    writer.add_feed_forward_length(config.ffn_hidden_dim)
    writer.add_head_count(config.n_heads)
    writer.add_head_count_kv(config.n_kv_heads)
    writer.add_rope_dimension_count(config.dim // config.n_heads)
    writer.add_rope_freq_base(config.rope_theta)
    writer.add_layer_norm_rms_eps(config.norm_eps)
    writer.add_vocab_size(config.vocab_size)
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16 if dtype == "f16" else gguf.LlamaFileType.ALL_F32)

    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("llama-bpe")
    writer.add_token_list(tokens)
    writer.add_token_types(token_types)
    writer.add_token_merges(merges)
    eos_token = tokenizer_config.get("eos_token") or "<|endoftext|>"
    writer.add_eos_token_id(token_ids[eos_token])
    writer.add_bos_token_id(token_ids[tokenizer_config.get("bos_token") or eos_token])
    writer.add_add_bos_token(False)
    if tokenizer_config.get("pad_token") in token_ids:
        writer.add_pad_token_id(token_ids[tokenizer_config["pad_token"]])
    if "<|im_end|>" in token_ids:
        writer.add_eot_token_id(token_ids["<|im_end|>"])
    if tokenizer_config.get("chat_template"):
        writer.add_chat_template(tokenizer_config["chat_template"])

    for name, tensor in model.state_dict().items():
        gguf_name = tensor_name(name)
        if gguf_name is None or (gguf_name == "output.weight" and config.tie_embeddings):
            continue
        array = tensor.detach().to(torch.float32).numpy()
        if gguf_name.endswith("attn_q.weight"):
            array = permute_for_llama_cpp(array, config.n_heads)
        elif gguf_name.endswith("attn_k.weight"):
            array = permute_for_llama_cpp(array, config.n_kv_heads)
        # Norm vectors stay f32 (llama.cpp convention); matrices take the requested dtype.
        if dtype == "f16" and array.ndim > 1:
            array = array.astype(np.float16)
        writer.add_tensor(gguf_name, array)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="MiniLLM checkpoint dir, .../latest, or hub:<repo>:<subdir>")
    parser.add_argument("--tokenizer", default="data/tokens/tokenizer")
    parser.add_argument("--out", default="artifacts/minillm-f16.gguf")
    parser.add_argument("--dtype", choices=["f16", "f32"], default="f16")
    args = parser.parse_args()
    export_gguf(args.model, args.tokenizer, args.out, args.dtype)


if __name__ == "__main__":
    main()
