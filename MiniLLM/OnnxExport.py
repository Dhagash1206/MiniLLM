"""Export an HF-format minillm model to ONNX (fp32 + int8) in the layout Transformers.js loads.

The graph takes `input_ids`, `attention_mask`, `position_ids` and `past_key_values.{i}.key/value`
and returns `logits` and `present.{i}.key/value`, matching what Transformers.js feeds a Llama
decoder, so the browser demo gets KV-cached generation.

Usage (needs `pip install -e ".[web]"`; onnxruntime has no Python 3.15 wheels yet):
    python -m MiniLLM.OnnxExport --model artifacts/hf_model --push-to-hub $HF_USERNAME/minillm-125m-chat
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from MiniLLM.HFExport import from_hf
from MiniLLM.Model import Transformer, apply_rope


class OnnxDecoder(nn.Module):
    """Re-wires the trained Transformer's weights into an explicit past-key-value interface."""

    def __init__(self, model: Transformer):
        super().__init__()
        self.model = model
        self.config = model.config

    def forward(self, input_ids, attention_mask, position_ids, *past_key_values):
        model, config = self.model, self.config
        batch_size, query_length = input_ids.shape
        past_length = past_key_values[0].shape[2]
        total_length = past_length + query_length

        cos = model.rope_cos[position_ids].unsqueeze(1)  # [batch, 1, query, head_dim]
        sin = model.rope_sin[position_ids].unsqueeze(1)
        query_index = torch.arange(query_length, device=input_ids.device) + past_length
        key_index = torch.arange(total_length, device=input_ids.device)
        allowed = (key_index[None, :] <= query_index[:, None])[None, None] & attention_mask[:, None, None, :].bool()
        # Additive mask with a finite minimum keeps fully-masked padding rows from producing NaN.
        attn_bias = torch.where(allowed, 0.0, torch.finfo(torch.float32).min)

        hidden = model.embed_tokens(input_ids)
        presents = []
        for layer_index, layer in enumerate(model.layers):
            attention = layer.self_attn
            normed = layer.input_layernorm(hidden)
            query = attention.q_proj(normed).view(batch_size, query_length, config.n_heads, config.head_dim)
            key = attention.k_proj(normed).view(batch_size, query_length, config.n_kv_heads, config.head_dim)
            value = attention.v_proj(normed).view(batch_size, query_length, config.n_kv_heads, config.head_dim)
            query = apply_rope(query.transpose(1, 2), cos, sin)
            key = apply_rope(key.transpose(1, 2), cos, sin)
            value = value.transpose(1, 2)

            key = torch.cat([past_key_values[2 * layer_index], key], dim=2)
            value = torch.cat([past_key_values[2 * layer_index + 1], value], dim=2)
            presents.extend([key, value])
            if config.n_kv_heads != config.n_heads:
                repeat = config.n_heads // config.n_kv_heads
                key = key.repeat_interleave(repeat, dim=1)
                value = value.repeat_interleave(repeat, dim=1)

            attn_out = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_bias)
            attn_out = attn_out.transpose(1, 2).reshape(batch_size, query_length, -1)
            hidden = hidden + attention.o_proj(attn_out)
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        logits = model.lm_head(model.norm(hidden))
        return (logits, *presents)


def io_names(n_layers: int) -> tuple[list[str], list[str]]:
    past_names = [f"past_key_values.{i}.{kind}" for i in range(n_layers) for kind in ("key", "value")]
    present_names = [f"present.{i}.{kind}" for i in range(n_layers) for kind in ("key", "value")]
    return ["input_ids", "attention_mask", "position_ids", *past_names], ["logits", *present_names]


def example_inputs(config, batch_size: int, query_length: int, past_length: int):
    input_ids = torch.randint(0, config.vocab_size, (batch_size, query_length))
    attention_mask = torch.ones(batch_size, past_length + query_length, dtype=torch.long)
    position_ids = torch.arange(past_length, past_length + query_length).expand(batch_size, -1)
    past = [torch.randn(batch_size, config.n_kv_heads, past_length, config.head_dim)
            for _ in range(2 * config.n_layers)]
    return (input_ids, attention_mask, position_ids, *past)


def export_onnx(model: Transformer, onnx_path: Path) -> None:
    decoder = OnnxDecoder(model).eval()
    input_names, output_names = io_names(model.config.n_layers)
    dynamic_axes = {"input_ids": {0: "batch", 1: "sequence"},
                    "attention_mask": {0: "batch", 1: "total_sequence"},
                    "position_ids": {0: "batch", 1: "sequence"},
                    "logits": {0: "batch", 1: "sequence"}}
    for name in input_names[3:]:
        dynamic_axes[name] = {0: "batch", 2: "past_sequence"}
    for name in output_names[1:]:
        dynamic_axes[name] = {0: "batch", 2: "total_sequence"}
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    # Sizes > 1 in the example avoid the exporter specialising batch/sequence/past to constants.
    with torch.no_grad():
        torch.onnx.export(
            decoder, example_inputs(model.config, 2, 3, 4), str(onnx_path),
            input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes,
            opset_version=18, dynamo=False,
        )


def quantize_int8(fp32_path: Path, int8_path: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gather"])


@torch.no_grad()
def verify(model: Transformer, onnx_path: Path, tolerance: float) -> float:
    """Compare ONNX vs PyTorch on a prefill with an empty cache, then one cached decoding step."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    config = model.config
    tokens = torch.randint(0, config.vocab_size, (1, 9))
    reference, _ = model(tokens)

    empty_past = [np.zeros((1, config.n_kv_heads, 0, config.head_dim), dtype=np.float32)
                  for _ in range(2 * config.n_layers)]
    input_names, _ = io_names(config.n_layers)
    prefill = session.run(None, dict(zip(input_names, [
        tokens[:, :8].numpy(), np.ones((1, 8), dtype=np.int64), np.arange(8)[None].astype(np.int64), *empty_past])))
    step = session.run(None, dict(zip(input_names, [
        tokens[:, 8:].numpy(), np.ones((1, 9), dtype=np.int64), np.array([[8]], dtype=np.int64), *prefill[1:]])))
    onnx_logits = np.concatenate([prefill[0], step[0]], axis=1)
    max_diff = float(np.abs(onnx_logits - reference.numpy()).max())
    if max_diff > tolerance:
        raise RuntimeError(f"{onnx_path.name} diverges from PyTorch (max logit diff {max_diff:.2e})")
    return max_diff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Exported HF folder (from MiniLLM.HFExport) or Hub repo id")
    parser.add_argument("--out", default=None, help="Folder to write onnx/ into (default: --model if local)")
    parser.add_argument("--push-to-hub", default=None, help="Model repo to upload onnx/ into")
    args = parser.parse_args()

    model = from_hf(args.model).eval()
    model.config.gradient_checkpointing = False
    out_dir = Path(args.out or args.model)
    fp32_path = out_dir / "onnx" / "model.onnx"
    int8_path = out_dir / "onnx" / "model_quantized.onnx"

    export_onnx(model, fp32_path)
    print(f"fp32 ONNX matches PyTorch (max logit diff {verify(model, fp32_path, 1e-3):.2e})")
    quantize_int8(fp32_path, int8_path)
    # int8 drifts from fp32 by design; only catch outright breakage here.
    print(f"int8 ONNX max logit diff {verify(model, int8_path, 5.0):.2e}")
    for path in (fp32_path, int8_path):
        print(f"{path}: {path.stat().st_size / 1e6:.0f} MB")

    updated_files = prepare_web_configs(out_dir)

    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_folder(folder_path=str(out_dir / "onnx"), path_in_repo="onnx", repo_id=args.push_to_hub)
        for path in updated_files:
            api.upload_file(path_or_fileobj=str(path), path_in_repo=path.name, repo_id=args.push_to_hub)
        print(f"Uploaded ONNX files to https://huggingface.co/{args.push_to_hub}")


def prepare_web_configs(out_dir: Path) -> list[Path]:
    """Adjust the HF configs for Transformers.js; returns the files that changed."""
    updated = []
    config_path = out_dir / "config.json"
    if config_path.exists():
        hf_config = json.loads(config_path.read_text())
        # Tells Transformers.js which ONNX file to load by default.
        hf_config["transformers.js_config"] = {"dtype": "q8"}
        config_path.write_text(json.dumps(hf_config, indent=2))
        updated.append(config_path)
    template_path = out_dir / "chat_template.jinja"
    tokenizer_config_path = out_dir / "tokenizer_config.json"
    if template_path.exists() and tokenizer_config_path.exists():
        # transformers 5 saves the template to its own file, which Transformers.js does not read.
        tokenizer_config = json.loads(tokenizer_config_path.read_text())
        tokenizer_config["chat_template"] = template_path.read_text()
        tokenizer_config_path.write_text(json.dumps(tokenizer_config, indent=2))
        updated.append(tokenizer_config_path)
    return updated


if __name__ == "__main__":
    main()
