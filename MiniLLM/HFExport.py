"""Convert between minillm checkpoints and Hugging Face `LlamaForCausalLM` folders.

Usage:
    python -m MiniLLM.HFExport --checkpoint checkpoints/sft/latest --tokenizer data/tokens/tokenizer \
        --out artifacts/hf_model --push-to-hub $HF_USERNAME/minillm-125m-chat
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from MiniLLM.Config import ModelConfig
from MiniLLM.Model import Transformer
from MiniLLM.Training.TrainingUtils import load_model_weights, resolve_checkpoint


def to_hf_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # Module names already match HF Llama; only the `model.` prefix differs.
    return {(name if name.startswith("lm_head") else f"model.{name}"): tensor for name, tensor in state_dict.items()}


def build_hf_model(model: Transformer, tokenizer):
    from transformers import LlamaConfig, LlamaForCausalLM

    config = model.config
    hf_config = LlamaConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.dim,
        intermediate_size=config.ffn_hidden_dim,
        num_hidden_layers=config.n_layers,
        num_attention_heads=config.n_heads,
        num_key_value_heads=config.n_kv_heads,
        max_position_embeddings=config.max_seq_len,
        rms_norm_eps=config.norm_eps,
        rope_parameters={"rope_type": "default", "rope_theta": config.rope_theta},
        tie_word_embeddings=config.tie_embeddings,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")],
        pad_token_id=tokenizer.pad_token_id,
        dtype=torch.float32,
    )
    hf_model = LlamaForCausalLM(hf_config)
    state_dict = to_hf_state_dict({k: v for k, v in model.state_dict().items()})
    missing, unexpected = hf_model.load_state_dict(state_dict, strict=False)
    # The only tolerated mismatch is the tied lm_head when HF does not store it separately.
    if [key for key in missing if key != "lm_head.weight"] or unexpected:
        raise RuntimeError(f"State dict mismatch. missing={missing} unexpected={unexpected}")
    if config.tie_embeddings:
        hf_model.tie_weights()
    hf_model.generation_config.do_sample = True
    hf_model.generation_config.temperature = 0.7
    hf_model.generation_config.top_p = 0.9
    hf_model.generation_config.max_new_tokens = 512
    return hf_model


@torch.no_grad()
def verify_equivalence(model: Transformer, hf_model, vocab_size: int) -> float:
    model.eval()
    hf_model.eval()
    sample_ids = torch.randint(0, vocab_size, (2, 64))
    ours, _ = model(sample_ids)
    theirs = hf_model(input_ids=sample_ids).logits
    max_diff = (ours - theirs).abs().max().item()
    if max_diff > 1e-3:
        raise RuntimeError(f"Exported model diverges from minillm model (max logit diff {max_diff:.2e})")
    return max_diff


def from_hf(model_dir_or_repo: str) -> Transformer:
    """Load an exported HF Llama folder/repo back into a minillm Transformer (for evals)."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    local_dir = Path(model_dir_or_repo)
    if not local_dir.exists():
        local_dir = Path(snapshot_download(model_dir_or_repo, allow_patterns=["*.json", "*.safetensors"]))
    hf_config = json.loads((local_dir / "config.json").read_text())
    rope_parameters = hf_config.get("rope_parameters") or {}
    config = ModelConfig(
        vocab_size=hf_config["vocab_size"], dim=hf_config["hidden_size"], n_layers=hf_config["num_hidden_layers"],
        n_heads=hf_config["num_attention_heads"], n_kv_heads=hf_config["num_key_value_heads"],
        ffn_hidden_dim=hf_config["intermediate_size"], max_seq_len=hf_config["max_position_embeddings"],
        rope_theta=rope_parameters.get("rope_theta", hf_config.get("rope_theta", 10000.0)),
        norm_eps=hf_config["rms_norm_eps"], tie_embeddings=hf_config.get("tie_word_embeddings", True),
        gradient_checkpointing=False,
    )
    state_dict = {}
    for shard in sorted(local_dir.glob("*.safetensors")):
        state_dict.update(load_file(str(shard)))
    renamed = {name.removeprefix("model."): tensor for name, tensor in state_dict.items()}
    if config.tie_embeddings:
        renamed["lm_head.weight"] = renamed["embed_tokens.weight"]
    model = Transformer(config)
    model.load_state_dict(renamed)
    return model


def load_any_model(path: str, device: str | torch.device = "cpu") -> Transformer:
    """Accepts a minillm checkpoint dir, `hub:<repo>:<subdir>`, or an HF Llama folder / repo id."""
    # `.../latest` (local or hub:) resolves to the newest step-XXXXX checkpoint folder.
    candidate = resolve_checkpoint(path) if path.startswith("hub:") or Path(path).name == "latest" else Path(path)
    if (candidate / "model.pt").exists():
        model, _ = load_model_weights(candidate, device)
    else:
        model = from_hf(path).to(device)
    model.config.gradient_checkpointing = False
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Local dir or hub:<repo>:<subdir>")
    parser.add_argument("--tokenizer", default="data/tokens/tokenizer")
    parser.add_argument("--out", default="artifacts/hf_model")
    parser.add_argument("--push-to-hub", default=None)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    model, _ = load_model_weights(resolve_checkpoint(args.checkpoint), "cpu")
    model.config.gradient_checkpointing = False
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    hf_model = build_hf_model(model, tokenizer)
    max_diff = verify_equivalence(model, hf_model, model.config.vocab_size)
    print(f"Equivalence check passed (max logit diff {max_diff:.2e})")

    hf_model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Saved HF model to {args.out}")
    if args.push_to_hub:
        hf_model.push_to_hub(args.push_to_hub, private=args.private)
        tokenizer.push_to_hub(args.push_to_hub, private=args.private)
        print(f"Pushed to https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()
