import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from MiniLLM.Config import ModelConfig
from MiniLLM.HFExport import build_hf_model, from_hf, verify_equivalence
from MiniLLM.Model import Transformer


def test_hf_llama_roundtrip(tokenizer_dir, tmp_path):
    torch.manual_seed(0)
    config = ModelConfig(vocab_size=640, dim=64, n_layers=2, n_heads=4, n_kv_heads=4, ffn_hidden_dim=128,
                         max_seq_len=64, gradient_checkpointing=False)
    model = Transformer(config).eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    hf_model = build_hf_model(model, tokenizer)
    assert verify_equivalence(model, hf_model, config.vocab_size) < 1e-3

    hf_model.save_pretrained(tmp_path)
    reloaded = from_hf(str(tmp_path)).eval()
    tokens = torch.randint(0, config.vocab_size, (1, 20))
    assert torch.allclose(model(tokens)[0], reloaded(tokens)[0], atol=1e-5)

    # HF generate (used by the Space) should agree with our KV-cached greedy decode.
    ours = model.generate(tokens, max_new_tokens=10, temperature=0)
    theirs = hf_model.generate(tokens, max_new_tokens=10, do_sample=False, eos_token_id=None,
                               attention_mask=torch.ones_like(tokens))
    assert torch.equal(ours, theirs)


def _random_gqa_model(vocab_size: int) -> Transformer:
    torch.manual_seed(0)
    config = ModelConfig(vocab_size=vocab_size, dim=64, n_layers=2, n_heads=8, n_kv_heads=2, ffn_hidden_dim=128,
                         max_seq_len=64, gradient_checkpointing=False)
    model = Transformer(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.mul_(4)  # sharper logits give a non-degenerate greedy path
    return model


def test_gguf_export_writes_llama_tensors_and_vocab(tokenizer_dir, tmp_path):
    gguf = pytest.importorskip("gguf")
    from MiniLLM.GGUFExport import permute_for_llama_cpp, write_gguf

    model = _random_gqa_model(vocab_size=640)
    out = write_gguf(model, tokenizer_dir, tmp_path / "model.gguf", dtype="f32")
    reader = gguf.GGUFReader(str(out))
    names = {tensor.name for tensor in reader.tensors}
    assert {"token_embd.weight", "output_norm.weight", "blk.1.attn_k.weight", "blk.1.ffn_down.weight"} <= names
    assert "output.weight" not in names  # tied embeddings are not duplicated
    assert len(reader.fields["tokenizer.ggml.tokens"].data) == 640

    key = model.state_dict()["layers.0.self_attn.k_proj.weight"].numpy()
    stored = next(tensor for tensor in reader.tensors if tensor.name == "blk.0.attn_k.weight")
    assert np.array_equal(np.asarray(stored.data).reshape(key.shape), permute_for_llama_cpp(key, 2))


def test_gguf_greedy_matches_llama_cpp(tmp_path):
    """End-to-end check against a real llama.cpp build (skipped when llama-completion is not installed)."""
    binary = shutil.which("llama-completion")
    tokenizer_dir = Path("data/tokens_debug/tokenizer")
    if binary is None or not tokenizer_dir.exists():
        pytest.skip("needs llama-completion on PATH and data/tokens_debug (Scripts/MakeDebugData.py)")
    pytest.importorskip("gguf")
    from MiniLLM.GGUFExport import write_gguf

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = _random_gqa_model(vocab_size=32768)
    out = write_gguf(model, tokenizer_dir, tmp_path / "model.gguf", dtype="f32")
    prompt = "The students learned that"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ours = model.generate(torch.tensor([prompt_ids]), max_new_tokens=8, temperature=0)[0, len(prompt_ids):]
    result = subprocess.run(
        [binary, "-m", str(out), "-p", prompt, "-n", "8", "--temp", "0", "--top-k", "1", "-no-cnv",
         "--no-warmup", "-lv", "0"], capture_output=True, text=True, timeout=120,
    )
    assert prompt + tokenizer.decode(ours.tolist()) in result.stdout
