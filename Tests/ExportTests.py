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
