import torch

from MiniLLM.Config import ModelConfig
from MiniLLM.LoRA import apply_lora, merged_state_dict
from MiniLLM.Model import KVCache, Transformer, sample_next_token

TINY = ModelConfig(vocab_size=97, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, ffn_hidden_dim=128,
                   max_seq_len=32, gradient_checkpointing=False)


def make_model(seed: int = 0) -> Transformer:
    torch.manual_seed(seed)
    return Transformer(TINY).eval()


def test_forward_shapes_and_loss():
    model = make_model()
    tokens = torch.randint(0, TINY.vocab_size, (3, 16))
    targets = torch.randint(0, TINY.vocab_size, (3, 16))
    logits, loss = model(tokens, targets)
    assert logits.shape == (3, 16, TINY.vocab_size)
    # Fresh init should be close to uniform: loss ~ ln(vocab).
    assert abs(loss.item() - torch.log(torch.tensor(97.0)).item()) < 0.5


def test_gradient_checkpointing_matches_plain_backward():
    tokens = torch.randint(0, TINY.vocab_size, (2, 12))
    grads = []
    for use_checkpointing in (False, True):
        model = make_model().train()
        model.config.gradient_checkpointing = use_checkpointing
        _, loss = model(tokens, tokens)
        loss.backward()
        grads.append(model.layers[0].mlp.up_proj.weight.grad.clone())
        model.config.gradient_checkpointing = False
    assert torch.allclose(grads[0], grads[1], atol=1e-6)


def test_kv_cache_matches_full_forward():
    model = make_model()
    tokens = torch.randint(0, TINY.vocab_size, (2, 10))
    full_logits, _ = model(tokens)

    cache = KVCache(TINY.n_layers)
    prefix_logits, _ = model(tokens[:, :4], kv_cache=cache)
    chunk_logits, _ = model(tokens[:, 4:7], kv_cache=cache)  # chunked prefill path (explicit mask)
    step_logits = [model(tokens[:, index : index + 1], kv_cache=cache)[0] for index in range(7, 10)]
    cached = torch.cat([prefix_logits, chunk_logits, *step_logits], dim=1)
    assert torch.allclose(full_logits, cached, atol=1e-5)


def test_greedy_generate_same_with_and_without_cache():
    model = make_model()
    prompt = torch.randint(0, TINY.vocab_size, (2, 5))
    with_cache = model.generate(prompt, max_new_tokens=40, temperature=0, use_cache=True)
    without_cache = model.generate(prompt, max_new_tokens=40, temperature=0, use_cache=False)
    # 45 tokens > max_seq_len=32 also exercises the cache-rebuild path.
    assert torch.equal(with_cache, without_cache)


def test_top_p_keeps_only_nucleus():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
    previous = torch.zeros((1, 1), dtype=torch.long)
    samples = {int(sample_next_token(logits, previous, 1.0, 0.7, None, 1.0)) for _ in range(300)}
    assert samples <= {0, 1}


def test_lora_merge_matches_adapter_forward():
    model = make_model()
    apply_lora(model, rank=4, alpha=8, dropout=0.0)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(std=0.05)
    tokens = torch.randint(0, TINY.vocab_size, (2, 8))
    adapter_logits, _ = model(tokens)

    merged_model = Transformer(TINY).eval()
    merged_model.load_state_dict(merged_state_dict(model))
    merged_logits, _ = merged_model(tokens)
    assert torch.allclose(adapter_logits, merged_logits, atol=1e-5)
