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

    cache = KVCache(TINY.n_layers, max_length=10)
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


def test_left_padded_batch_matches_single_prompts():
    model = make_model()
    prompts = [torch.randint(1, TINY.vocab_size, (length,)) for length in (3, 7, 5)]
    width = max(len(prompt) for prompt in prompts)
    padded = torch.zeros((len(prompts), width), dtype=torch.long)
    mask = torch.zeros((len(prompts), width), dtype=torch.long)
    for row, prompt in enumerate(prompts):
        padded[row, width - len(prompt) :] = prompt
        mask[row, width - len(prompt) :] = 1

    batched = model.generate(padded, max_new_tokens=12, temperature=0, attention_mask=mask)
    for row, prompt in enumerate(prompts):
        single = model.generate(prompt[None], max_new_tokens=12, temperature=0)
        assert torch.equal(batched[row, width:], single[0, len(prompt) :])
    # Padded forward logits at real positions match the unpadded forward.
    padded_logits, _ = model(padded, attention_mask=mask)
    plain_logits, _ = model(prompts[0][None])
    assert torch.allclose(padded_logits[0, width - 3 :], plain_logits[0], atol=1e-5)


class CharTokenizer:
    """Token id i decodes to chr(ord('a') + i % 26): enough to test stop strings without a real vocab."""

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(ord("a") + token_id % 26) for token_id in token_ids)


def test_stop_strings_and_eos_end_rows_independently():
    model = make_model()
    prompt = torch.randint(0, TINY.vocab_size, (2, 4))
    free_run = model.generate(prompt, max_new_tokens=10, temperature=0)[:, 4:]
    tokenizer = CharTokenizer()
    # Stop row 0 on the text of its own 3rd-4th tokens; stop row 1 on its 2nd token as EOS.
    stop_text = tokenizer.decode(free_run[0, 2:4].tolist())
    eos_id = int(free_run[1, 1])
    output = model.generate(prompt, max_new_tokens=10, temperature=0, stop_strings=[stop_text],
                            tokenizer=tokenizer, eos_token_ids=[eos_id], pad_token_id=0)[:, 4:]
    row0_end = next(i for i in range(10) if stop_text in tokenizer.decode(free_run[0, : i + 1].tolist())) + 1
    row1_end = next(i for i in range(10) if int(free_run[1, i]) == eos_id) + 1
    assert row0_end <= 4 and row1_end <= 2
    assert torch.equal(output[0, :row0_end], free_run[0, :row0_end])
    assert torch.equal(output[1, :row1_end], free_run[1, :row1_end])
    # Both rows stopped, so generation ended early; finished rows were filled with the pad id.
    assert output.shape[1] == max(row0_end, row1_end)
    assert (output[1, row1_end:] == 0).all()


def test_stream_generate_yields_the_same_tokens():
    model = make_model()
    prompt = torch.randint(0, TINY.vocab_size, (1, 5))
    streamed = torch.cat(list(model.stream_generate(prompt, max_new_tokens=8, temperature=0)), dim=1)
    assert torch.equal(streamed, model.generate(prompt, max_new_tokens=8, temperature=0)[:, 5:])


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
