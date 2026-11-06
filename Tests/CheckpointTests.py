import torch

from MiniLLM.Config import ModelConfig, RunConfig
from MiniLLM.Data.ChatData import row_to_messages
from MiniLLM.Model import Transformer
from MiniLLM.Training.TrainingUtils import (
    CheckpointManager,
    LossHealthGuard,
    load_model_weights,
    resolve_checkpoint,
)


def test_health_guard_blocks_saves_after_nan_and_spikes():
    guard = LossHealthGuard(spike_threshold=0.5, cooldown_steps=10)
    for step in range(1, 51):
        assert guard.update(step, 3.0 - step * 0.01)  # steadily falling loss is healthy
    assert guard.safe_to_save(50)
    assert not guard.update(51, float("nan"))
    assert not guard.safe_to_save(55) and guard.safe_to_save(62)
    assert not guard.update(70, 9.0)  # sudden spike
    assert not guard.safe_to_save(75)


def test_checkpoints_keep_newest_and_write_fp16_milestones(tmp_path):
    config = RunConfig()
    config.model = ModelConfig(vocab_size=64, dim=32, n_layers=1, n_heads=2, n_kv_heads=2, ffn_hidden_dim=64,
                               max_seq_len=16)
    model = Transformer(config.model)
    optimizer = torch.optim.AdamW(model.parameters())
    manager = CheckpointManager(tmp_path, hub_repo=None, push=False, private=True, squash=False,
                                path_in_repo="pretrain", keep_last=2, milestone_interval=40)
    for step in (20, 40, 60, 70):
        assert manager.save(model, optimizer, None, {"step": step, "total_steps": 70}, config)

    assert sorted(path.name for path in tmp_path.glob("step-*")) == ["step-00060", "step-00070"]
    milestones = sorted(path.name for path in (tmp_path / "milestones").iterdir())
    assert milestones == ["step-00040-fp16", "step-00070-fp16"]  # interval hit + final step
    milestone_weights = torch.load(tmp_path / "milestones" / "step-00040-fp16" / "model.pt")
    assert all(tensor.dtype == torch.float16 for tensor in milestone_weights.values())

    assert manager.find_resume_dir() == tmp_path / "step-00070"
    assert resolve_checkpoint(str(tmp_path / "latest")) == tmp_path / "step-00070"
    reloaded, _ = load_model_weights(tmp_path / "milestones" / "step-00040-fp16")
    assert reloaded.embed_tokens.weight.dtype == torch.float32


def test_row_to_messages_handles_common_layouts():
    assert row_to_messages({"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]})[1] \
        == {"role": "assistant", "content": "b"}
    assert row_to_messages({"problem": "p", "solution": "s"}) == [
        {"role": "user", "content": "p"}, {"role": "assistant", "content": "s"}]
    assert row_to_messages({"conversations": [{"from": "human", "value": "q"}]}) == [{"role": "human", "content": "q"}]
    assert row_to_messages({"instruction": "i", "input": "x", "output": "o"})[0]["content"] == "i\n\nx"
    assert row_to_messages({"text": "plain"}) is None
