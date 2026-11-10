"""Shared eval plumbing: device choice, model/tokenizer loading (MiniLLM or HF baselines), code execution.

Model paths:
    checkpoints/pretrain/latest              MiniLLM checkpoint (or hub:<repo>:<subdir>, or an HF Llama export)
    hf:openai-community/gpt2                 any HF causal LM, wrapped so every eval treats it like MiniLLM
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

HF_PREFIX = "hf:"
# Generated programs only learn from tests that finish quickly; slow ones count as failures.
CODE_TIMEOUT_SECONDS = 10.0
# Base-model code completions end where truncate_completion would cut them anyway; stopping there saves compute.
CODE_STOP_STRINGS = ["\nif __name__", "\nprint(", "\nassert ", '\n"""', "\n```"]
# Generous for MBPP/HumanEval tests, small enough that a runaway program cannot take the machine down.
SANDBOX_MEMORY_BYTES = 2 * 1024**3
SANDBOX_FILE_BYTES = 16 * 1024**2
SANDBOX_OPEN_FILES = 64


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if device.type == "mps":
        return torch.autocast(device_type="mps", dtype=torch.bfloat16)
    return contextlib.nullcontext()


class HFCausalLMAdapter(torch.nn.Module):
    """Gives an HF causal LM MiniLLM's interface: forward -> (logits, loss), generate(), config.max_seq_len."""

    def __init__(self, hf_model, max_seq_len: int):
        super().__init__()
        self.hf_model = hf_model
        self.config = SimpleNamespace(max_seq_len=max_seq_len)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None):
        logits = self.hf_model(input_ids=input_ids).logits
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 0.8,
                 top_p: float | None = 0.95, top_k: int | None = None, repetition_penalty: float = 1.0,
                 eos_token_ids: list[int] | None = None, use_cache: bool = True,
                 attention_mask: torch.Tensor | None = None, stop_strings: list[str] | None = None,
                 tokenizer=None, pad_token_id: int | None = None) -> torch.Tensor:
        do_sample = temperature > 0
        eos_ids = [token_id for token_id in (eos_token_ids or []) if token_id is not None]
        if pad_token_id is None:
            pad_token_id = eos_ids[0] if eos_ids else 0
        stop_kwargs = {"stop_strings": stop_strings, "tokenizer": tokenizer} if stop_strings else {}
        return self.hf_model.generate(
            input_ids=input_ids, attention_mask=attention_mask if attention_mask is not None else
            torch.ones_like(input_ids), max_new_tokens=max_new_tokens,
            do_sample=do_sample, temperature=temperature if do_sample else None, top_p=top_p if do_sample else None,
            top_k=top_k if do_sample else None, repetition_penalty=repetition_penalty,
            eos_token_id=eos_ids or None, pad_token_id=pad_token_id, use_cache=use_cache, **stop_kwargs,
        )


def load_eval_model(path: str, device: torch.device, max_seq_len: int = 2048):
    """MiniLLM checkpoints load natively; `hf:<repo>` loads an HF causal LM behind the same interface."""
    if path.startswith(HF_PREFIX):
        from transformers import AutoModelForCausalLM

        hf_model = AutoModelForCausalLM.from_pretrained(path[len(HF_PREFIX):], dtype=torch.float32).to(device).eval()
        context = getattr(hf_model.config, "max_position_embeddings", None) or getattr(hf_model.config, "n_positions")
        # Cap long-context baselines so prompts are truncated the same way as for MiniLLM.
        return HFCausalLMAdapter(hf_model, min(int(context), max_seq_len))
    from MiniLLM.HFExport import load_any_model

    return load_any_model(path, device)


def load_eval_tokenizer(model_path: str, tokenizer_dir: str | None):
    """Baselines default to their own tokenizer; MiniLLM models need --tokenizer."""
    from transformers import AutoTokenizer

    if tokenizer_dir is None and model_path.startswith(HF_PREFIX):
        tokenizer_dir = model_path[len(HF_PREFIX):]
    if tokenizer_dir is None:
        raise ValueError("--tokenizer is required for MiniLLM checkpoints")
    return AutoTokenizer.from_pretrained(tokenizer_dir)


def truncate_completion(completion: str) -> str:
    """Keep the function body: stop at the first top-level line that does not start a new definition."""
    kept = []
    for line in completion.splitlines():
        if line and not line[0].isspace() and not line.startswith(("def ", "class ", "import ", "from ", "#", "@")):
            break
        kept.append(line)
    return "\n".join(kept)


def limit_child_resources(cpu_seconds: int) -> None:
    """Runs in the forked child before exec: caps CPU, memory, file writes, open files and new processes."""
    import resource

    limits = [
        (resource.RLIMIT_CPU, cpu_seconds),
        (resource.RLIMIT_AS, SANDBOX_MEMORY_BYTES),
        (resource.RLIMIT_FSIZE, SANDBOX_FILE_BYTES),
        (resource.RLIMIT_NOFILE, SANDBOX_OPEN_FILES),
        (resource.RLIMIT_NPROC, 0),
        (resource.RLIMIT_CORE, 0),
    ]
    for limit, value in limits:
        try:
            resource.setrlimit(limit, (value, value))
        except (ValueError, OSError):
            pass  # macOS rejects RLIMIT_AS; the other caps still apply


def run_python(program: str, timeout_seconds: float = CODE_TIMEOUT_SECONDS) -> bool:
    """Runs untrusted generated code in a resource-capped interpreter with an empty environment.

    This blocks fork bombs, memory and disk exhaustion, runaway loops and leaked secrets (HF_TOKEN etc.),
    but not network or file access, so still prefer a disposable machine (Kaggle/Colab) for code evals.
    """
    with tempfile.TemporaryDirectory() as work_dir:
        script = Path(work_dir) / "candidate.py"
        script.write_text(program)
        try:
            # -X utf8: with no LANG in the empty environment, printing non-ASCII must not fail the program.
            result = subprocess.run(
                [sys.executable, "-I", "-X", "utf8", str(script)], cwd=work_dir, env={}, capture_output=True,
                timeout=timeout_seconds, preexec_fn=lambda: limit_child_resources(int(timeout_seconds) + 1),
            )
        except subprocess.TimeoutExpired:
            return False
    return result.returncode == 0
