# minillm

A ~110M-parameter Llama-style English and code model, trained from scratch on free GPUs (Kaggle 2x T4), with a FastAPI + JavaScript chat demo on HF Spaces.

| Phase | What | Where | Time |
|---|---|---|---|
| 1 | Setup: secrets, repos | your laptop + HF | 15 min |
| 2 | 32k BPE tokenizer + 2.6B filtered tokens | Kaggle CPU (`Step1PrepareData`) | 2-4 h |
| 3-4 | Pretrain 2.5B tokens (70% FineWeb-Edu, 30% Python/JS/SQL) | Kaggle 2x T4 (`Step2Pretrain`) | 15-20 h, 2 sessions |
| 5 | SFT, optional DPO, evals, export, Space | Kaggle 2x T4 (`Step3FinetuneDeploy`) | 2-3 h |

## Phase 1: setup

1. Accept the dataset terms for [bigcode/the-stack-dedup](https://huggingface.co/datasets/bigcode/the-stack-dedup), which is gated.
2. Create an HF **write** token and a W&B API key.
3. Push this repo to GitHub, then set `GITHUB_REPO` in the notebooks.
4. In each Kaggle notebook, add these secrets: `HF_TOKEN`, `HF_USERNAME`, `WANDB_API_KEY`. Turn Internet on.

For local development:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                                   # 15 tests: model, KV cache, export, filters, tokenizer, LoRA
python Scripts/MakeDebugData.py           # offline tiny dataset
python -m MiniLLM.Training.Pretrain --config Configs/SmokeTest.yaml
```

## What's inside

- **Model** ([MiniLLM/Model.py](MiniLLM/Model.py)): RoPE, RMSNorm, SwiGLU, SDPA attention (T4 falls back to the memory-efficient kernel), tied embeddings and a KV cache. Sampling supports temperature, top-p, top-k and a repetition penalty. Module names match HF `LlamaForCausalLM`, so export is a key rename that is verified numerically.
- **Tokenizer** ([MiniLLM/Data/Tokenizer.py](MiniLLM/Data/Tokenizer.py)): 32,768-entry byte-level BPE with a GPT-4-style split. Whitespace after a newline is isolated, so code indentation becomes its own token. ChatML special tokens are included.
- **Data** ([MiniLLM/Data/](MiniLLM/Data/)):
  - Streams FineWeb-Edu and the-stack-dedup (Python, JS, SQL).
  - Quality filters drop low-quality text and minified or generated code.
  - Files are dropped for non-permissive licenses or GPL headers, and for secrets (cloud keys, tokens, private keys, DB URLs).
  - Exact dedup.
  - Output is uint16 shards on the HF Hub.
- **Pretraining** ([MiniLLM/Training/Pretrain.py](MiniLLM/Training/Pretrain.py)):
  - DDP, fp16 with loss scaling, gradient accumulation and gradient checkpointing.
  - AdamW with warmup and cosine decay; gradient clipping.
  - Separate English and code validation loss and perplexity, logged to W&B.
  - Auto-stops at 11.25 h, uploads the checkpoint to the Hub, and resumes on the next run.
- **Fine-tuning**:
  - [SFT.py](MiniLLM/Training/SFT.py): loss on assistant tokens only; length-grouped batches; `sft.lora_enabled=true` for LoRA.
  - [DPO.py](MiniLLM/Training/DPO.py): DPO on UltraFeedback with a frozen reference model.
- **Evaluation** ([MiniLLM/Evaluation/](MiniLLM/Evaluation/)):
  - HellaSwag (acc_norm)
  - MBPP pass@1, executed in a subprocess
  - Perplexity
  - Your own prompts in [EvalPrompts/CustomPrompts.jsonl](EvalPrompts/CustomPrompts.jsonl)
- **Demo** ([ChatSpace/](ChatSpace/)): FastAPI server streaming tokens over server-sent events to a plain HTML/JS chat page, in a Docker Space on the free CPU tier. Deployed by [Scripts/DeploySpace.py](Scripts/DeploySpace.py).

## Commands

All settings live in [Configs/Run125M.yaml](Configs/Run125M.yaml). Override any of them with `section.key=value`.

```bash
torchrun --standalone --nproc_per_node=2 -m MiniLLM.Training.Pretrain --config Configs/Run125M.yaml
torchrun --standalone --nproc_per_node=2 -m MiniLLM.Training.SFT --config Configs/Run125M.yaml sft.lora_enabled=true
torchrun --standalone --nproc_per_node=2 -m MiniLLM.Training.DPO --config Configs/Run125M.yaml
python -m MiniLLM.HFExport --checkpoint checkpoints/sft/latest --push-to-hub $HF_USERNAME/minillm-125m-chat
python -m MiniLLM.Evaluation.MBPP --model checkpoints/sft/latest --chat
```

## Hub repos created

All three are private by default.

- `datasets/$HF_USERNAME/minillm-tokens`: token shards and the tokenizer
- `$HF_USERNAME/minillm-125m-ckpt`: resumable checkpoints (`pretrain/`, `sft/`, `dpo/`); history is squashed to save space
- `$HF_USERNAME/minillm-125m-chat`: the exported model the Space loads

## Expectations

At this size, expect short, mostly correct code snippets and simple explanations. Rough targets:

- HellaSwag acc_norm: about 0.30
- MBPP pass@1: a few percent

It will not reason like a large assistant.

## Troubleshooting

- **OOM on T4:** set `train.micro_batch_size=8`. Gradient accumulation adjusts automatically.
- **Loss spikes or NaN:** lower `train.max_lr` to `4e-4`. The fp16 loss scale is logged as `train/loss_scale`.
- **Testing DDP on a Mac:** `MINILLM_DEVICE=cpu torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29511 -m MiniLLM.Training.Pretrain --config Configs/SmokeTest.yaml`
