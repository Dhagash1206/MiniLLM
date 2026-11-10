# minillm

A ~110M-parameter Llama-style English and code model, trained from scratch on free GPUs (Kaggle 2x T4), with a chat demo that runs in the visitor's browser (Transformers.js) on a free static HF Space.

| Phase | What | Where | Time |
|---|---|---|---|
| 1 | Setup: secrets, repos | your laptop + HF | 15 min |
| 2 | 32k BPE tokenizer + 2.6B filtered tokens | Kaggle CPU (`Step1PrepareData`) | 2-4 h |
| 3-4 | Pretrain 2.5B tokens (70% FineWeb-Edu, 30% Python/JS/SQL) | Kaggle 2x T4 (`Step2Pretrain`) | 15-20 h, 2 sessions |
| 5 | SFT, optional DPO, evals, export, Space | Kaggle 2x T4 (`Step3FinetuneDeploy`) | 2-3 h |

## Phase 1: setup

1. Accept the dataset terms for [bigcode/the-stack-dedup](https://huggingface.co/datasets/bigcode/the-stack-dedup), which is gated.
2. Create an HF **write** token and a W&B API key.
3. The notebooks clone https://github.com/Dhagash1206/MiniLLM. If you fork it, change `GITHUB_REPO` in each notebook.
4. In each Kaggle notebook, add these secrets: `HF_TOKEN`, `HF_USERNAME`, `WANDB_API_KEY`. Turn Internet on.

For local development:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                                   # 29 tests: model, KV cache, batching, HF/GGUF export, checkpoints, data, evals
python Scripts/MakeDebugData.py           # offline tiny dataset
python -m MiniLLM.Training.Pretrain --config Configs/SmokeTest.yaml
```

## What's inside

- **Model** ([MiniLLM/Model.py](MiniLLM/Model.py)): RoPE, RMSNorm, SwiGLU, SDPA attention (T4 falls back to the memory-efficient kernel), tied embeddings and a preallocated KV cache (written in place, no `torch.cat` per token). `generate` takes left-padded batches through an `attention_mask` (RoPE positions follow the mask), stops rows on EOS ids or `stop_strings`, and `stream_generate` yields tokens as they are sampled. Sampling supports temperature, top-p, top-k and a repetition penalty. Module names match HF `LlamaForCausalLM`, so export is a key rename that is verified numerically.
- **Tokenizer** ([MiniLLM/Data/Tokenizer.py](MiniLLM/Data/Tokenizer.py)): 32,768-entry byte-level BPE with a GPT-4-style split. Whitespace after a newline is isolated, so code indentation becomes its own token. ChatML special tokens are included.
- **Data** ([MiniLLM/Data/](MiniLLM/Data/)):
  - Streams FineWeb-Edu and the-stack-dedup (Python, JS, SQL).
  - Quality filters drop low-quality text and minified or generated code.
  - Files are dropped for non-permissive licenses or GPL headers, and for secrets (cloud keys, tokens, private keys, DB URLs).
  - Exact dedup; optional MinHash near-dedup (`PrepareShards --near-dedup`, Jaccard ~0.7).
  - Output is uint16 shards on the HF Hub.
- **Pretraining** ([MiniLLM/Training/Pretrain.py](MiniLLM/Training/Pretrain.py)):
  - DDP, fp16 with loss scaling and gradient accumulation; peak GPU memory is logged. Gradient checkpointing is optional (off: it costs ~33% more compute).
  - AdamW with warmup and cosine decay; gradient clipping.
  - Separate English and code validation loss and perplexity, logged to W&B.
  - Auto-stops at 11.25 h and resumes on the next run from the Hub. Resume loads weights and optimizer state with `weights_only=True` (no pickle).
  - Windows are drawn in a seeded random order; from the second epoch on, each shard's window grid shifts by a random offset, so repeated data is cut at new boundaries.
  - Checkpoints: the newest 2 full `step-XXXXX/` folders, plus fp16 weights-only milestones every 1,000 steps (model versions). A loss health guard stops saving while the loss is NaN or spiking, so a diverging run never overwrites good checkpoints.
- **Fine-tuning**:
  - [SFT.py](MiniLLM/Training/SFT.py): 200k smol-smoltalk + Magicoder-OSS-Instruct (set the mix in `sft.datasets`); loss on assistant tokens only; length-grouped batches; `sft.lora_enabled=true` for LoRA. The same loss health guard as pretraining (`sft.loss_spike_threshold`) keeps a diverging run from overwriting good checkpoints.
  - [DPO.py](MiniLLM/Training/DPO.py): DPO on UltraFeedback with a frozen reference model.
- **Evaluation** ([MiniLLM/Evaluation/](MiniLLM/Evaluation/)):
  - HellaSwag (acc_norm), BLiMP (grammaticality, 67 minimal-pair sets)
  - MBPP and HumanEval pass@1 with batched greedy decoding. Model-written code runs in a subprocess with `resource.setrlimit` caps (CPU, memory, file size, open files, no new processes) and an empty environment, so tokens like `HF_TOKEN` never reach it. Network and file access are not blocked, so prefer a disposable machine.
  - Perplexity on the English and code validation shards
  - Your own prompts in [EvalPrompts/CustomPrompts.jsonl](EvalPrompts/CustomPrompts.jsonl)
  - [RunSuite.py](MiniLLM/Evaluation/RunSuite.py) runs all of them for one model; `hf:<repo>` models (GPT-2, SmolLM2) go through the same harness as baselines. [Report.py](MiniLLM/Evaluation/Report.py) turns the scores and training log into `RESULTS.md`, `MODEL_CARD.md` and a loss-curve SVG.
- **Demo** ([ChatSpace/](ChatSpace/)): a static Space (free for every account). [OnnxExport.py](MiniLLM/OnnxExport.py) exports the model to ONNX with a KV cache (fp32, plus int8 at ~135 MB), and Transformers.js runs it in a Web Worker in the visitor's browser: on WebGPU when available, otherwise int8 on WebAssembly (`?device=wasm` forces it). Deployed by [Scripts/DeploySpace.py](Scripts/DeploySpace.py).

## Commands

All settings live in [Configs/Run125M.yaml](Configs/Run125M.yaml). Override any of them with `section.key=value`.

```bash
torchrun --standalone --nproc_per_node=2 -m MiniLLM.Training.Pretrain --config Configs/Run125M.yaml
torchrun --standalone --nproc_per_node=2 -m MiniLLM.Training.SFT --config Configs/Run125M.yaml
torchrun --standalone --nproc_per_node=2 -m MiniLLM.Training.DPO --config Configs/Run125M.yaml
python -m MiniLLM.HFExport --checkpoint checkpoints/sft/latest --push-to-hub $HF_USERNAME/minillm-125m-chat
python -m MiniLLM.OnnxExport --model artifacts/hf_model --push-to-hub $HF_USERNAME/minillm-125m-chat
python Scripts/DeploySpace.py --space $HF_USERNAME/minillm-chat --model-repo $HF_USERNAME/minillm-125m-chat
python -m MiniLLM.Evaluation.MBPP --model checkpoints/sft/latest --chat
python -m MiniLLM.Training.Rollback --repo $HF_USERNAME/minillm-125m-ckpt --to step-03500   # dry run; add --yes
python -m MiniLLM.GGUFExport --model checkpoints/sft/latest --tokenizer data/tokens/tokenizer --out artifacts/minillm-f16.gguf
```

## Results, baselines and model card

```bash
python -m MiniLLM.Evaluation.RunSuite --name minillm-sft --model checkpoints/sft/latest \
    --tokenizer data/tokens/tokenizer --tokens-dir data/tokens --chat --allow-code-exec
Scripts/RunBaselines.sh --allow-code-exec    # GPT-2 small and SmolLM2-135M on the same harness
python -m MiniLLM.Evaluation.Report --pretrain-log <pretrain log> --config Configs/Run125M.yaml --model-name minillm-sft
```

The report lands in `results/report/` (`RESULTS.md`, `MODEL_CARD.md`, `loss_curve.svg`). No measured numbers are published yet; the targets under [Expectations](#expectations) are estimates until this has been run.

## Ablations

[Configs/Proxy30M.yaml](Configs/Proxy30M.yaml) is a ~27M-parameter proxy with a fixed 200M-token budget, so each variant fits in one Kaggle session. [Scripts/RunAblations.py](Scripts/RunAblations.py) trains it once per variant, changing one setting at a time, and compares the final validation loss in `results/ablations/SUMMARY.md`:

- data mix: 70/30 (natural, default) vs 50/50 vs 85/15 English/code (`data.domain_mix`)
- peak learning rate: 1e-3 (proxy default) vs 6e-4 vs 4e-4
- attention: full multi-head (6 KV heads) vs grouped-query (2 KV heads, the same 3:1 grouping as 12 -> 4 at full size)
- data: exact dedup vs MinHash near-dedup (shards built with `PrepareShards --near-dedup`)

```bash
python Scripts/RunAblations.py --wandb                 # all variants, resumable, one W&B project
python Scripts/RunAblations.py --only base gqa_kv2 --total-tokens 5e7
python Scripts/RunAblations.py --only near_dedup --dedup-tokens-dir data/tokens_neardedup
python Scripts/BenchmarkCompile.py --config Configs/Run125M.yaml   # train.compile=true vs eager, on a T4
```

The winning settings go in the table below once the sweep has run on a GPU; none are claimed yet.

| Setting | Variants | Winner |
|---|---|---|
| English/code mix | 70/30, 50/50, 85/15 | not run yet |
| Peak learning rate | 1e-3, 6e-4, 4e-4 | not run yet |
| KV heads | 6 vs 2 | not run yet |
| Near-dedup | off vs on | not run yet |
| `train.compile` on T4 | false vs true | not run yet |

## Running on a Mac (no GPU quota, no HF login)

`Scripts/PrepareDataLocal.sh` prepares the data locally. Without `hf auth login` it uses only English (FineWeb-Edu), because The Stack is gated. [Configs/RunLocalMac.yaml](Configs/RunLocalMac.yaml) trains a 42M model on 100M tokens with bf16 on Apple GPUs: about 3 h on an M5 (~9k tok/s).

```bash
TOTAL_TOKENS=1.6e8 Scripts/PrepareDataLocal.sh
Scripts/RunLocalMac.sh        # pretrain -> SFT -> eval suite + GPT-2 / SmolLM2 baselines -> results/report
ALLOW_CODE_EXEC=1 Scripts/RunLocalMac.sh   # also MBPP + HumanEval (runs model-written code in the sandbox)
```

Every stage resumes or is skipped when it is already done, so rerun the script after an interruption.

## Running locally with llama.cpp, Ollama or LM Studio

[GGUFExport.py](MiniLLM/GGUFExport.py) writes a llama-architecture GGUF (`pip install -e ".[gguf]"`). A test checks that llama.cpp's greedy output matches PyTorch token for token.

```bash
llama-cli -m artifacts/minillm-f16.gguf -cnv
llama-quantize artifacts/minillm-f16.gguf artifacts/minillm-q8_0.gguf Q8_0
```

llama.cpp has no exact equivalent of MiniLLM's split rule that isolates indentation, so leading code indentation may tokenize slightly differently there.

For LoRA instead of full fine-tuning, add `sft.lora_enabled=true` to the SFT command. `.../latest` always means the newest `step-XXXXX` folder.

## Hub repos created

- `datasets/$HF_USERNAME/minillm-tokens` (private): token shards and the tokenizer
- `$HF_USERNAME/minillm-125m-ckpt` (private): `pretrain/`, `sft/` and `dpo/` step folders plus `pretrain/milestones/`, about 4 GB; history is squashed to save space
- `$HF_USERNAME/minillm-125m-chat` (public, so browsers can download it): the exported model with its `onnx/` folder

## Expectations

At this size, expect short, mostly correct code snippets and simple explanations. Rough targets:

- HellaSwag acc_norm: about 0.30
- MBPP pass@1: a few percent

It will not reason like a large assistant.

## Troubleshooting

- **OOM on T4:** set `train.micro_batch_size=4` or `model.gradient_checkpointing=true`. Gradient accumulation adjusts automatically.
- **Loss spikes or NaN:** look for `[health]` lines in the log. Roll back with `MiniLLM.Training.Rollback`, and lower `train.max_lr` to `4e-4`. The fp16 loss scale is logged as `train/loss_scale`.
- **Testing DDP on a Mac:** `MINILLM_DEVICE=cpu torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29511 -m MiniLLM.Training.Pretrain --config Configs/SmokeTest.yaml`

## License

Apache-2.0 (see [LICENSE](LICENSE)). Datasets and baseline models keep their own licenses.
