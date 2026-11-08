# System One Model

Fast next-tool selection with a **frozen LLM backbone** and small **contrastive projection heads**.

Instead of fine-tuning an LLM, the backbone (default: `Qwen/Qwen2.5-0.5B`) is frozen and two small MLP
heads are trained, one for the *state* (instruction + history) and one for each candidate *action*
(tool name). Both are projected into a shared space, and the correct tool is the one whose embedding
is most cosine-similar to the state. Training uses cross-entropy over the candidate tools with a
learnable temperature.

Data: [`ScaleAI/MCP-Atlas`](https://huggingface.co/datasets/ScaleAI/MCP-Atlas), expanded so each
assistant tool call becomes one (state, next-tool) example.

## Layout

```
systemone/
  data.py     # dataset prep, Dataset, collate_fn, dataloaders
  model.py    # SystemOneModel (frozen backbone + heads, loss, checkpointing)
  train.py    # train/evaluate loop, top-k accuracy, TensorBoard update monitor
scripts/
  train.py    # CLI training entry point
  demo.py     # toy inference example
```

## Setup

```bash
pip install -r requirements.txt
```

A GPU is strongly recommended.

## Train

```bash
python scripts/train.py --epochs 50 --batch-size 32 --proj-dim 512

# resume
python scripts/train.py --resume checkpoints/proj_heads_latest.pt

# sanity check: overfit a tiny subset (loss should go to ~0)
python scripts/train.py --overfit-n 64 --batch-size 8 --proj-dim 128 --epochs 40
```

Monitor with `tensorboard --logdir runs`.

## Demo

```bash
python scripts/demo.py --checkpoint checkpoints/proj_heads_latest.pt --proj-dim 512
```

`--proj-dim` must match the value used at training time.

## Notes

- Padding and truncation are on the **left** since the last-token hidden state is used.
- Only head weights and `logit_scale` are checkpointed, not the backbone.
- Checkpoints are saved once per epoch to `--checkpoint-dir`.
