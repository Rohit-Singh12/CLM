"""Train the System One projection heads on MCP-Atlas.

Example:
    python scripts/train.py --epochs 50 --batch-size 32 --proj-dim 512
    python scripts/train.py --resume checkpoints/proj_heads_latest.pt
"""

import argparse
import os
import sys

# Must be set before CUDA is initialised.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from torch.utils.data import DataLoader, Subset  # noqa: E402

from systemone import build_dataloaders, collate_fn, systemone_model, train  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--proj-dim", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--normalize-hidden", action="store_true", help="L2-normalize backbone hidden state")
    p.add_argument("--resume", default=None, help="Path to a checkpoint to resume from")
    p.add_argument("--log-dir", default="runs/mcp_contrastive")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--overfit-n", type=int, default=0,
                   help="Sanity check: train on only the first N training examples")
    return p.parse_args()


def main():
    args = parse_args()

    train_loader, test_loader, train_ds, _ = build_dataloaders(batch_size=args.batch_size)

    if args.overfit_n > 0:
        small = Subset(train_ds, range(args.overfit_n))
        train_loader = DataLoader(small, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = None

    model = systemone_model(
        args.model_id,
        max_length=args.max_length,
        proj_dim=args.proj_dim,
        normalize_base_hidden_state=args.normalize_hidden,
    )
    if args.resume:
        model.load_checkpoint(args.resume)

    train(
        epochs=args.epochs,
        dataloader=train_loader,
        model=model,
        eval_dataloader=test_loader,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
