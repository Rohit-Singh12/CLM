"""Tiny smoke test: rank toy actions for a few states with a trained (or fresh) model.

Example:
    python scripts/demo.py --checkpoint checkpoints/proj_heads_latest.pt --proj-dim 512
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

from systemone import systemone_model  # noqa: E402

STATES = [
    "Search the github for a latest PR of this repo",
    "Search the web for meaning of 'Hi'",
    "How to shop online?",
]
ACTIONS = [
    ["github_log", "github_clone", "github_stash", "", ""],
    ["search_web", "search_amazon", "", "", ""],
    ["search_web", "github_log", "search_google", "get_food", "buy_online"],
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--proj-dim", type=int, default=512)
    p.add_argument("--max-length", type=int, default=2048)
    args = p.parse_args()

    model = systemone_model(max_length=args.max_length, proj_dim=args.proj_dim)
    if args.checkpoint:
        model.load_checkpoint(args.checkpoint)
    model.proj_head_states.eval()
    model.proj_head_actions.eval()

    with torch.no_grad():
        logits, mask = model(STATES, ACTIONS)
        logits = logits.masked_fill(~mask, float("-inf"))
        probs = torch.softmax(logits * model.logit_scale.exp().clamp(max=100.0), dim=-1)

    for state, acts, row in zip(STATES, ACTIONS, probs):
        best = row.argmax().item()
        print(f"\nState: {state}")
        for a, p_ in zip(acts, row.tolist()):
            if a:
                print(f"  {a:<15} {p_:.3f}{'  <-- top' if a == acts[best] else ''}")


if __name__ == "__main__":
    main()
