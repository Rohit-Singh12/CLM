"""Training, evaluation and logging utilities."""

import math
import os

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .model import SystemOneModel


class UpdateMonitor:
    """Logs what optimizer.step() actually did to the weights.

    Update/ratio/<param>         ||delta_w|| / ||w|| per step (rule of thumb ~1e-3)
    Update/frac_changed/<param>  fraction of weights that changed at all
    Update/ratio_total, Update/frac_changed_total   same over all params
    Grad/norm/<param>            per-tensor gradient norm (after clipping)
    Grad/hist/*, Weights/hist/*  histograms every `hist_every` steps
    """

    def __init__(self, named_params, writer, hist_every: int = 50):
        self.params = [(n.replace(".", "/"), p) for n, p in named_params if p.requires_grad]
        self.writer = writer
        self.hist_every = hist_every
        self._prev = {}

    @torch.no_grad()
    def before_step(self, step: int):
        for n, p in self.params:
            self._prev[n] = p.detach().clone()
            if p.grad is not None:
                g = p.grad.float()
                self.writer.add_scalar(f"Grad/norm/{n}", g.norm().item(), step)
                if step % self.hist_every == 0:
                    self.writer.add_histogram(f"Grad/hist/{n}", g, step)

    @torch.no_grad()
    def after_step(self, step: int):
        sq_delta = sq_weight = 0.0
        changed_total = numel_total = 0
        for n, p in self.params:
            new, old = p.detach().float(), self._prev[n].float()
            delta = new - old
            d_norm, w_norm = delta.norm().item(), new.norm().item()
            changed = (delta != 0).sum().item()

            self.writer.add_scalar(f"Update/ratio/{n}", d_norm / (w_norm + 1e-12), step)
            self.writer.add_scalar(f"Update/frac_changed/{n}", changed / delta.numel(), step)
            if step % self.hist_every == 0:
                self.writer.add_histogram(f"Weights/hist/{n}", new, step)

            sq_delta += d_norm ** 2
            sq_weight += w_norm ** 2
            changed_total += changed
            numel_total += delta.numel()

        self.writer.add_scalar(
            "Update/ratio_total", (sq_delta ** 0.5) / (sq_weight ** 0.5 + 1e-12), step
        )
        self.writer.add_scalar("Update/frac_changed_total", changed_total / numel_total, step)


def topk_accuracy(logits, target_indices, action_mask, ks=(1, 3)):
    """logits: (B, S) cosine sims, target_indices: (B,), action_mask: (B, S) bool."""
    masked = logits.masked_fill(~action_mask, float("-inf"))
    n_valid = action_mask.sum(dim=-1)
    out = {}
    for k in ks:
        kk = min(k, masked.shape[1])
        topk = masked.topk(kk, dim=-1).indices
        out[k] = (topk == target_indices.unsqueeze(1)).any(dim=-1).float().mean().item()
    # Chance levels, to read the numbers properly.
    out["chance_top1"] = (1.0 / n_valid.float()).mean().item()
    out["chance_top3"] = (3.0 / n_valid.float()).clamp(max=1.0).mean().item()
    return out


@torch.no_grad()
def evaluate(model: SystemOneModel, dataloader: DataLoader):
    """Returns (mean loss per sample, top-1 accuracy)."""
    model.proj_head_states.eval()
    model.proj_head_actions.eval()
    total_loss, total_correct, n = 0.0, 0.0, 0
    for states, actions, labels in tqdm(dataloader, desc="Evaluating", unit="batch", leave=False):
        logits, action_mask = model(states, actions)
        loss = model.contrastive_loss(logits, labels, action_mask)
        target = labels.to(model.device).argmax(dim=-1)
        masked = logits.masked_fill(~action_mask, float("-inf"))
        bs = target.shape[0]
        total_loss += loss.item() * bs
        total_correct += (masked.argmax(dim=-1) == target).float().sum().item()
        n += bs
    model.proj_head_states.train()
    model.proj_head_actions.train()
    return total_loss / n, total_correct / n


def train(
    epochs: int,
    dataloader: DataLoader,
    model: SystemOneModel,
    eval_dataloader: DataLoader = None,
    log_dir: str = "runs/mcp_contrastive",
    checkpoint_dir: str = "checkpoints",
    checkpoint_name: str = "proj_heads_latest.pt",
    lr: float = 1e-3,
    warmup_steps: int = 20,
    min_lr_ratio: float = 0.05,
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, checkpoint_name)

    writer = SummaryWriter(log_dir=log_dir)
    model.model.eval()  # frozen backbone
    model.proj_head_states.train()
    model.proj_head_actions.train()

    head_params = list(model.proj_head_states.parameters()) + list(model.proj_head_actions.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "weight_decay": 0.01},
            {"params": [model.logit_scale], "weight_decay": 0.0},
        ],
        lr=lr,
    )

    total_steps = epochs * len(dataloader)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    scheduler = LambdaLR(optimizer, lr_lambda)
    monitor = UpdateMonitor(
        list(model.proj_head_states.named_parameters(prefix="states"))
        + list(model.proj_head_actions.named_parameters(prefix="actions")),
        writer,
    )
    global_step = 0

    for epoch in range(epochs):
        total_loss, n_ok = 0.0, 0

        for idx, (states, actions, labels) in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
        ):
            try:
                optimizer.zero_grad()
                logits, action_mask = model(states, actions)
                loss = model.contrastive_loss(logits, labels, action_mask)
                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(head_params + [model.logit_scale], max_norm=1.0)

                monitor.before_step(global_step)
                writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step)
                optimizer.step()
                monitor.after_step(global_step)
                scheduler.step()

                with torch.no_grad():
                    target_indices = labels.to(model.device).argmax(dim=-1)
                    m = topk_accuracy(logits, target_indices, action_mask)

                    pos_sim = logits.gather(1, target_indices.unsqueeze(1)).squeeze(1)
                    mean_pos_sim = pos_sim.mean().item()

                    neg_mask = action_mask.clone()
                    neg_mask[torch.arange(neg_mask.shape[0]), target_indices] = False
                    mean_neg_sim = logits[neg_mask].mean().item() if neg_mask.any() else 0.0

                writer.add_scalar("Loss/train", loss.item(), global_step)
                writer.add_scalar("Accuracy/Top-1", m[1], global_step)
                writer.add_scalar("Accuracy/Top-3", m[3], global_step)
                writer.add_scalar("Accuracy/Chance-Top-1", m["chance_top1"], global_step)
                writer.add_scalar("Similarity/Positive", mean_pos_sim, global_step)
                writer.add_scalar("Similarity/Negative", mean_neg_sim, global_step)
                writer.add_scalar("Similarity/Gap", mean_pos_sim - mean_neg_sim, global_step)
                writer.add_scalar("Gradients/proj_head_norm", grad_norm.item(), global_step)

                global_step += 1
                total_loss += loss.item()
                n_ok += 1
            except torch.cuda.OutOfMemoryError:
                print(f"OOM on batch {idx} in epoch {epoch}; skipping")
                optimizer.zero_grad()
                torch.cuda.empty_cache()

        # Save once per epoch (not every step).
        model.save_checkpoint(ckpt_path)

        avg_epoch_loss = total_loss / max(1, n_ok)
        print(f"Epoch [{epoch + 1}/{epochs}] | Avg Loss: {avg_epoch_loss:.4f}")
        if eval_dataloader is not None:
            eval_loss, eval_acc = evaluate(model, eval_dataloader)
            writer.add_scalar("Epoch/Loss_eval", eval_loss, epoch + 1)
            writer.add_scalar("Epoch/Top1_eval", eval_acc, epoch + 1)
            print(f" | Eval Loss: {eval_loss:.4f} | Eval Top-1: {eval_acc:.3f}")

    writer.close()
    print(f"Training complete. Checkpoint: {ckpt_path} | TensorBoard logs: {log_dir}")
