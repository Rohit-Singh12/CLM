"""System One model: a frozen LLM backbone with trainable state/action projection heads."""

import gc

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer


def make_head(hidden_dim: int, proj_dim: int, dim: int = 1024, p: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(hidden_dim),  # tames outlier dims in LM features
        nn.Linear(hidden_dim, dim),
        nn.GELU(),
        nn.Dropout(p),
        nn.Linear(dim, proj_dim),
    )


class SystemOneModel(nn.Module):
    """Contrastive state/action embedder.

    The backbone is frozen and shared (class-level cache). Two separate MLP heads
    project the last-token hidden state of the state text and of each action text
    into a shared space; the logit for an action is the cosine similarity between
    its embedding and the state embedding.
    """

    _model = None
    _tokenizer = None

    def __init__(
        self,
        model_id: str,
        max_length: int = 2048,
        proj_dim: int = 16,
        normalize_base_hidden_state: bool = False,
        init_temperature: float = 0.07,
        chunk_size: int = 32,
    ):
        super().__init__()
        self.model_name = model_id
        self._load_resources(model_id)
        self.model = SystemOneModel._model
        self.tokenizer = SystemOneModel._tokenizer
        self.device = self.model.device
        self.normalize_base_hidden_state = normalize_base_hidden_state
        self.max_length = max_length
        self.chunk_size = chunk_size

        self.hidden_dim = self.model.config.hidden_size
        print(f"Size of hidden_dim: {self.hidden_dim}")

        # Heads stay in float32 so gradient updates are accurate.
        self.proj_head_states = make_head(self.hidden_dim, proj_dim).to(self.device, dtype=torch.float32)
        self.proj_head_actions = make_head(self.hidden_dim, proj_dim).to(self.device, dtype=torch.float32)

        # Learnable temperature, stored in log space so the scale stays positive.
        self.logit_scale = nn.Parameter(
            torch.tensor(1.0 / init_temperature, device=self.device).log()
        )

        # Decoder-only model + last-token pooling: pad on the LEFT so the last
        # position is always a real token. Truncate from the left to keep the
        # most recent context.
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

    @classmethod
    def _load_resources(cls, model_name: str) -> None:
        if cls._model is None:
            cls._model = AutoModel.from_pretrained(model_name, device_map="auto")
            for param in cls._model.parameters():
                param.requires_grad = False
            cls._model.eval()
            cls._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _encode(self, texts, projection_head: nn.Module) -> torch.Tensor:
        tokenized = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**tokenized)
            # (B, seq_len, hidden) -> (B, hidden); cast to fp32 for the fp32 head
            last_hidden = outputs.last_hidden_state[:, -1, :].float()

        if self.normalize_base_hidden_state:
            last_hidden = F.normalize(last_hidden, p=2, dim=-1)
        return projection_head(last_hidden)

    @staticmethod
    def _similarity_score(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Cosine similarity. states: (B, d), actions: (B, S, d) -> (B, S)."""
        states = F.normalize(states, p=2, dim=-1).unsqueeze(1)  # (B, 1, d)
        actions = F.normalize(actions, p=2, dim=-1)             # (B, S, d)
        sim = (states * actions).sum(dim=-1)
        assert sim.shape == actions.shape[:2]
        return sim

    def contrastive_loss(self, logits, labels, action_mask=None) -> torch.Tensor:
        labels = labels.to(self.device)
        target_indices = labels.argmax(dim=-1)
        if action_mask is not None:
            # Exclude padded actions from the softmax.
            logits = logits.masked_fill(~action_mask, -1e4)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return F.cross_entropy(logits * scale, target_indices)

    def forward(self, states, actions):
        if isinstance(states, (list, tuple)):
            states = self._encode(states, self.proj_head_states)

        # Empty strings are padding; give the tokenizer something non-empty.
        flat_actions = [a if a.strip() else self.tokenizer.pad_token for row in actions for a in row]
        B, S = len(actions), len(actions[0])

        # Encode in chunks to avoid activation-memory spikes.
        encoded = [
            self._encode(flat_actions[i : i + self.chunk_size], self.proj_head_actions)
            for i in range(0, len(flat_actions), self.chunk_size)
        ]
        action_embeddings = torch.cat(encoded, dim=0).view(B, S, -1)

        action_mask = torch.tensor(
            [[act != "" for act in row] for row in actions],
            dtype=torch.bool,
            device=self.device,
        )
        logits = self._similarity_score(states, action_embeddings)
        return logits, action_mask  # (B, S), (B, S)

    # ---- checkpointing -------------------------------------------------
    def save_checkpoint(self, path: str) -> None:
        torch.save(
            {
                "proj_head_states": self.proj_head_states.state_dict(),
                "proj_head_actions": self.proj_head_actions.state_dict(),
                "logit_scale": self.logit_scale.data,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.proj_head_states.load_state_dict(ckpt["proj_head_states"])
        self.proj_head_actions.load_state_dict(ckpt["proj_head_actions"])
        self.logit_scale.data = ckpt["logit_scale"].to(self.device)
        print(f"Checkpoint loaded from {path}")

    def unload_current_model(self) -> None:
        """Free GPU memory held by the backbone."""
        if self.model is not None:
            self.model.to("cpu")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            SystemOneModel._model = None
            SystemOneModel._tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()


def systemone_model(model_id: str = "Qwen/Qwen2.5-0.5B", max_length: int = 2048, proj_dim: int = 16, **kwargs):
    return SystemOneModel(model_id, max_length=max_length, proj_dim=proj_dim, **kwargs)
