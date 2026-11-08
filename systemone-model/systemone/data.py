"""Dataset preparation for the System One model (ScaleAI/MCP-Atlas)."""

import copy
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, random_split

PROMPT_TEMPLATE = """\
Follow the instruction and predict the next tool call from the list of
available_tools based on the provided context.

## Instruction:
{instruction}

## Available Tools:
{tools}

## Context:
{context}

Predict the next tool to call.
"""


@dataclass
class StateAction:
    """One training example: (state = instruction + history, action = next tool)."""

    instruction: str
    enabled_tools: List[str]
    trajectory: List[Dict[str, Any]]
    action: str


def load_mcp_atlas():
    """Load the MCP-Atlas train split from the Hugging Face Hub."""
    return load_dataset("ScaleAI/MCP-Atlas")["train"]


def is_valid_trajectory(trajectory: List[Dict[str, Any]]) -> bool:
    """Trajectory must alternate assistant, tool, assistant, tool, ..."""
    for i, turn in enumerate(trajectory):
        expected = "assistant" if i % 2 == 0 else "tool"
        if turn.get("role") != expected:
            return False
    return True


def _tool_names(enabled_tools: List[Any], idx: int) -> List[str]:
    names = []
    for tool in enabled_tools:
        if isinstance(tool, dict):
            names.append(tool["name"])
        elif isinstance(tool, str):
            names.append(tool)
        else:
            raise ValueError(f"Invalid enabled-tool format at dataset index {idx}")
    return names


def prepare_input(ds) -> List[StateAction]:
    """Expand each trajectory into one StateAction per assistant tool call."""
    examples: List[StateAction] = []

    for idx, row in enumerate(ds):
        prompt = row["PROMPT"]
        trajectory = json.loads(row["TRAJECTORY"])
        enabled_tools = _tool_names(json.loads(row["ENABLED_TOOLS"]), idx)

        if not is_valid_trajectory(trajectory):
            continue

        for i in range(0, len(trajectory), 2):
            tool_calls = trajectory[i].get("tool_calls") or []
            if not tool_calls:
                continue
            assert len(tool_calls) == 1, f"Row {idx}: expected a single tool call"
            tool = tool_calls[0]

            # Keep only earlier assistant turns; blank out call arguments and
            # drop tool responses. Deep-copy so the source data is not mutated.
            history = []
            for turn in copy.deepcopy(trajectory[:i]):
                if isinstance(turn.get("content"), str):
                    turn["tool_calls"][0]["function"]["arguments"] = ""
                    history.append(turn)

            examples.append(
                StateAction(
                    instruction=prompt,
                    enabled_tools=list(enabled_tools),
                    trajectory=history,
                    action=tool["function"]["name"],
                )
            )

    return examples


class InputDataset(Dataset):
    def __init__(self, inputs: List[StateAction]):
        self.inputs = inputs

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = self.inputs[idx]
        tools = list(data.enabled_tools)  # never mutate the stored example

        # Some trajectories call tools that are not in ENABLED_TOOLS; add them.
        if data.action not in tools:
            tools.append(data.action)

        labels = [0] * len(tools)
        labels[tools.index(data.action)] = 1
        assert sum(labels) == 1

        instruction = PROMPT_TEMPLATE.format(
            instruction=data.instruction,
            tools=tools,
            context=data.trajectory,
        )
        return {"instruction": instruction, "actions": tools, "labels": labels}


def collate_fn(batch):
    """Pad action lists with "" and labels with 0 to the longest in the batch."""
    instructions = [item["instruction"] for item in batch]
    max_s = max(len(item["actions"]) for item in batch)

    padded_actions, padded_labels = [], []
    for item in batch:
        rem = max_s - len(item["actions"])
        padded_actions.append(list(item["actions"]) + [""] * rem)
        padded_labels.append(list(item["labels"]) + [0] * rem)

    return instructions, padded_actions, torch.tensor(padded_labels, dtype=torch.float32)


def build_dataloaders(
    batch_size: int = 32,
    train_frac: float = 0.8,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Dataset, Dataset]:
    """Build train/test loaders from MCP-Atlas."""
    dataset = InputDataset(prepare_input(load_mcp_atlas()))

    train_size = int(train_frac * len(dataset))
    test_size = len(dataset) - train_size
    print(f"Train size: {train_size} | Test size: {test_size}")

    generator = torch.Generator().manual_seed(seed)
    train_ds, test_ds = random_split(dataset, [train_size, test_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")
    return train_loader, test_loader, train_ds, test_ds
