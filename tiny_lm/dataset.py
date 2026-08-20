from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

from .tokenizer import ShadowTokenizer


@dataclass
class ShadowExample:
    input_text: str
    cleaned_text: str
    intent_label: str
    command: dict


class ShadowDataset(Dataset):
    def __init__(
        self,
        file_path: Path,
        tokenizer: ShadowTokenizer,
        intent_vocabulary: Iterable[str],
        max_input_length: int,
        max_cleaned_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_cleaned_length = max_cleaned_length
        self.intent_to_id = {intent: idx for idx, intent in enumerate(intent_vocabulary)}
        self.examples: list[ShadowExample] = []

        with file_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                payload = json.loads(line)
                self.examples.append(
                    ShadowExample(
                        input_text=payload["input"],
                        cleaned_text=payload["cleaned"],
                        intent_label=payload.get("intent", ""),
                        command=payload.get("command", {}),
                    )
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        intent_id = self.intent_to_id.get(example.intent_label, -1)
        input_ids = self.tokenizer.encode(example.input_text, self.max_input_length)
        cleaned_ids = self.tokenizer.encode(example.cleaned_text, self.max_cleaned_length)
        pad_id = self.tokenizer.token_to_id.get("<pad>", 0)
        input_mask = [0 if token == pad_id else 1 for token in input_ids]
        cleaned_mask = [0 if token == pad_id else 1 for token in cleaned_ids]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "input_mask": torch.tensor(input_mask, dtype=torch.long),
            "cleaned_ids": torch.tensor(cleaned_ids, dtype=torch.long),
            "cleaned_mask": torch.tensor(cleaned_mask, dtype=torch.long),
            "intent_id": torch.tensor(intent_id, dtype=torch.long),
        }




