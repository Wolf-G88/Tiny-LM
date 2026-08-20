from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import Iterable, List


@dataclass
class ShadowTokenizer:
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]

    @classmethod
    def build(cls, texts: Iterable[str], base_tokens: Iterable[str] | None = None) -> "ShadowTokenizer":
        base_tokens = list(base_tokens or ["<pad>", "<unk>"])
        unique_chars = set(chain.from_iterable(texts))
        extras = [c for c in sorted(unique_chars) if c not in base_tokens]
        ordered = base_tokens + extras
        token_to_id = {token: idx for idx, token in enumerate(ordered)}
        return cls(token_to_id, {idx: token for token, idx in token_to_id.items()})

    def encode(self, text: str, max_length: int) -> List[int]:
        tokens = [self.token_to_id.get(char, self.token_to_id["<unk>"]) for char in text]
        if len(tokens) >= max_length:
            return tokens[:max_length]
        tokens += [self.token_to_id["<pad>"]] * (max_length - len(tokens))
        return tokens

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.id_to_token.get(idx, "") for idx in ids)

    def vocab_size(self) -> int:
        return len(self.token_to_id)

