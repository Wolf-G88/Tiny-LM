from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn

EncoderMode = Literal["gru", "transformer"]


@dataclass
class TinyLMConfig:
    vocab_size: int
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int
    d_model: int = 96
    max_seq_len: int = 256
    gru_hidden_size: int = 96
    gru_num_layers: int = 1
    gru_bidirectional: bool = False
    num_layers: int = 2
    num_heads: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1
    num_intents: int = 16
    use_command_head: bool = True
    length_threshold: int = 96


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class GRUEncoder(nn.Module):
    def __init__(self, config: TinyLMConfig) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=config.d_model,
            hidden_size=config.gru_hidden_size,
            num_layers=config.gru_num_layers,
            batch_first=True,
            bidirectional=config.gru_bidirectional,
        )
        hidden_out_dim = config.gru_hidden_size * (2 if config.gru_bidirectional else 1)
        self.proj = nn.Linear(hidden_out_dim, config.d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        output, _ = self.gru(x)
        encoded = self.proj(output)
        return encoded


class TransformerEncoder(nn.Module):
    def __init__(self, config: TinyLMConfig) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            key_padding_mask = (mask == 0)
        else:
            key_padding_mask = None
        encoded = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return encoded


class CleanupHead(nn.Module):
    def __init__(self, config: TinyLMConfig) -> None:
        super().__init__()
        self.proj = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.proj(encoded)


class IntentHead(nn.Module):
    def __init__(self, config: TinyLMConfig) -> None:
        super().__init__()
        self.linear = nn.Linear(config.d_model, config.num_intents)

    def forward(self, encoded: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            pooled = encoded.mean(dim=1)
        else:
            mask = mask.unsqueeze(-1).to(encoded.dtype)
            summed = (encoded * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = summed / denom
        logits = self.linear(pooled)
        return logits


class CommandHead(nn.Module):
    def __init__(self, config: TinyLMConfig, command_dim: int = 64) -> None:
        super().__init__()
        self.linear = nn.Linear(config.d_model, command_dim)
        self.command_dim = command_dim

    def forward(self, encoded: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            pooled = encoded.mean(dim=1)
        else:
            mask = mask.unsqueeze(-1).to(encoded.dtype)
            summed = (encoded * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = summed / denom
        return self.linear(pooled)


class Router(nn.Module):
    def __init__(self, config: TinyLMConfig) -> None:
        super().__init__()
        self.length_threshold = config.length_threshold

    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> EncoderMode:
        if mask is not None:
            lengths = mask.sum(dim=1)
        else:
            lengths = (input_ids != 0).sum(dim=1)
        avg_len = lengths.float().mean().item()
        if avg_len > self.length_threshold:
            return "transformer"
        return "gru"


class TinyLM(nn.Module):
    def __init__(self, config: TinyLMConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.positional_encoding = PositionalEncoding(d_model=config.d_model, max_len=config.max_seq_len)
        self.gru_encoder = GRUEncoder(config)
        self.transformer_encoder = TransformerEncoder(config)
        self.cleanup_head = CleanupHead(config)
        self.intent_head = IntentHead(config)
        self.command_head = CommandHead(config) if config.use_command_head else None
        self.router = Router(config)

    def encode(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor],
        encoder_mode: Optional[EncoderMode] = None,
    ) -> Tuple[torch.Tensor, EncoderMode]:
        x = self.embedding(input_ids)
        x = self.positional_encoding(x)
        if encoder_mode is None:
            encoder_mode = self.router(input_ids, mask)
        if encoder_mode == "gru":
            encoded = self.gru_encoder(x, mask)
        elif encoder_mode == "transformer":
            encoded = self.transformer_encoder(x, mask)
        else:
            raise ValueError(f"Unknown encoder_mode: {encoder_mode}")
        return encoded, encoder_mode

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_mode: Optional[EncoderMode] = None,
        return_command: bool = False,
    ) -> Dict[str, Any]:
        encoded, mode_used = self.encode(input_ids, attention_mask, encoder_mode)
        clean_logits = self.cleanup_head(encoded)
        intent_logits = self.intent_head(encoded, attention_mask)
        command_repr = None
        if return_command and self.command_head is not None:
            command_repr = self.command_head(encoded, attention_mask)
        return {
            "clean_logits": clean_logits,
            "intent_logits": intent_logits,
            "command_repr": command_repr,
            "encoder_mode": mode_used,
        }

    @torch.no_grad()
    def generate_clean(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 0,
        encoder_mode: Optional[EncoderMode] = None,
    ) -> torch.Tensor:
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_mode=encoder_mode,
            return_command=False,
        )
        clean_ids = outputs["clean_logits"].argmax(dim=-1)
        return clean_ids


