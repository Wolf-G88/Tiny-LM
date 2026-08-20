from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from tiny_lm.dataset import ShadowDataset
from tiny_lm.model import TinyLM, TinyLMConfig
from tiny_lm.tokenizer import ShadowTokenizer


def load_examples(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_tokenizer(examples: Iterable[dict], base_tokens: Iterable[str]) -> ShadowTokenizer:
    texts = (entry["input"] + entry["cleaned"] for entry in examples)
    return ShadowTokenizer.build(texts, base_tokens)


def save_metadata(output_dir: Path, tokenizer: ShadowTokenizer, config: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "token_to_id": tokenizer.token_to_id,
        "config": config,
    }
    with (output_dir / "tokenizer_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


def create_dataloader(config: dict, tokenizer: ShadowTokenizer) -> DataLoader:
    dataset = ShadowDataset(
        file_path=Path(config["dataset_path"]),
        tokenizer=tokenizer,
        intent_vocabulary=config["intents"],
        max_input_length=config["max_input_length"],
        max_cleaned_length=config["max_cleaned_length"],
    )
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)


def training_loop(
    model: TinyLM,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    device: torch.device,
    pad_id: int,
    loss_weights: dict[str, float],
    epochs: int,
) -> None:
    clean_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    intent_loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_clean = 0.0
        total_intent = 0.0
        for batch in dataloader:
            inputs = batch["input_ids"].to(device)
            attention_mask = batch["input_mask"].to(device)
            cleaned = batch["cleaned_ids"].to(device)
            intent = batch["intent_id"].to(device)
            outputs = model(inputs, attention_mask=attention_mask)
            clean_logits = outputs["clean_logits"].reshape(-1, model.config.vocab_size)
            loss_clean = clean_loss_fn(clean_logits, cleaned.reshape(-1))
            loss_intent = intent_loss_fn(outputs["intent_logits"], intent)
            loss = loss_weights["clean"] * loss_clean + loss_weights["intent"] * loss_intent
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_clean += loss_clean.item()
            total_intent += loss_intent.item()
        scheduler.step()
        avg_clean = total_clean / len(dataloader)
        avg_intent = total_intent / len(dataloader)
        lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch + 1}/{epochs} — clean loss {avg_clean:.4f}, intent loss {avg_intent:.4f}, lr {lr:.5f}"
        )


def build_config(config: dict, tokenizer: ShadowTokenizer) -> TinyLMConfig:
    return TinyLMConfig(
        vocab_size=tokenizer.vocab_size(),
        pad_token_id=tokenizer.token_to_id["<pad>"],
        bos_token_id=tokenizer.token_to_id.get("<bos>", tokenizer.token_to_id["<pad>"]),
        eos_token_id=tokenizer.token_to_id.get("<eos>", tokenizer.token_to_id["<pad>"]),
        d_model=config["d_model"],
        max_seq_len=config["max_input_length"],
        gru_hidden_size=config.get("gru_hidden_size", 96),
        gru_num_layers=config.get("gru_num_layers", 1),
        gru_bidirectional=config.get("gru_bidirectional", False),
        num_layers=config.get("num_layers", 2),
        num_heads=config.get("num_heads", 4),
        dim_feedforward=config.get("dim_feedforward", 256),
        dropout=config.get("dropout", 0.1),
        num_intents=len(config["intents"]),
        use_command_head=config.get("use_command_head", True),
        length_threshold=config.get("length_threshold", 96),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Shadow Tiny LM v0")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.json"),
        help="Configuration file path",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    dataset_path = Path(config["dataset_path"])
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found at {dataset_path}")
    examples = load_examples(dataset_path)
    tokenizer = build_tokenizer(examples, config["tokenizer_base_tokens"])
    dataloader = create_dataloader(config, tokenizer)
    model_config = build_config(config, tokenizer)
    model = TinyLM(model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, config["epochs"]))
    loss_weights = {
        "clean": config["clean_loss_weight"],
        "intent": config["intent_loss_weight"],
    }
    training_loop(
        model,
        dataloader,
        optimizer,
        scheduler,
        device,
        pad_id=tokenizer.token_to_id["<pad>"],
        loss_weights=loss_weights,
        epochs=config["epochs"],
    )
    output_dir = Path(config["output_dir"])
    save_metadata(output_dir, tokenizer, config)
    torch.save(model.state_dict(), output_dir / "shadow_tiny_lm.pt")
    print(f"Model and metadata exported to {output_dir}")



