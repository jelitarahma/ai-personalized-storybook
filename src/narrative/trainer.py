"""
Domain fine-tuning module for GPT-2 on Indonesian children's stories.
Implements Trainer / Training Loop with loss tracking and perplexity calculation.
"""

import os
import sys
import json
import math
import inspect
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path for standalone script execution
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from src.utils.reproducibility import set_seed

DEFAULT_PRETRAINED = "cahya/gpt2-small-indonesian-522M"


def load_dataset_from_json(json_path: str) -> Dataset:
    """
    Loads JSON dataset and prepares for Hugging Face Dataset format.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    texts = [item["formatted_text"] for item in data]
    return Dataset.from_dict({"text": texts})


def fine_tune_gpt2(
    train_json: str = "data/processed/train.json",
    val_json: str = "data/processed/val.json",
    model_name: str = DEFAULT_PRETRAINED,
    output_dir: str = "models/checkpoints/gpt2_storybook",
    num_epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 5e-5,
    max_length: int = 128,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Fine-tune GPT-2 model on children's stories dataset.

    Returns:
        dict: Training history including final training/eval loss and perplexity.
    """
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    logs_dir = "outputs"
    os.makedirs(logs_dir, exist_ok=True)

    print(f"Initializing fine-tuning for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load pretrained model
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Load datasets
    train_dataset = load_dataset_from_json(train_json)
    val_dataset = load_dataset_from_json(val_json)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    tokenized_train = train_dataset.map(tokenize_fn, batched=True)
    tokenized_val = val_dataset.map(tokenize_fn, batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Build TrainingArguments dynamically to support both new and older transformers versions
    training_args_kwargs: Dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "learning_rate": learning_rate,
        "save_strategy": "epoch",
        "logging_strategy": "epoch",
        "save_total_limit": 1,
        "seed": seed,
        "fp16": torch.cuda.is_available(),
        "dataloader_pin_memory": torch.cuda.is_available(),
        "report_to": "none",
        "remove_unused_columns": True
    }

    ta_sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in ta_sig.parameters:
        training_args_kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in ta_sig.parameters:
        training_args_kwargs["evaluation_strategy"] = "epoch"

    if "overwrite_output_dir" in ta_sig.parameters:
        training_args_kwargs["overwrite_output_dir"] = True

    training_args = TrainingArguments(**training_args_kwargs)

    # Build Trainer dynamically (supports processing_class for transformers>=4.45 or tokenizer for older)
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_train,
        "eval_dataset": tokenized_val,
        "data_collator": data_collator
    }

    trainer_sig = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    print("Starting training loop...")
    train_result = trainer.train()

    # Evaluation
    eval_metrics = trainer.evaluate()
    eval_loss = eval_metrics.get("eval_loss", 0.0)
    eval_perplexity = math.exp(eval_loss) if eval_loss < 20 else float("inf")

    total_flos = getattr(train_result, "total_flos", train_result.metrics.get("total_flos", 0.0))
    train_loss = getattr(train_result, "training_loss", train_result.metrics.get("train_loss", 0.0))

    history = {
        "model_name": model_name,
        "epochs": num_epochs,
        "train_loss": train_loss,
        "train_perplexity": round(math.exp(train_loss), 4) if train_loss < 20 else float("inf"),
        "eval_loss": eval_loss,
        "eval_perplexity": round(eval_perplexity, 4),
        "total_flos": total_flos,
        "train_runtime_seconds": round(train_result.metrics.get("train_runtime", 0.0), 2)
    }

    # Save fine-tuned checkpoint & tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    log_file = os.path.join(logs_dir, "training_log.json")
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Fine-tuning complete. Checkpoint saved to: {output_dir}")
    print(f"Training Loss: {history['train_loss']:.4f} | Perplexity: {history['train_perplexity']:.2f}")
    print(f"Validation Loss: {history['eval_loss']:.4f} | Perplexity: {history['eval_perplexity']:.2f}")

    return history


if __name__ == "__main__":
    fine_tune_gpt2(num_epochs=1, batch_size=2)
