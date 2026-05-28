import os
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

# --- constants ---
MODEL_NAME = "distilbert-base-uncased"
TRAIN_PATH = "data/processed/train.csv"
VAL_PATH = "data/processed/val.csv"
OUTPUT_DIR = "models/promptshield"
MAX_LENGTH = 256  # most prompts fit in 256 tokens; 512 uses more GPU memory for minimal gain
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
EPOCHS = 4


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    return train_df, val_df


def tokenize(batch, tokenizer):
    # padding="max_length" pads shorter sequences to MAX_LENGTH with zeros
    # truncation=True cuts anything longer than MAX_LENGTH
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )


def build_datasets(train_df, val_df, tokenizer):
    # Trainer expects HuggingFace Dataset objects, not pandas DataFrames
    train_ds = Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df[["text", "label"]].reset_index(drop=True))

    # batched=True processes rows in chunks, which is much faster than row-by-row
    train_ds = train_ds.map(lambda b: tokenize(b, tokenizer), batched=True)
    val_ds = val_ds.map(lambda b: tokenize(b, tokenizer), batched=True)

    # Trainer looks for the column "labels" (plural) - rename from "label"
    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")

    # tell the Dataset which columns are tensors to pass to the model
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    return train_ds, val_ds


def compute_class_weights(train_df):
    # compute inverse-frequency weights: rarer class gets a higher weight
    counts = train_df["label"].value_counts().sort_index()  # [count_0, count_1]
    total = len(train_df)
    # formula: total / (num_classes * count_for_that_class)
    weights = total / (2 * counts.values)
    return torch.tensor(weights, dtype=torch.float)


def compute_metrics(eval_pred):
    # eval_pred is a tuple of (logits, labels) from the val set
    logits, labels = eval_pred
    # argmax picks the class with the higher logit (0 = safe, 1 = unsafe)
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predictions, average="macro")
    acc = accuracy_score(labels, predictions)
    return {"f1": f1, "accuracy": acc}


class WeightedTrainer(Trainer):
    """Subclass of Trainer that injects class weights into CrossEntropyLoss."""

    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # pop "labels" from inputs so they don't get passed to the model forward pass
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # move weights to the same device as logits (handles both CPU and GPU)
        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def main():
    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    # num_labels=2 adds a classification head with 2 output logits (safe / unsafe)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    train_df, val_df = load_data()
    print(f"Train: {len(train_df)} rows | Val: {len(val_df)} rows")

    train_ds, val_ds = build_datasets(train_df, val_df, tokenizer)
    class_weights = compute_class_weights(train_df)
    print(f"Class weights: safe={class_weights[0]:.3f}, unsafe={class_weights[1]:.3f}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        # eval doesn't need gradient storage, so a larger batch fits
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        # warmup gradually increases LR from 0 to 2e-5 over the first 100 steps
        # this prevents the large gradient updates that can destabilise fine-tuning early on
        warmup_steps=100,
        # L2 regularisation: small penalty on large weights to reduce overfitting
        weight_decay=0.01,
        eval_strategy="epoch",       # evaluate on the val set after every epoch
        save_strategy="epoch",       # save a checkpoint after every epoch
        # after training, reload the checkpoint with the best val F1 (not the last epoch)
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",            # disable wandb / tensorboard
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    # save the best checkpoint and tokenizer together so evaluate.py can load both
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Best model saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
