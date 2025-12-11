#!/usr/bin/env python3

import pandas as pd
import numpy as np
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)
from sklearn.preprocessing import LabelEncoder


def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )

    return {
        "accuracy": acc,
        "precision_weighted": precision_w,
        "recall_weighted": recall_w,
        "f1_weighted": f1_w,
        "precision_macro": precision_m,
        "recall_macro": recall_m,
        "f1_macro": f1_m,
    }


def main():
    # -------------------------
    # 1. Load and prepare data
    # -------------------------
    print("Loading data...")
    # TODO: change this if your file is named differently
    df = pd.read_csv("preprocessed_edos_labelled_data.csv")  # or whatever your CSV is

    # Expecting: rewire_id, text, label, split
    print("Data shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df["split"].value_counts())

    # Encode labels
    label_encoder = LabelEncoder()
    df["label_id"] = label_encoder.fit_transform(df["label"])
    num_labels = len(label_encoder.classes_)
    print("Classes:", list(label_encoder.classes_))

    # Train / test split
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    print("Train size:", len(train_df))
    print("Test size:", len(test_df))

    # -------------------------
    # 2. Tokenizer + datasets
    # -------------------------
    model_name = "roberta-base"
    print(f"\nLoading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    # HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df[["text", "label_id"]])
    test_dataset = Dataset.from_pandas(test_df[["text", "label_id"]])

    train_dataset = train_dataset.map(tokenize_batch, batched=True)
    test_dataset = test_dataset.map(tokenize_batch, batched=True)

    # Remove text, keep tensors
    train_dataset = train_dataset.remove_columns(["text"])
    test_dataset = test_dataset.remove_columns(["text"])

    train_dataset = train_dataset.rename_column("label_id", "labels")
    test_dataset = test_dataset.rename_column("label_id", "labels")

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # -------------------------
    # 3. Model
    # -------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    # -------------------------
    # 4. TrainingArguments (no evaluation_strategy, etc.)
    # -------------------------
    training_args = TrainingArguments(
        output_dir="./roberta_sexism",
        num_train_epochs=4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=1e-5,
        weight_decay=0.01,
        logging_steps=100,
        save_total_limit=2,
        # older versions don’t support evaluation_strategy/save_strategy/etc
    )

    # -------------------------
    # 5. Trainer
    # -------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,   # used when we call trainer.evaluate / predict
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # -------------------------
    # 6. Train
    # -------------------------
    print("\nStarting training...")
    trainer.train()

    # -------------------------
    # 7. Evaluate on test set
    # -------------------------
    print("\nEvaluating on test set with Trainer.evaluate...")
    metrics = trainer.evaluate(test_dataset)
    print("Eval metrics:", metrics)

    print("\nGetting predictions for detailed classification report...")
    preds_output = trainer.predict(test_dataset)
    logits = preds_output.predictions
    y_pred = np.argmax(logits, axis=-1)
    y_true = preds_output.label_ids

    print("\nClassification report (sklearn):")
    print(classification_report(
        y_true,
        y_pred,
        target_names=label_encoder.classes_
    ))


if __name__ == "__main__":
    main()
