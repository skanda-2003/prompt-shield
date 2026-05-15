import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

MODEL_PATH = "models/promptshield"
VAL_PATH = "data/processed/val.csv"
TEST_PATH = "data/processed/test.csv"
PREDICTIONS_PATH = "data/processed/test_predictions.csv"
MAX_LENGTH = 256  # must match train.py - tokens beyond this were truncated during training
BATCH_SIZE = 32   # larger than training batch is fine - inference needs no gradient storage

# the 5 named attack categories PromptShield is evaluated against
ATTACK_CATEGORIES = [
    "role_playing",
    "instruction_override",
    "prompt_leaking",
    "context_manipulation",
    "encoded_attacks",
]


def load_model():
    print(f"Loading model from {MODEL_PATH}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

    # eval mode disables dropout so inference is deterministic
    model.eval()

    # use GPU if available, fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Running on: {device}")
    return tokenizer, model, device


def get_probabilities(texts, tokenizer, model, device):
    """Run inference on a list of texts. Returns P(unsafe) for each text."""
    all_probs = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]

        # tokenize with the same settings used in training
        encoding = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",  # return PyTorch tensors, not lists
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # no_grad tells PyTorch not to track gradients - saves memory, speeds up inference
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # softmax converts raw logits into probabilities that sum to 1
        # we take column 1 = P(unsafe), which is what we threshold against
        probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_probs)


def tune_threshold(probs, labels):
    """
    Try thresholds from 0.30 to 0.70 in steps of 0.05.
    Returns the threshold with the best macro F1 on the validation set.
    I do this on val (not test) so the threshold decision doesn't leak into evaluation.
    """
    best_threshold = 0.5
    best_f1 = 0.0

    print(f"\n  {'Threshold':>9}  {'Macro F1':>8}")
    for threshold in np.arange(0.30, 0.71, 0.05):
        preds = (probs >= threshold).astype(int)
        f1 = f1_score(labels, preds, average="macro")
        marker = " <-- best" if f1 > best_f1 else ""
        print(f"  {threshold:>9.2f}  {f1:>8.4f}{marker}")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = round(float(threshold), 2)

    return best_threshold, round(best_f1, 4)


def print_metrics(labels, preds, probs, threshold):
    acc = accuracy_score(labels, preds)
    roc_auc = roc_auc_score(labels, probs)
    cm = confusion_matrix(labels, preds)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, labels=[0, 1], zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )

    print(f"\nDecision threshold used: {threshold}")
    print(f"Accuracy : {acc:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}  (threshold-independent - measures overall separability)")

    print(f"\nPer-class metrics:")
    print(f"  {'Class':<12} {'Precision':>9}  {'Recall':>6}  {'F1':>6}  {'N':>5}")
    print(f"  {'-'*44}")
    print(f"  {'safe (0)':<12} {precision[0]:>9.4f}  {recall[0]:>6.4f}  {f1[0]:>6.4f}  {support[0]:>5}")
    print(f"  {'unsafe (1)':<12} {precision[1]:>9.4f}  {recall[1]:>6.4f}  {f1[1]:>6.4f}  {support[1]:>5}")
    print(f"  {'macro avg':<12} {macro_p:>9.4f}  {macro_r:>6.4f}  {macro_f1:>6.4f}")

    # TN = safe correctly predicted safe, FP = safe wrongly flagged as unsafe
    # FN = attack missed (most dangerous failure mode), TP = attack correctly caught
    print(f"\nConfusion matrix (rows = actual, cols = predicted):")
    print(f"  [[TN={cm[0,0]:>3}  FP={cm[0,1]:>3}]   <- actual safe")
    print(f"   [FN={cm[1,0]:>3}  TP={cm[1,1]:>3}]]  <- actual unsafe")


def print_per_category(test_df, preds, probs):
    df = test_df.copy()
    df["pred"] = preds
    df["prob"] = probs

    print(f"\nPer-attack-category breakdown:")
    print(f"  {'Category':<25} {'N':>4}  {'Precision':>9}  {'Recall':>6}  {'F1':>6}")
    print(f"  {'-'*55}")

    # evaluate the 5 named attack categories
    for cat in ATTACK_CATEGORIES:
        subset = df[df["attack_category"] == cat]
        n = len(subset)

        if n == 0:
            print(f"  {cat:<25} {n:>4}  (no test samples)")
            continue

        p, r, f, _ = precision_recall_fscore_support(
            subset["label"], subset["pred"],
            average="binary", pos_label=1, zero_division=0
        )
        print(f"  {cat:<25} {n:>4}  {p:>9.4f}  {r:>6.4f}  {f:>6.4f}")

        # false negatives = attacks the model missed - the most important failure mode
        fn = subset[(subset["label"] == 1) & (subset["pred"] == 0)]
        if len(fn) > 0:
            print(f"    False negatives ({len(fn)}):")
            for _, row in fn.head(2).iterrows():
                snippet = row["text"][:100].replace("\n", " ")
                print(f"      [prob={row['prob']:.3f}] {snippet}...")

    # "unknown" is also unsafe but didn't match any of the 5 named patterns
    unknown = df[df["attack_category"] == "unknown"]
    if len(unknown) > 0:
        caught = (unknown["pred"] == 1).sum()
        print(f"\n  unknown (unsafe, unclassified): {len(unknown)} samples, {caught} caught ({caught/len(unknown):.1%} recall)")

    # false positive rate on safe prompts
    safe = df[df["attack_category"] == "safe"]
    if len(safe) > 0:
        fp = (safe["pred"] == 1).sum()
        print(f"  safe: {len(safe)} samples, {fp} false positives ({fp/len(safe):.1%} false positive rate)")


def main():
    tokenizer, model, device = load_model()

    # --- step 1: tune threshold on val set ---
    print("\nTuning decision threshold on validation set...")
    val_df = pd.read_csv(VAL_PATH)
    val_probs = get_probabilities(val_df["text"].tolist(), tokenizer, model, device)
    best_threshold, best_val_f1 = tune_threshold(val_probs, val_df["label"].values)
    print(f"\nSelected threshold: {best_threshold}  (val macro F1: {best_val_f1})")

    # --- step 2: evaluate on test set - only opened once, right here ---
    print("\n" + "=" * 55)
    print("Test Set Evaluation")
    print("=" * 55)
    test_df = pd.read_csv(TEST_PATH)
    print(f"Test set: {len(test_df)} rows ({test_df['label'].sum()} unsafe, {(test_df['label']==0).sum()} safe)")

    test_probs = get_probabilities(test_df["text"].tolist(), tokenizer, model, device)
    test_preds = (test_probs >= best_threshold).astype(int)

    print_metrics(test_df["label"].values, test_preds, test_probs, best_threshold)
    print_per_category(test_df, test_preds, test_probs)

    # save predictions so the notebook can plot calibration curve + confusion matrix heatmap
    test_df["prob_unsafe"] = test_probs
    test_df["pred"] = test_preds
    test_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"\nPredictions saved to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
