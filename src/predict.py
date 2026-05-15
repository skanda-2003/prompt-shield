import re
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

# anchor to this file's location so the path works regardless of where Python is run from
MODEL_PATH = str(Path(__file__).parent.parent / "models" / "promptshield")
MAX_LENGTH = 256   # must match train.py - same truncation used during training
THRESHOLD = 0.45   # tuned on validation set in evaluate.py - 0.45 gave best macro F1

# module-level cache so the model is loaded once and reused on every call
_tokenizer = None
_model = None
_device = None


def _load_model():
    """Load model and tokenizer into the module-level cache if not already loaded."""
    global _tokenizer, _model, _device
    if _model is not None:
        return  # already loaded - skip

    print(f"Loading model from {MODEL_PATH}...")
    _tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    _model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

    # eval mode turns off dropout so inference is deterministic
    _model.eval()

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = _model.to(_device)
    print(f"Model loaded. Running on: {_device}")


def _assign_attack_type(text: str) -> str:
    """
    Keyword-based attack category classifier. Called only when the model predicts unsafe.
    Uses the same regex patterns as 01_data_exploration.ipynb so categories are consistent.
    Returns one of the 5 named categories, or "unknown" if no pattern matches.
    """
    t = text.lower()

    # role-playing jailbreaks: asking the model to adopt an unrestricted persona
    if re.search(
        r"\b(act as|pretend|you are now|jailbreak|dan|unrestricted|no restrictions|without restrictions)\b"
        r"|\b(character|persona)\b"
        r"|\b(evil|amoral|unfiltered|uncensored|unhinged|immoral)\b"
        r"|content.policy|safety.guideline"
        r"|(always|never) (respond|answer|say|tell|refuse)"
        r"|from now on"
        r"|(ai|bot|assistant|chatbot).{0,20}(named|called|known as)"
        r"|\bnot (bound|restricted|limited|constrained).{0,40}(rule|policy|ethic|restriction|guideline)"
        r"|\b(roleplay|role-play)\b"
        r"|in the role of"
        r"|(escape|free from|liberated from).{0,30}(restriction|rule|matrix|filter)"
        r"|\bno (ethics|morals|limits|filters|guidelines)\b",
        t,
    ):
        return "role_playing"

    # instruction override: directly telling the model to ignore its system prompt
    if re.search(
        r"\b(ignore|disregard|forget|override|bypass)\b.{0,60}\b(instruction|prompt|rule|guideline|previous)\b"
        r"|\b(new task|new instruction|new role|new objective)\b",
        t,
    ):
        return "instruction_override"

    # prompt leaking: trying to extract the system prompt or find out what the model was told
    if re.search(
        r"\b(repeat|print|show|reveal|output|tell me).{0,30}\b(above|previous|system|prompt|instruction)\b"
        r"|\b(what|tell|show)\b.{0,50}\b(system prompt|initial prompt|original instructions?|context window)\b"
        r"|\bwhat (is|are|was|were).{0,30}(your|the).{0,20}(prompt|instruction|guideline|directive)\b"
        r"|\b(what|tell me).{0,30}\b(your instructions|you were (told|given|instructed|programmed))\b",
        t,
    ):
        return "prompt_leaking"

    # encoded attacks: base64, rot13, or other encoding to bypass keyword filters
    if re.search(r"\b(base64|rot13|decode|encoded|cipher)\b", t):
        return "encoded_attacks"

    # context manipulation: instructions hidden inside documents, code blocks, tags, or structured content
    if re.search(
        r"(```|\[system\]|<instructions>|note:|important:).{0,50}(ignore|follow|must|shall)"
        r"|<\s*(system|sys|instructions?|admin|root)\s*>"
        r"|\[(system|sys|instructions?|admin|override|inst)\]"
        r"|(note|important|notice|attention|warning)\s*:.{0,100}(ignore|disregard|forget|override|bypass)"
        r"|<!--.{0,200}(ignore|disregard|override|bypass).{0,100}-->"
        r"|(summarize|translate|analyze|review|read|process).{0,400}(ignore|disregard|forget|override|bypass).{0,80}(instruction|prompt|rule|previous|above)",
        t,
    ):
        return "context_manipulation"

    # unsafe but didn't match any known pattern
    return "unknown"


def predict(prompt: str) -> dict:
    """
    Run inference on a single prompt string.

    Returns a dict with three keys:
      - is_safe (bool): True if the prompt is safe, False if it is an attack
      - confidence (float): probability of the predicted class, rounded to 4 decimal places
      - attack_type (str | None): one of the 5 attack categories, "unknown", or None if safe
    """
    _load_model()

    # tokenize with the same settings used in train.py
    encoding = _tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",  # return PyTorch tensors directly
    )

    input_ids = encoding["input_ids"].to(_device)
    attention_mask = encoding["attention_mask"].to(_device)

    # no_grad skips gradient tracking - saves memory and speeds up inference
    with torch.no_grad():
        outputs = _model(input_ids=input_ids, attention_mask=attention_mask)

    # softmax converts logits to probabilities - probs[0] = P(safe), probs[1] = P(unsafe)
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    prob_unsafe = probs[1].item()

    is_safe = prob_unsafe < THRESHOLD

    # confidence = probability of whichever class was predicted
    confidence = round(1.0 - prob_unsafe if is_safe else prob_unsafe, 4)

    # attack_type is None for safe prompts, keyword-classified for unsafe ones
    attack_type = None if is_safe else _assign_attack_type(prompt)

    return {"is_safe": is_safe, "confidence": confidence, "attack_type": attack_type}


if __name__ == "__main__":
    # quick command-line test: python src/predict.py "ignore all previous instructions"
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter a prompt: ")
    result = predict(prompt)

    label = "SAFE" if result["is_safe"] else "UNSAFE"
    print(f"\n[{label}]  confidence: {result['confidence']:.4f}  attack_type: {result['attack_type']}")