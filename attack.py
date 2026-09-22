# coding=utf-8
"""Watermark robustness evaluation for Google Colab.

This script implements lightweight versions of the attacks discussed in
"A Watermark for Large Language Models" (arXiv:2301.10226v4).  It generates
one watermarked and one ordinary sample, applies several attack levels, and
writes an evaluation matrix to ``attack_results.csv``.

Optional model-based attacks (generative, LM span replacement, and T5 span
replacement) are skipped unless an attacker model is supplied.  The default
attacks are dependency-light and reproducible.
"""

import argparse
import csv
import os
import random
import re
import time
from argparse import Namespace
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import torch


LEVELS = (0.01, 0.05, 0.10, 0.30)
ZERO_WIDTH = "\u200b\u200c\u200d\ufeff"
HOMOGLYPHS = {
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "s": "ѕ",
    "x": "х", "y": "у", "A": "А", "B": "В", "C": "С", "E": "Е",
    "H": "Н", "K": "К", "M": "М", "O": "О", "P": "Р", "T": "Т",
    "X": "Х", "Y": "У",
}
SYNONYMS = {
    "big": "large", "small": "little", "important": "notable",
    "show": "demonstrate", "use": "utilize", "help": "assist",
    "common": "usual", "different": "distinct", "good": "useful",
    "make": "create", "many": "numerous", "begin": "start",
}
EMOJIS = ("🙂", "🔍", "✨", "🌊", "📚")


@dataclass
class Result:
    attack: str
    level: float
    source: str
    status: str
    z_score: Optional[float] = None
    green_fraction: Optional[float] = None
    token_count: Optional[int] = None
    detected: Optional[bool] = None
    char_change_ratio: Optional[float] = None
    length_ratio: Optional[float] = None
    elapsed_seconds: Optional[float] = None
    note: str = ""


def _positions(text: str, ratio: float, seed: int) -> list[int]:
    rng = random.Random(seed)
    count = min(len(text), max(1, round(len(text) * ratio)))
    return sorted(rng.sample(range(len(text)), count)) if text else []


def _indices(length: int, ratio: float, seed: int) -> list[int]:
    rng = random.Random(seed)
    count = min(length, max(1, round(length * ratio)))
    return sorted(rng.sample(range(length), count)) if length else []


def discreet(text: str, level: float, seed: int = 0) -> str:
    """Small whitespace/punctuation edits."""
    chars = list(text)
    for i in reversed(_positions(text, level, seed)):
        if chars[i].isalnum():
            chars[i] = chars[i].lower() if chars[i].isupper() else chars[i]
            if i + 1 < len(chars) and chars[i + 1] == " ":
                chars[i + 1] = "  "
    return "".join(chars)


def tokenization(text: str, level: float, seed: int = 0) -> str:
    """Change BPE boundaries by inserting spaces inside selected words."""
    words = list(re.finditer(r"[A-Za-z]{4,}", text))
    rng = random.Random(seed)
    rng.shuffle(words)
    target = max(1, round(len(words) * level)) if words else 0
    edits = set()
    for match in words[:target]:
        edits.add(match.start() + max(1, len(match.group()) // 2))
    return "".join((" " if i in edits else "") + ch for i, ch in enumerate(text))


def homoglyph(text: str, level: float, seed: int = 0) -> str:
    positions = [i for i, c in enumerate(text) if c in HOMOGLYPHS]
    for i in _indices(len(positions), level, seed):
        if i < len(positions):
            pos = positions[i]
            text = text[:pos] + HOMOGLYPHS[text[pos]] + text[pos + 1:]
    return text


def zero_width(text: str, level: float, seed: int = 0) -> str:
    """Insert invisible Unicode characters; level is insertion ratio."""
    rng = random.Random(seed)
    out = []
    interval = max(1, round(1.0 / max(level, 1e-6)))
    for i, char in enumerate(text):
        out.append(char)
        if (i + 1) % interval == 0:
            out.append(rng.choice(ZERO_WIDTH))
    return "".join(out)


def emoji(text: str, level: float, seed: int = 0) -> str:
    rng = random.Random(seed)
    words = text.split(" ")
    count = max(1, round(len(words) * level)) if words else 0
    for index in rng.sample(range(len(words)), min(count, len(words))):
        words[index] += rng.choice(EMOJIS)
    return " ".join(words)


def paraphrase(text: str, level: float, seed: int = 0) -> str:
    """A deterministic, lightweight paraphrase approximation."""
    words = text.split()
    rng = random.Random(seed)
    candidates = [i for i, word in enumerate(words) if word.strip(".,!?;:").lower() in SYNONYMS]
    rng.shuffle(candidates)
    for index in candidates[:max(1, round(len(candidates) * level))]:
        word = words[index]
        punctuation = "".join(c for c in word if c in ".,!?;:")
        bare = word.strip(".,!?;:")
        replacement = SYNONYMS.get(bare.lower(), bare)
        if bare[:1].isupper():
            replacement = replacement.capitalize()
        words[index] = replacement + punctuation
    return " ".join(words)


def insertion(text: str, level: float, seed: int = 0) -> str:
    rng = random.Random(seed)
    words = text.split(" ")
    count = max(1, round(len(words) * level)) if words else 0
    for _ in range(count):
        index = rng.randrange(len(words) + 1)
        words.insert(index, rng.choice(("however", "indeed", "notably", "also")))
    return " ".join(words)


def model_rewrite(text: str, level: float, model, tokenizer, device, seed: int = 0) -> str:
    """Optional generative attack using a supplied causal LM."""
    prompt = "Rewrite the following text while preserving its meaning and details:\n" + text
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    torch.manual_seed(seed)
    output = model.generate(**inputs, max_new_tokens=max(32, int(200 * level)), do_sample=True, temperature=0.8)
    generated = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    return generated or text


def t5_span(text: str, level: float, model, tokenizer, device, seed: int = 0) -> str:
    """Optional T5-style span replacement; requires a seq2seq infilling model."""
    words = text.split()
    if not words:
        return text
    rng = random.Random(seed)
    count = max(1, round(len(words) * level))
    for _ in range(min(count, len(words))):
        index = rng.randrange(len(words))
        original = words[index]
        masked = " ".join(words[:index] + ["<extra_id_0>"] + words[index + 1:])
        inputs = tokenizer(masked, return_tensors="pt", truncation=True, max_length=1024).to(device)
        output = model.generate(**inputs, max_new_tokens=8, num_beams=5)
        candidate = tokenizer.decode(output[0], skip_special_tokens=False)
        match = re.search(r"<extra_id_0>\s*(.*?)\s*(?:<extra_id_1>|$)", candidate)
        if match and match.group(1).strip() and match.group(1).strip() != original:
            words[index] = match.group(1).strip()
    return " ".join(words)


def score(text: str, args, tokenizer, device) -> dict:
    try:
        rows, _ = detect(text, args, device=device, tokenizer=tokenizer)
        values = {row[0]: row[1] for row in rows if row and row[0]}
        return {
            "z_score": float(values.get("z-score", "nan")),
            "green_fraction": float(str(values.get("Fraction of T in Greenlist", "nan")).rstrip("%")) / 100,
            "token_count": int(values.get("Tokens Counted (T)", 0)),
            "detected": values.get("Prediction") == "Watermarked",
        }
    except Exception as exc:
        return {"z_score": None, "green_fraction": None, "token_count": None, "detected": None, "error": str(exc)}


def build_args() -> Namespace:
    return Namespace(
        run_gradio=False, demo_public=False,
        model_name_or_path=os.environ.get("WATERMARK_MODEL", "facebook/opt-125m"),
        load_fp16=os.environ.get("WATERMARK_FP16", "false").lower() == "true",
        prompt_max_length=None, max_new_tokens=120, generation_seed=123,
        use_sampling=True, n_beams=1, sampling_temp=0.7,
        use_gpu=torch.cuda.is_available(), seeding_scheme="simple_1",
        gamma=0.25, delta=2.0, normalizers=[], ignore_repeated_bigrams=False,
        detection_z_threshold=4.0, select_green_tokens=True,
        skip_model_load=False, seed_separately=True,
    )


def evaluate(args: Namespace, output_csv: str, attacker_model=None, attacker_tokenizer=None) -> list[Result]:
    from demo_watermark import detect, generate, load_model
    model, tokenizer, device = load_model(args)
    prompt = ("The diamondback terrapin is a species of turtle native to coastal marshes. "
              "It has a distinctive shell and lives in brackish water. The species is")
    _, _, plain, watermarked, _ = generate(prompt, args, model=model, device=device, tokenizer=tokenizer)
    attacks: dict[str, Callable] = {
        "paraphrasing": paraphrase, "discreet_alterations": discreet,
        "tokenization": tokenization, "homoglyph": homoglyph,
        "zero_width": zero_width, "emoji": emoji, "generative": model_rewrite,
        "lm_span_replacement": t5_span, "t5_span_replacement": t5_span,
        "insertion": insertion,
    }
    results: list[Result] = []
    for attack_name, attack_fn in attacks.items():
        for level in LEVELS:
            for source, original in (("unwatermarked", plain), ("watermarked", watermarked)):
                started = time.perf_counter()
                try:
                    if attack_name in ("generative", "lm_span_replacement", "t5_span_replacement") and attacker_model is None:
                        raise RuntimeError("optional attacker model not supplied; skipped")
                    attacked = attack_fn(original, level, attacker_model, attacker_tokenizer, device) if attack_name in ("generative", "lm_span_replacement", "t5_span_replacement") else attack_fn(original, level, 123)
                    metrics = score(attacked, args, tokenizer, device)
                    note = metrics.pop("error", "")
                    status = "ok" if metrics.get("z_score") is not None else "error"
                    results.append(Result(attack_name, level, source, status, **metrics, char_change_ratio=1 - len(set(attacked) & set(original)) / max(1, len(set(original))), length_ratio=len(attacked) / max(1, len(original)), elapsed_seconds=time.perf_counter() - started, note=note))
                except Exception as exc:
                    results.append(Result(attack_name, level, source, "skipped", elapsed_seconds=time.perf_counter() - started, note=str(exc)))
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_csv", default="attack_results.csv")
    parser.add_argument("--attacker_model", default="", help="Optional local Hugging Face attacker model")
    args = build_args()
    cli = parser.parse_args()
    attacker_model = attacker_tokenizer = None
    if cli.attacker_model:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        attacker_tokenizer = AutoTokenizer.from_pretrained(cli.attacker_model)
        attacker_model = AutoModelForSeq2SeqLM.from_pretrained(cli.attacker_model).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    results = evaluate(args, cli.output_csv, attacker_model, attacker_tokenizer)
    print(f"Wrote {len(results)} rows to {cli.output_csv}")
    print("attack,level,source,status,z_score,green_fraction,detected")
    for row in results:
        print(f"{row.attack},{row.level:.2f},{row.source},{row.status},{row.z_score},{row.green_fraction},{row.detected}")


if __name__ == "__main__":
    main()
