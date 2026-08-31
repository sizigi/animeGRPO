"""Build English train.parquet (~900 sentences) matching the JP train data
profile (Wikipedia-style encyclopedic prose).

JP source (jp_tts_mixed_v2): 900 encyclopedic sentences from Japanese
Wikipedia-style text, mean 27.6 chars (range 14-58 chars).

EN equivalent target:
  - Source: wikitext-103-raw-v1 (Salesforce/wikitext) — pre-cleaned Wikipedia EN
  - Length: 80-180 chars (~13-30 words) — proportional info density to JP
  - 900 sentences, deterministic seed=0 shuffle
  - Quality filters: no markup tokens, balanced parens, sentence ends with ./?/!,
    starts with capital, no Wikipedia article headings (== xxx ==), no URLs
  - Schema matches JP test_v2.parquet so verl trainer can swap data files

Output: ~/data/llasa-tts-rl-grpo-en-v2/train.parquet
"""
import os, re, random, sys
from pathlib import Path
import pandas as pd

OUT_DIR = Path("~/data/llasa-tts-rl-grpo-en-v2").expanduser()
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_TRAIN = 900
LEN_MIN, LEN_MAX = 80, 180
SEED = 0

# Patterns to reject (markup, headings, URLs, bullet, code, lists)
REJECT_PATTERNS = [
    r"==.*==",          # Wikipedia headings
    r"http[s]?://",     # URLs
    r"^\s*\d+\s*$",     # bare numbers
    r"\[.*\]",          # bracket markup [edit] [1]
    r"^\s*\*",          # bullet lists
    r"^\s*#",           # numbered lists
    r"<\w+>",           # HTML-like tags
    r"&\w+;",           # HTML entities
]
REJECT_RE = re.compile("|".join(REJECT_PATTERNS))

# Sentence splitter: split on ". " "! " "? " not followed by lowercase/digit
SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def detokenize_wikitext103(s: str) -> str:
    """Reverse wikitext-103 BPE-style artifacts."""
    # Numeric/punctuation atoms: @-@ @.@ @,@ @ -> bound
    s = s.replace(" @-@ ", "-").replace(" @.@ ", ".").replace(" @,@ ", ",")
    s = s.replace("@-@", "-").replace("@.@", ".").replace("@,@", ",")
    # Spaces around punctuation
    s = re.sub(r"\s+([,.;:!?'\"])", r"\1", s)         # remove space BEFORE punct
    s = re.sub(r"\s+'\s*([a-z])", r"'\1", s)            # " 's" -> "'s"
    s = re.sub(r"\(\s+", "(", s)                         # "( x" -> "(x"
    s = re.sub(r"\s+\)", ")", s)                         # "x )" -> "x)"
    s = re.sub(r"\s+", " ", s).strip()                   # collapse spaces
    return s


def good_sentence(s: str) -> bool:
    s = s.strip()
    if not (LEN_MIN <= len(s) <= LEN_MAX):
        return False
    if not re.match(r"^[A-Z\"'`]", s):
        return False
    if s[-1] not in ".!?":
        return False
    if REJECT_RE.search(s):
        return False
    # Must contain a space (multi-word) and at least one letter
    if " " not in s or not re.search(r"[a-zA-Z]", s):
        return False
    # Reject sentences with too many digits (lists of stats)
    n_digits = sum(c.isdigit() for c in s)
    if n_digits > len(s) * 0.20:
        return False
    # Reject sentences with too many parens (likely tables/abbrev-heavy)
    if s.count("(") + s.count(")") > 4:
        return False
    return True


def main():
    print("Loading wikitext-103-raw-v1 (train split)...")
    from datasets import load_dataset
    # Use streaming to avoid full load — wikitext-103 is ~500MB
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)

    rng = random.Random(SEED)
    sentences = []
    seen = set()
    SAMPLE_TARGET = NUM_TRAIN * 5  # over-sample to allow filtering / dedup

    print("Streaming + filtering sentences...")
    for i, row in enumerate(ds):
        text = row.get("text", "").strip()
        if not text:
            continue
        # Wikitext-103 has paragraphs separated; each row is a chunk
        for sent in SENT_SPLIT_RE.split(text):
            sent = detokenize_wikitext103(sent.strip())
            if not good_sentence(sent):
                continue
            # Dedup
            if sent in seen:
                continue
            seen.add(sent)
            sentences.append(sent)
            if len(sentences) >= SAMPLE_TARGET:
                break
        if len(sentences) >= SAMPLE_TARGET:
            break
        if i and i % 50000 == 0:
            print(f"  scanned {i} rows, kept {len(sentences)} sentences")

    print(f"\nFiltered candidates: {len(sentences)}")
    rng.shuffle(sentences)
    chosen = sentences[:NUM_TRAIN]
    print(f"Final: {len(chosen)}")

    rows = []
    for idx, text in enumerate(chosen):
        prompt = [
            {"content": f"Convert the text to speech:<|TEXT_UNDERSTANDING_START|>{text}<|TEXT_UNDERSTANDING_END|>",
             "role": "user"},
            {"content": "<|SPEECH_GENERATION_START|>", "role": "assistant"},
        ]
        rows.append({
            "data_source":  "en_tts_mixed_v2",
            "prompt":       prompt,
            "ability":      "text-to-speech",
            "reward_model": {"ground_truth": text, "style": "rule"},
            "extra_info":   {"index": idx, "split": "train", "text": text},
        })

    df = pd.DataFrame(rows)
    out = OUT_DIR / "train.parquet"
    df.to_parquet(out, index=False)
    print(f"\nwrote {out}")
    import numpy as np
    lens = [len(r["text"]) for r in [e["extra_info"] for _, e in df.iterrows()]]
    print(f"length: mean={np.mean(lens):.1f} median={np.median(lens):.0f} "
          f"min={min(lens)} max={max(lens)}")
    print(f"\nfirst 5 samples:")
    for i in range(5):
        print(f"  [{i}] {df.iloc[i]['extra_info']['text']}")


if __name__ == "__main__":
    main()
