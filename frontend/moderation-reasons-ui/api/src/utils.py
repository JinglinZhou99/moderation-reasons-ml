import re
from typing import Dict, List, Tuple

VIOLENCE_WORDS = [r"kill", r"threat", r"hurt", r"beat", r"shoot", r"bomb"]
SEXUAL_WORDS = [r"sex", r"explicit", r"nude", r"porn", r"nsfw", r"obscene"]
HATE_PHRASES = [r"go back to", r"you people", r"\b(.*?)(chink|spic|kike|fag)\b", r"dirty (immigrant|[a-z]+)"]

KEYS = {
    "violence": VIOLENCE_WORDS,
    "sexual": SEXUAL_WORDS,
    "hate": HATE_PHRASES,
}

def find_spans(text: str, patterns: List[str]) -> List[Tuple[int, int]]:
    spans = []
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            spans.append((m.start(), m.end()))
    spans.sort()
    merged = []
    for s,e in spans:
        if not merged or s > merged[-1][1]:
            merged.append([s,e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return [(s,e) for s,e in merged]

def mock_probs(text: str) -> Dict[str, float]:
    out = {}
    for label, pats in KEYS.items():
        spans = find_spans(text, pats)
        p = min(0.95, 0.2 + 0.3*len(spans)) if spans else 0.05
        out[label] = round(p, 3)
    return out

def reasons_from_spans(text: str, spans_by_label: Dict[str, List[Tuple[int,int]]]) -> Dict[str, str]:
    r = {}
    for label, spans in spans_by_label.items():
        if not spans:
            continue
        snippet = text[spans[0][0]:spans[0][1]]
        if label == "violence":
            r[label] = f"Likely threat because of: \"{snippet}\"."
        elif label == "sexual":
            r[label] = f"Sexual content indicated by: \"{snippet}\"."
        else:
            r[label] = f"Identity-based attack near: \"{snippet}\"."
    return r
