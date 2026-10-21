# Original task answer parsers extracted without importing the training entry points.

import re

from typing import Optional

VALID_LETTERS = set("ABCD")

def get_completion_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return completion[0].get("content", "") or ""
    return ""

def medical_answer(completion):
    # Same extraction as LMOps/opcd/verl/verl/trainer/ppo/reward.py:get_medmcqa_acc(version="v1").
    text = get_completion_text(completion)
    matches = re.findall(r"<answer>([\s\S]*?)<\/answer>", text, re.IGNORECASE)
    if not matches:
        return None
    answer = re.search(r"\(([ABCDabcd]?)\)", matches[-1])
    if answer and answer.group(1):
        return answer.group(1).upper()
    answer = re.search(r"[ABCDabcd]", matches[-1])
    return answer.group(0).upper() if answer else None

def _normalize_letter(snippet: str, case_sensitive: bool=False) -> Optional[str]:
    if not snippet:
        return None
    cleaned = re.sub('[\\*\\(\\)\\[\\]\\.\\:`\\"\']', ' ', snippet).strip()
    if case_sensitive:
        if cleaned in VALID_LETTERS:
            return cleaned
        m = re.match('\\s*([ABCD])\\b', cleaned)
        return m.group(1) if m else None
    else:
        if cleaned.upper() in VALID_LETTERS:
            return cleaned.upper()
        m = re.match('\\s*([ABCD])\\b', cleaned, flags=re.IGNORECASE)
        return m.group(1).upper() if m else None

def biology_answer(text: str) -> Optional[str]:
    """Extract A/B/C/D from a model response with multiple fallbacks."""
    if not text:
        return None
    text = re.sub('<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    matches = re.findall('<answer>\\s*(.*?)\\s*</answer>', text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        cand = _normalize_letter(matches[-1], case_sensitive=False)
        if cand:
            return cand
    m = re.search('<answer>\\s*(.*)', text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        cand = _normalize_letter(m.group(1), case_sensitive=False)
        if cand:
            return cand
    cue_patterns = ['(?:final\\s+answer|the\\s+answer\\s+is|answer\\s*[:\\-])\\s*\\(?\\*?\\*?([ABCDabcd])\\b', '\\boption\\s*\\(?\\*?\\*?([ABCDabcd])\\b', '\\b([ABCDabcd])\\s*(?:is\\s+(?:the\\s+)?correct|is\\s+right)', '\\\\boxed\\{\\s*\\(?([ABCDabcd])\\)?\\s*\\}']
    for pat in cue_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    m = re.search('^\\s*\\(?\\*?\\*?([ABCD])\\)?\\s*[\\.\\):]', text, flags=re.MULTILINE)
    if m:
        return m.group(1)
    for line in reversed(text.strip().splitlines()):
        stripped = line.strip().strip('*`()[].:"\' ')
        if stripped in VALID_LETTERS:
            return stripped
    return None
