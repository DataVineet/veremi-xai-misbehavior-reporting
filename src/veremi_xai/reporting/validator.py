"""Automatic faithfulness validation of a report against its evidence package.

Checks (all must pass):
  structure        the five required headings, in order
  prediction       the predicted class is named; no class outside {prediction, listed alternatives} is mentioned
  no_override      no language that disputes or replaces the ML prediction
  numbers          every number in the text exists in the evidence (exactly, rounded, or as a percentage)
  features         every feature name mentioned is part of the evidence package
  length           at most 400 words
"""
from __future__ import annotations

import re

from ..config import class_names
from ..features import DESCRIPTIONS
from .prompts import SECTIONS

NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?:\s?%)?")
DASHES = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2212": "-", "\u00a0": " ", "\u202f": " ", "\u2009": " "})   # LLMs emit typographic hyphens/spaces
OVERRIDE = re.compile(r"\b(misclassif\w*|incorrect(ly)? (prediction|classif\w*)|should (instead )?be classified|is actually an?|"
                      r"more likely (to be )?an?|i (believe|think|suspect)|the (true|real|actual) class)\b", re.I)


def _numbers_in(obj, out: set) -> set:
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, (int, float)):
        out.add(float(obj))
    elif isinstance(obj, str):
        for m in NUM.findall(normalise(obj)):
            out.add(float(m.rstrip("% ")))
    elif isinstance(obj, dict):
        for v in obj.values():
            _numbers_in(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _numbers_in(v, out)
    return out


_SUP = str.maketrans("⁻⁺⁰¹²³⁴⁵⁶⁷⁸⁹", "-+0123456789")


def normalise(text: str) -> str:
    """Undo LLM typography so that numbers can be compared with the evidence:
    '57\u202f596.435' / '57,596.435' -> 57596.435 ; '1.43 × 10⁻¹²' / '1.43 x 10^-12' -> 1.43e-12 ; typographic hyphens/spaces -> ASCII."""
    text = re.sub(r"(?<=\d)[\u202f\u2009\u00a0,](?=\d{3}(?!\d))", "", text)
    text = text.translate(DASHES)
    text = re.sub(r"(\d)\s*[×x\*]\s*10\s*\^?\s*\(?([-+⁻⁺]?[\d⁰¹²³⁴⁵⁶⁷⁸⁹]+)\)?", lambda m: f"{m.group(1)}e{m.group(2).translate(_SUP)}", text)
    return text


def _keys_in(obj, out: set) -> set:
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.add(k)
            _keys_in(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _keys_in(v, out)
    return out


def _classes_in(text: str, names: dict) -> set:
    """Class names mentioned in `text` (longest names first so 'DoS' inside 'DoS random' is not double-counted)."""
    found = set()
    for nm in sorted(names.values(), key=len, reverse=True):
        pat = re.compile(rf"\b{re.escape(nm)}s?\b")        # case-sensitive: "random position offsets" in prose is a description, not a class claim
        if pat.search(text):
            found.add(nm)
            text = pat.sub("¤", text)
    return found


def _strings_in(obj, out: list) -> list:
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _strings_in(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _strings_in(v, out)
    return out


def _supported(tok: str, allowed: set) -> bool:
    """A number in the text is supported if it equals an evidence number exactly, after rounding the evidence
    number to the precision used in the text, as a percentage of it, or with the sign dropped ("a negative
    contribution of 0.31"). Known limitation: existence is checked, not which feature the number is attached to."""
    pct = tok.endswith("%")
    raw = tok.rstrip("% ")
    n = float(raw)
    dec = len(raw.split(".")[1]) if "." in raw and "e" not in raw.lower() else 0
    forms = [(n, dec)] + ([(n / 100, dec + 2)] if pct else [])
    for v in allowed:
        for c, d in forms:
            if round(abs(v), d) == round(abs(c), d):
                return True
    return False


def validate(text: str, ev: dict) -> dict:
    text = normalise(text)
    violations, checks = [], {}
    names = class_names()
    pred = ev["prediction"]["class_name"]
    ok_classes = {pred} | {a["class_name"] for a in ev["prediction"]["alternatives"]}

    heads = [h.strip().lower() for h in re.findall(r"^#{1,3}\s*(.+?)\s*$", text, flags=re.M)]
    checks["structure"] = heads == [s.lower() for s in SECTIONS]
    if not checks["structure"]:
        violations.append(f"headings must be exactly {SECTIONS}; found {heads}")

    mentioned = _classes_in(text, names)
    ok_classes |= _classes_in(ev["prediction"]["definition"], names)      # a definition may name a related class ("DoS disruptive in which ...")
    extra = mentioned - ok_classes
    named = re.search(rf"\b{re.escape(pred)}\b", text, flags=re.I) is not None
    checks["prediction"] = named and not extra
    if not named:
        violations.append(f"predicted class '{pred}' is not named")
    if extra:
        violations.append(f"mentions classes that are not in the evidence: {sorted(extra)}")

    m = OVERRIDE.search(text)
    checks["no_override"] = m is None
    if m:
        violations.append(f"language that disputes/overrides the ML prediction: '{m.group(0)}'")

    body = re.sub(r"^#{1,3}.*$", "", text, flags=re.M)
    body = re.sub(r"^\s*\d+[.)]\s+", "", body, flags=re.M)             # ordered-list markers
    allowed = _numbers_in(ev, set())
    toks = NUM.findall(body)
    bad = [t for t in toks if not _supported(t, allowed)]
    checks["numbers"] = not bad
    if bad:
        violations.append(f"numbers not found in the evidence: {sorted(set(bad))[:8]}")

    in_ev = {c["feature"] for c in ev["shap"]["top_contributions"]} | set(ev["transmitted_values"])
    in_ev |= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", " ".join(_strings_in(ev, []))))   # names quoted inside evidence text (descriptions)
    words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))
    unknown = sorted((words & set(DESCRIPTIONS)) - in_ev)
    ticked = set(re.findall(r"`([^`]+)`", text))
    allowed_ticks = _keys_in(ev, set()) | in_ev | ok_classes | set(_strings_in(ev["model"], []))
    unknown += sorted(t for t in ticked if t not in DESCRIPTIONS and t not in allowed_ticks and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", t))
    checks["features"] = not unknown
    if unknown:
        violations.append(f"feature names that are not in the evidence: {unknown[:8]}")

    n_words = sum(1 for w in text.split() if re.search(r"[A-Za-z0-9]", w))      # markdown table pipes/dashes are not words
    checks["length"] = n_words <= 400
    if not checks["length"]:
        violations.append("report longer than 400 words")
    return {"passed": all(checks.values()), "checks": checks, "violations": violations, "numbers_checked": len(toks)}
