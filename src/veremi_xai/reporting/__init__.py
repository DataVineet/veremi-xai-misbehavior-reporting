"""LLM-assisted reporting layer: provider-agnostic, validated, cached, with a safe template fallback.

    report = generate_report(evidence)                 # backend from config.yaml (auto)
    report = generate_report(evidence, backend="groq") # force a profile; falls back to template if unusable
"""
from __future__ import annotations

import json
import os

from ..config import load_config, load_dotenv, path
from . import template_backend
from .base import Report, ReporterError
from .openai_compat import OpenAICompatibleBackend
from .validator import validate

__all__ = ["generate_report", "available_backends", "Report", "validate"]


def _profile(name: str) -> OpenAICompatibleBackend:
    llm = load_config()["llm"]
    p = llm["profiles"][name]
    return OpenAICompatibleBackend(name, p["base_url"], p["model"], p.get("api_key_env"), llm["temperature"],
                                   p.get("max_tokens", llm["max_tokens"]), llm["timeout_s"], p.get("extra"))


def available_backends() -> dict[str, bool]:
    """name -> usable right now? (template is always usable; keyless profiles such as ollama are listed as selectable)."""
    load_dotenv()
    out = {"template": True}
    for name in load_config()["llm"]["profiles"]:
        out[name] = _profile(name).available()
    return out


def _resolve(backend: str | None) -> str:
    load_dotenv()
    llm = load_config()["llm"]
    backend = backend or llm["backend"]
    if backend != "auto":
        return backend
    for name, p in llm["profiles"].items():
        if p.get("api_key_env") and os.environ.get(p["api_key_env"]):
            return name
    return "template"


def _template(ev: dict, pv: str, reason: str | None = None, rejected: str | None = None) -> Report:
    text = template_backend.render(ev)
    return Report(text=text, backend="template", model="deterministic-template", prompt_version=pv, evidence_hash=ev["evidence_hash"],
                  validation=validate(text, ev), fallback_reason=reason, rejected_llm_text=rejected)


def generate_report(evidence: dict, backend: str | None = None, use_cache: bool = True) -> Report:
    pv = load_config()["llm"]["prompt_version"]
    name = _resolve(backend)
    if name == "template":
        return _template(evidence, pv)
    be = _profile(name)
    cache_file = path("report_cache") / f"{evidence['evidence_hash']}_{pv}_{name}_{be.model.replace('/', '-').replace(':', '-')}.json"
    if use_cache and cache_file.exists():
        d = json.loads(cache_file.read_text(encoding="utf-8"))
        d["from_cache"] = True
        return Report(**d)
    attempts = 1
    try:
        text = be.generate(evidence)
        val = validate(text, evidence)
        if not val["passed"]:                                   # one corrective retry, then fall back
            first, attempts = text, 2
            text = be.generate(evidence, feedback="; ".join(val["violations"]))
            val = validate(text, evidence)
            if not val["passed"]:
                rep = _template(evidence, pv, "LLM report failed faithfulness validation twice: " + "; ".join(val["violations"]), text or first)
                rep.attempts = 2
                cache_file.write_text(json.dumps(rep.to_dict(), indent=1), encoding="utf-8")   # deterministic at temperature 0: do not pay for it again
                return rep
    except ReporterError as e:
        return _template(evidence, pv, f"LLM backend '{name}' unavailable: {e}")
    rep = Report(text=text, backend=name, model=be.model, prompt_version=pv, evidence_hash=evidence["evidence_hash"], validation=val, attempts=attempts)
    cache_file.write_text(json.dumps(rep.to_dict(), indent=1), encoding="utf-8")
    return rep
