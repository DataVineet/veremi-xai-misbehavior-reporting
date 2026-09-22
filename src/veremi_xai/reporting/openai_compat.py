"""One backend for every OpenAI-compatible chat endpoint (Groq, Gemini compat, OpenRouter, Mistral, Ollama, ...).

Switching provider = editing a profile in config.yaml (base_url / model / api_key_env). No SDK dependency.
"""
from __future__ import annotations

import os
import re
import time

import requests

from .base import ReporterError
from .prompts import SYSTEM, user_prompt


class OpenAICompatibleBackend:
    def __init__(self, name: str, base_url: str, model: str, api_key_env: str | None, temperature=0.0, max_tokens=700, timeout_s=60, extra: dict | None = None):
        self.name, self.base_url, self.model = name, base_url.rstrip("/"), model
        self.extra = extra or {}
        self.api_key_env, self.temperature, self.max_tokens, self.timeout_s = api_key_env, temperature, max_tokens, timeout_s

    def available(self) -> bool:
        return self.api_key_env is None or bool(os.environ.get(self.api_key_env))

    def generate(self, evidence: dict, feedback: str | None = None) -> str:
        if not self.available():
            raise ReporterError(f"environment variable {self.api_key_env} is not set")
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt(evidence)}]
        if feedback:
            messages.append({"role": "user", "content": "Your previous draft violated the rules: " + feedback + " Rewrite the full report and fix this."})
        headers = {"Content-Type": "application/json"}
        if self.api_key_env:
            headers["Authorization"] = f"Bearer {os.environ[self.api_key_env]}"
        payload = {"model": self.model, "messages": messages, "temperature": self.temperature, "max_tokens": self.max_tokens, **self.extra}
        last = "no attempt made"
        for wait in (0, 2, 6, 15):
            time.sleep(wait)
            try:
                r = requests.post(f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=self.timeout_s)
            except requests.RequestException as e:
                last = f"network error: {type(e).__name__}"
                continue
            if r.status_code == 200:
                try:
                    text = r.json()["choices"][0]["message"]["content"]
                except (KeyError, IndexError, ValueError, TypeError):
                    raise ReporterError("unexpected response format")
                text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()      # some reasoning models inline their thoughts
                if not text:
                    raise ReporterError("empty completion (for reasoning models raise max_tokens in the profile)")
                return text
            last = f"HTTP {r.status_code}: {r.text[:300]}"
            if r.status_code == 429:                       # honour the provider's "try again in 7.6s" / "2m3s" hint
                m = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", r.text)
                hint = (int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60 + float(m.group(3))) if m else 20.0
                if hint > 90:                              # daily quota, not a per-minute limit: waiting is pointless
                    break
                time.sleep(hint + 1.0)
                continue
            if r.status_code not in (408, 409, 500, 502, 503, 504):
                break
        raise ReporterError(last)
