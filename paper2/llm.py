"""Chat-model adapters for the optional LLM arms (extraction, llm_gate baseline, answering).

Offline mode needs none of this. Two backends:
  AnthropicClient     Claude through the official `anthropic` SDK (adaptive thinking, effort knob).
                      Sampling is not controllable (see its docstring), so it is NOT the client for the
                      pre-registered "one fixed small instruct model at temperature 0" arm.
  OpenAICompatClient  a *local open-weight instruct model* behind vLLM / Ollama / llama.cpp, which
                      expose /v1/chat/completions. This is the client for the pre-registered fixed
                      answerer and extractor (temperature 0, optional request seed); it is not a route
                      to Claude.

Caching (critique A6.6). Every completion is cached on disk under paper2/out/cache so reruns are free
and deterministic. The cache is keyed by model, temperature, seed and repeat index as well as by the
prompt, so three extraction seeds or k = 5 repeated answers are k distinct calls, never one cached
answer copied k times. `repeat=None` is the unrepeated call; `repeat=0..k-1` are the k samples.

Accounting (critique A7/D). Usage (calls, tokens, cache hits, replayed tokens) is accumulated per client
and `CachedClient.cost_summary(n_candidates)` renders it as calls and tokens per 1,000 candidates for
the write-path table.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "paper2" / "out" / "cache"

ANSWER_SYSTEM = ("You answer questions about a conversation using only the memory facts provided. Reply with the answer "
                 "in a few words, no explanation. If the facts do not contain the answer, reply exactly: No information available.")

_USAGE_COUNTERS = ("calls", "input_tokens", "output_tokens", "cache_hits", "cached_input_tokens", "cached_output_tokens")


def _fmt_temperature(temperature: float | None) -> str:
    """'default' when the backend controls sampling (Anthropic under thinking), else the float as written."""
    return "default" if temperature is None else repr(float(temperature))


class CachedClient:
    """Disk-cached, usage-counted completion client. Subclasses implement `_call`.

    `_call(prompt, system, max_tokens, seed=None)` may return a plain string, `(text, input_tokens,
    output_tokens)`, or `(text, input_tokens, output_tokens, extra)` where `extra` is a dict of further
    integer usage counters (e.g. Anthropic's cache_read_input_tokens) that are summed into `usage`; any
    non-integer entries are stored in the cache record under "meta". A `_call` without a `seed`
    parameter is still called with the old three arguments.

    `model` and `temperature` belong in the namespace (the cache sub-directory) and are hashed into every
    cache key as well, together with `seed` and `repeat`, so distinct sampling settings never share an
    entry even if a subclass builds its own namespace string.
    """

    def __init__(self, namespace: str, cache_dir: Path | str = CACHE, *, model: str | None = None,
                 temperature: float | None = None, seed: int | None = None):
        self.namespace = namespace
        self.model, self.temperature, self.seed = model, temperature, seed
        self.cache_dir = Path(cache_dir) / re.sub(r"[^A-Za-z0-9._-]+", "_", namespace)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage = {k: 0 for k in _USAGE_COUNTERS}

    # -- subclass hook -------------------------------------------------------------------------------
    def _call(self, prompt: str, system: str | None, max_tokens: int, seed: int | None = None):
        raise NotImplementedError

    # -- cache key -----------------------------------------------------------------------------------
    def _effective_seed(self, seed: int | None) -> int | None:
        return self.seed if seed is None else seed

    def cache_key_for(self, prompt: str, system: str | None = None, cache_key: str | None = None, max_tokens: int = 1024,
                      seed: int | None = None, repeat: int | None = None) -> str:
        """The sha256 cache key for one completion: namespace, model, temperature, seed, repeat, prompt."""
        parts = [self.namespace, self.model, _fmt_temperature(self.temperature),
                 None if self._effective_seed(seed) is None else int(self._effective_seed(seed)),
                 None if repeat is None else int(repeat), system, prompt, cache_key, int(max_tokens)]
        return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()

    # -- dispatch ------------------------------------------------------------------------------------
    def _dispatch(self, prompt: str, system: str | None, max_tokens: int, seed: int | None):
        params = inspect.signature(self._call).parameters
        accepts_seed = "seed" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        out = self._call(prompt, system, max_tokens, seed=seed) if accepts_seed else self._call(prompt, system, max_tokens)
        if isinstance(out, str):
            return out, 0, 0, {}
        out = tuple(out)
        text = out[0]
        tin = int(out[1]) if len(out) > 1 and out[1] is not None else 0
        tout = int(out[2]) if len(out) > 2 and out[2] is not None else 0
        extra = dict(out[3]) if len(out) > 3 and out[3] else {}
        return text, tin, tout, extra

    def complete(self, prompt: str, system: str | None = None, cache_key: str | None = None, max_tokens: int = 1024,
                 seed: int | None = None, repeat: int | None = None) -> str:
        """One completion. `seed` (sampling seed, passed to the backend when it supports one) and `repeat`
        (index of a repeated sample, 0..k-1) are part of the cache key; both default to None."""
        eff_seed = self._effective_seed(seed)
        key = self.cache_key_for(prompt, system, cache_key, max_tokens, seed=eff_seed, repeat=repeat)
        f = self.cache_dir / f"{key}.json"
        if f.exists():
            rec = json.loads(f.read_text())
            self.usage["cache_hits"] += 1
            self.usage["cached_input_tokens"] += int(rec.get("input_tokens") or 0)
            self.usage["cached_output_tokens"] += int(rec.get("output_tokens") or 0)
            return rec["text"]
        text, tin, tout, extra = self._dispatch(prompt, system, max_tokens, eff_seed)
        self.usage["calls"] += 1
        self.usage["input_tokens"] += tin
        self.usage["output_tokens"] += tout
        meta = {}
        for k, v in extra.items():
            if isinstance(v, bool) or not isinstance(v, int):
                meta[k] = v
            else:
                self.usage[k] = self.usage.get(k, 0) + v
        rec = {"text": text, "input_tokens": tin, "output_tokens": tout, "cache_key": cache_key, "namespace": self.namespace,
               "model": self.model, "temperature": self.temperature, "seed": eff_seed, "repeat": repeat, "max_tokens": max_tokens,
               **{k: v for k, v in extra.items() if k not in meta}}
        if meta:
            rec["meta"] = meta
        tmp = f.with_name(f"{f.name}.{os.getpid()}.tmp")   # per-process name: parallel runs may race on one key
        tmp.write_text(json.dumps(rec))
        os.replace(tmp, f)  # atomic: a crash mid-write never leaves a half record that later reads as a hit
        return text

    def complete_json(self, prompt: str, cache_key: str | None = None, max_tokens: int = 1024, system: str | None = None,
                      seed: int | None = None, repeat: int | None = None) -> dict | None:
        text = self.complete(prompt, system=system, cache_key=cache_key, max_tokens=max_tokens, seed=seed, repeat=repeat)
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None
        try:
            v = json.loads(m.group(0))
            return v if isinstance(v, dict) else None
        except json.JSONDecodeError:
            return None

    # -- accounting ----------------------------------------------------------------------------------
    def cost_summary(self, n_candidates: int) -> dict:
        """Write-path accounting for the per-1,000-candidate table (critique A7/D).

        `requests` is every completion the pipeline needed, whether this process paid for it (`calls`) or
        replayed it from disk (`cache_hits`); `tokens_*` likewise include replayed tokens read back from
        the cache records, so a cached rerun reports the same cost as the run that filled the cache.
        `fresh_*` are this process's paid calls only. Per-1,000 rates are None when n_candidates <= 0.
        """
        u = self.usage
        n = int(n_candidates)
        requests = u["calls"] + u["cache_hits"]
        tin = u["input_tokens"] + u["cached_input_tokens"]
        tout = u["output_tokens"] + u["cached_output_tokens"]
        per_1k = (lambda x: round(1000.0 * x / n, 3)) if n > 0 else (lambda x: None)
        out = {"namespace": self.namespace, "model": self.model, "temperature": self.temperature, "n_candidates": n,
               "requests": requests, "calls": u["calls"], "cache_hits": u["cache_hits"],
               "input_tokens": tin, "output_tokens": tout, "tokens": tin + tout,
               "fresh_input_tokens": u["input_tokens"], "fresh_output_tokens": u["output_tokens"],
               "calls_per_1k": per_1k(requests), "input_tokens_per_1k": per_1k(tin), "output_tokens_per_1k": per_1k(tout),
               "tokens_per_1k": per_1k(tin + tout)}
        for k in ("cache_read_input_tokens", "cache_creation_input_tokens"):
            if k in u:
                out[k] = u[k]
        return out


class AnthropicClient(CachedClient):
    """Claude through the official SDK. Credentials come from the environment (ANTHROPIC_API_KEY); nothing is hard-coded.

    Sampling caveat (critique A6.5). The current models (Claude Opus 5.5, Claude Fable 5.1, Claude Haiku
    5.5) run adaptive thinking that cannot be disabled (`thinking: {"type": "disabled"}` is rejected), and
    the Messages API offers no temperature / top_p / top_k control under thinking. Depth is steered only
    by `output_config.effort`. Consequently this client cannot implement the pre-registered "one fixed
    small instruct model at temperature 0" arm (pre-reg section 6.1 and 6.6), nor the "3 seeds at
    temperature 0.3" seed-variance estimate: use `OpenAICompatClient` (`local:<base_url>:<model>:0.0`)
    for those. Here `seed` and `repeat` separate cache entries so repeated answers are genuinely
    repeated requests, but nothing about the request pins the sample; the namespace records
    `temp=default` for the same reason.

    The call shape (`messages.create` with `thinking={"type": "adaptive"}` and `output_config={"effort":
    ...}`) follows the SDK documentation and must stay as it is. Usage fields that exist on `resp.usage`
    are recorded: `input_tokens`, `output_tokens`, and `cache_read_input_tokens` /
    `cache_creation_input_tokens` when present (they are None when prompt caching was not used).
    """

    def __init__(self, model: str = "claude-opus-5-5", effort: str | None = "low", thinking: bool = True, **kw):
        # thinking=False sends {"type": "disabled"}: accepted on Claude Haiku 5.5 at effort high or below (not on
        # Opus 5.5 / Sonnet 5.5, which reject it). Without it, thinking tokens count against max_tokens and the
        # short answer / gate budgets truncate (15% of gate calls at 256 tokens in the 19 Oct smoke test).
        think = "" if thinking else ":think=off"
        super().__init__(namespace=f"anthropic:{model}:{effort}{think}:temp=default", model=model, temperature=None, **kw)
        import anthropic  # imported lazily so offline runs never need the package

        self.client = anthropic.Anthropic()
        self.effort = effort
        self.thinking = thinking

    def _call(self, prompt, system, max_tokens, seed=None):
        # `seed` is intentionally unused: the Messages API has no sampling seed. It is still part of the cache key.
        kwargs = dict(model=self.model, max_tokens=max_tokens, thinking={"type": "adaptive" if self.thinking else "disabled"},
                      messages=[{"role": "user", "content": prompt}])
        if system:
            kwargs["system"] = system
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        resp = self.client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if b.type == "text")
        usage = resp.usage
        extra: dict = {}
        for k in ("cache_read_input_tokens", "cache_creation_input_tokens"):
            v = getattr(usage, k, None)
            if v is not None:
                extra[k] = int(v)
        stop = getattr(resp, "stop_reason", None)
        if stop is not None:
            extra["stop_reason"] = str(stop)
        return text, int(usage.input_tokens), int(usage.output_tokens), extra


class OpenAICompatClient(CachedClient):
    """Local open model server (vLLM, Ollama, llama.cpp) speaking the /v1/chat/completions shape.

    `temperature` (default 0.0) goes in every request body and in the cache namespace. `seed` is the
    client-wide default sampling seed; a per-call `seed` passed to `complete()` overrides it. Whichever is
    in effect is sent as `"seed"` in the request body (vLLM and llama.cpp honour it; Ollama accepts it
    under `options`, which is not the chat-completions shape and is ignored there) and is part of the
    cache key either way.
    """

    def __init__(self, base_url: str, model: str, api_key: str | None = None, temperature: float = 0.0,
                 seed: int | None = None, **kw):
        temperature = float(temperature)
        super().__init__(namespace=f"local:{model}:temp={_fmt_temperature(temperature)}", model=model,
                         temperature=temperature, seed=seed, **kw)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("LOCAL_LLM_API_KEY", "none")

    def request_body(self, prompt: str, system: str | None, max_tokens: int, seed: int | None = None) -> dict:
        """The JSON body a call would send (exposed so tests can check it without a server)."""
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body = {"model": self.model, "messages": msgs, "temperature": self.temperature, "max_tokens": max_tokens}
        eff_seed = self._effective_seed(seed)
        if eff_seed is not None:
            body["seed"] = int(eff_seed)
        return body

    def _call(self, prompt, system, max_tokens, seed=None):
        body = json.dumps(self.request_body(prompt, system, max_tokens, seed)).encode()
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body,
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.loads(r.read().decode())
        u = d.get("usage", {})
        return d["choices"][0]["message"]["content"], int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))


def _parse_local_spec(rest: str) -> tuple[str, str, float]:
    """'<base_url>:<model>[:<temperature>]' -> (base_url, model, temperature). base_url may contain ':'.

    The trailing field is a temperature only when it parses as a float in [0, 2] AND a model field remains
    before it; otherwise the last field is the model and the temperature is 0.0 (the pre-registered
    default for the fixed answerer).
    """
    head, _, tail = rest.rpartition(":")
    temperature = 0.0
    if head and ":" in head:
        try:
            t = float(tail)
        except ValueError:
            t = None
        if t is not None and 0.0 <= t <= 2.0:
            temperature = t
            head, _, tail = head.rpartition(":")
    base_url, model = head, tail
    if not base_url or not model:
        raise ValueError(f"local spec needs '<base_url>:<model>[:<temperature>]', got {rest!r}")
    return base_url, model, temperature


def make_client(spec: str | None):
    """spec: 'anthropic:<model>[:<effort>]' or 'local:<base_url>:<model>[:<temperature>]' (base_url may contain ':').

    The local temperature defaults to 0.0. Only the local client can realise the pre-registered fixed
    answerer at temperature 0 (see AnthropicClient's docstring).
    """
    if not spec:
        return None
    kind, _, rest = spec.partition(":")
    if kind == "anthropic":
        model, _, effort = rest.partition(":")
        effort, _, flag = effort.partition(":")                  # 'anthropic:<model>:<effort>[:nothink]'
        return AnthropicClient(model=model or "claude-opus-5-5", effort=effort or "low", thinking=flag != "nothink")
    if kind == "local":
        base, model, temperature = _parse_local_spec(rest)
        return OpenAICompatClient(base_url=base, model=model, temperature=temperature)
    raise ValueError(f"unknown client spec {spec!r}")


TEMPORAL_HINT = "Use DATE of CONVERSATION to answer with an approximate date."   # the official LoCoMo suffix for category 2


def answer_question(client: CachedClient, question: str, facts: list[str], cache_key: str,
                    seed: int | None = None, repeat: int | None = None, hint: str | None = None) -> str:
    """Fixed answer prompt. `repeat` (0..k-1) makes the k repeated answers of the H2 design distinct calls.
    `hint` is the official per-category suffix (TEMPORAL_HINT for temporal questions), nothing else."""
    q = f"{question} {hint}" if hint else question
    prompt = "Memory facts:\n" + "\n".join(f"- {f}" for f in facts) + f"\n\nQuestion: {q}\nAnswer:"
    return client.complete(prompt, system=ANSWER_SYSTEM, cache_key=cache_key, max_tokens=128, seed=seed, repeat=repeat).strip()
