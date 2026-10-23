"""Soma o uso da API (tokens, buscas) por modelo, para estimar o custo da sessão."""

from __future__ import annotations

import collections
import json
from pathlib import Path

from .files import now, write_atomic

# US$ por milhão de tokens: (entrada, saída, leitura de cache). Escrita de cache = 1,25 × entrada.
# Haiku 5.5: tarifa de prompts até 100K tokens; acima disso é 5× (não contabilizado).
PRICES = {"claude-sonnet-5-5": (2.0, 10.0, 0.20), "claude-opus-5-5": (4.0, 20.0, 0.20),
          "claude-haiku-5-5": (0.10, 0.50, 0.01)}
SEARCH_PRICE = 10.0 / 1000  # US$ por busca na web; ler páginas não é cobrado à parte
FIELDS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")

_totals: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)


def record(model: str, usage) -> None:
    t = _totals[model]
    t["chamadas"] += 1
    for f in FIELDS:
        t[f] += getattr(usage, f, 0) or 0
    server = getattr(usage, "server_tool_use", None)
    t["buscas"] += getattr(server, "web_search_requests", 0) or 0
    t["leituras"] += getattr(server, "web_fetch_requests", 0) or 0


def cost(model: str, t: collections.Counter) -> float | None:
    """Tokens + buscas, em US$."""
    if model not in PRICES:
        return None
    inp, out, cache = PRICES[model]
    return (t["input_tokens"] * inp + t["output_tokens"] * out + t["cache_read_input_tokens"] * cache
            + t["cache_creation_input_tokens"] * inp * 1.25) / 1e6 + t["buscas"] * SEARCH_PRICE


def summary() -> list[str]:
    lines = []
    for model, t in _totals.items():
        c = cost(model, t)
        lines.append(f"{model}: {t['chamadas']} chamadas, {t['input_tokens']:,} tokens de entrada "
                     f"(+{t['cache_read_input_tokens']:,} do cache), {t['output_tokens']:,} de saída, "
                     f"{t['buscas']} buscas, {t['leituras']} páginas lidas"
                     + (f" — ≈ US$ {c:.2f}" if c is not None else ""))
    return lines


def save(path: Path, **extra) -> None:
    """Grava o uso acumulado da sessão (sobrescreve: o arquivo sempre tem o total até agora)."""
    models = {m: {**t, "custo_usd": cost(m, t)} for m, t in _totals.items()}
    total = sum(m["custo_usd"] or 0 for m in models.values())
    write_atomic(path, json.dumps({**extra, "atualizado_em": now(), "custo_total_usd": round(total, 4),
                                   "modelos": models}, ensure_ascii=False, indent=2) + "\n")
