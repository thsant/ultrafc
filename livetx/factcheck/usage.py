"""Soma o uso da API (tokens, buscas) por modelo, para estimar o custo da sessão."""

from __future__ import annotations

import collections

# US$ por milhão de tokens: (entrada, saída, leitura de cache). Escrita de cache = 1,25 × entrada.
PRICES = {"claude-sonnet-5-5": (2.0, 10.0, 0.20), "claude-opus-5-5": (4.0, 20.0, 0.20)}
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


def token_cost(model: str, t: collections.Counter) -> float | None:
    if model not in PRICES:
        return None
    inp, out, cache = PRICES[model]
    return (t["input_tokens"] * inp + t["output_tokens"] * out + t["cache_read_input_tokens"] * cache
            + t["cache_creation_input_tokens"] * inp * 1.25) / 1e6


def summary() -> list[str]:
    lines = []
    for model, t in _totals.items():
        cost = token_cost(model, t)
        lines.append(f"{model}: {t['chamadas']} chamadas, {t['input_tokens']:,} tokens de entrada "
                     f"(+{t['cache_read_input_tokens']:,} do cache), {t['output_tokens']:,} de saída, "
                     f"{t['buscas']} buscas, {t['leituras']} páginas lidas"
                     + (f" — tokens ≈ US$ {cost:.2f}" if cost is not None else ""))
    return lines
