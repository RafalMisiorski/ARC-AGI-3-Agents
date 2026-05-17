"""Per-call cost tracker for ARC-AGI-3 agent CLI invocations.

Mirrors the schema of NH's ``data/opus_credit_burn.jsonl`` so any
downstream tool that consumes burn-format JSONL (including NH's
``burn_summary``) can read this file too. Adds three ARC-specific
fields: ``game_id``, ``action_counter``, ``parse_ok``.

Why this exists: Day 1 of Phase 0 burned 170 EUR of the 180 EUR Claude
Max Extra-Usage cap because the race architecture kept 3 Claude
replicas billing in parallel even when Codex/Gemini won. We had **no
live cost monitoring during runs**. From Day 2 onwards: every CLI
subprocess call goes through ``record_call`` here.

Public API:
* ``record_call(...)`` -> float USD            -- appends one record
* ``session_cost_summary(game_id)`` -> dict    -- per-game totals
* ``sweep_cost_summary(game_ids)`` -> dict     -- multi-game totals
* ``pricing_for(provider, model)`` -> tuple    -- per-1M-token rates
* ``get_log_path()`` -> Path                   -- where the JSONL lives

The pricing table is intentionally **conservative** -- it uses the
Anthropic / OpenAI public API rates even though subscription quotas
typically cover the calls. Overestimating cost is safer than waking up
to a surprise charge.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

LOG_DIR = Path("logs")
LOG_PATH = LOG_DIR / "cost_burn.jsonl"

# (input_per_1M_USD, output_per_1M_USD). Cache reads ignored at v0 -- not
# every provider exposes cache token counts and the safest move for v0 is
# to bill them as regular input.
PRICING_USD_PER_1M: dict[str, tuple[float, float]] = {
    "opus":             (15.00, 75.00),  # Anthropic claude-opus API rate (5x sonnet)
    "sonnet":           (3.00, 15.00),   # Anthropic claude-sonnet API rate
    "haiku":            (1.00,  5.00),   # Anthropic claude-haiku API rate
    "codex_default":    (1.10,  4.40),   # OpenAI o4-mini rate (Codex CLI default)
    "gemini-flash-2.5": (0.0,   0.0),    # free Gemini CLI subscription
}

EUR_PER_USD = 0.88  # rough mid-May 2026 rate

# Rough conversion from USD to "% all-models weekly bucket on Max 20x".
# Derived from Day 1 empirics (170 EUR / 193 USD burned -> 75% Sonnet weekly
# + 62% all-models). The all-models bucket is bigger than Sonnet-only.
# This is a ±50% estimate -- ALWAYS cross-check with claude.ai/settings/usage
# rather than relying on this for hard quota decisions.
USD_PER_PERCENT_ALL_MODELS_WEEKLY: float = 2.0


def pricing_for(provider: str, model: str | None = None) -> tuple[float, float]:
    """Lookup pricing by model slug, falling back to provider key.

    Returns (0.0, 0.0) for unknown slugs -- the call still gets logged
    with zero cost rather than crashing.
    """
    if model:
        rate = PRICING_USD_PER_1M.get(model.lower())
        if rate is not None:
            return rate
    return PRICING_USD_PER_1M.get(provider.lower(), (0.0, 0.0))


def estimate_cost_usd(
    provider: str, model: str | None, usage: dict[str, Any]
) -> float:
    """Compute USD cost for a single call from token usage."""
    in_rate, out_rate = pricing_for(provider, model)
    in_tok = int(usage.get("input_tokens") or 0)
    out_tok = int(usage.get("output_tokens") or 0)
    return round((in_tok * in_rate + out_tok * out_rate) / 1_000_000, 6)


def record_call(
    provider: str,
    model: str,
    usage: dict[str, Any],
    duration_s: float,
    game_id: str,
    action_counter: int,
    parse_ok: bool,
    purpose: str = "arc_planner",
    status: str = "winner",
) -> float:
    """Append one cost-tracking record. Returns estimated USD.

    Args:
        status: "winner" | "empty" | "timeout" | "error" | "skipped"
            -- winner: this provider produced the plan we used.
            -- empty: returned text but unparseable (no actions found).
            -- timeout: subprocess killed at timeout (often = rate limit).
            -- error: subprocess errored / wasn't on PATH.
            -- skipped: cascade skipped this tier (e.g. budget too low).

    Safe to call from hook hot paths -- fails open on IO errors (logs are
    advisory; the agent continues regardless).
    """
    estimated_usd = estimate_cost_usd(provider, model, usage)
    record = {
        "call_id": uuid.uuid4().hex[:12],
        "model": model,
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "cache_read_tokens": int(usage.get("cache_read_tokens") or 0),
        "cache_creation_tokens": int(usage.get("cache_creation_tokens") or 0),
        "estimated_overage_usd": estimated_usd,
        "caller_module": "nh_arc_baseline_v0",
        "purpose": purpose,
        "duration_s": round(duration_s, 3),
        "backend": f"{provider}_cli",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "game_id": game_id,
        "action_counter": action_counter,
        "parse_ok": parse_ok,
        "status": status,
    }
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass
    return estimated_usd


def _iter_records():
    if not LOG_PATH.is_file():
        return
    try:
        with LOG_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except OSError:
        return


def session_cost_summary(game_id: str) -> dict[str, Any]:
    """Sum cost for one game across all calls in the log."""
    by_provider: dict[str, float] = {}
    total_usd = 0.0
    calls = 0
    for rec in _iter_records():
        if rec.get("game_id") != game_id:
            continue
        calls += 1
        cost = float(rec.get("estimated_overage_usd") or 0.0)
        total_usd += cost
        provider = str(rec.get("backend") or "").replace("_cli", "") or "unknown"
        by_provider[provider] = by_provider.get(provider, 0.0) + cost
    return {
        "game_id": game_id,
        "total_usd": round(total_usd, 4),
        "total_eur": round(total_usd * EUR_PER_USD, 4),
        "calls": calls,
        "by_provider": {k: round(v, 4) for k, v in by_provider.items()},
    }


def sweep_cost_summary(game_ids: list[str]) -> dict[str, Any]:
    """Sum cost across multiple games -- used by eval_local.py."""
    targets = set(game_ids)
    by_provider: dict[str, float] = {}
    by_game: dict[str, float] = {g: 0.0 for g in game_ids}
    total_usd = 0.0
    calls = 0
    for rec in _iter_records():
        gid = rec.get("game_id")
        if gid not in targets:
            continue
        calls += 1
        cost = float(rec.get("estimated_overage_usd") or 0.0)
        total_usd += cost
        by_game[gid] = by_game.get(gid, 0.0) + cost
        provider = str(rec.get("backend") or "").replace("_cli", "") or "unknown"
        by_provider[provider] = by_provider.get(provider, 0.0) + cost
    return {
        "total_usd": round(total_usd, 4),
        "total_eur": round(total_usd * EUR_PER_USD, 4),
        "calls": calls,
        "by_provider": {k: round(v, 4) for k, v in by_provider.items()},
        "by_game": {k: round(v, 4) for k, v in by_game.items()},
    }


def get_log_path() -> Path:
    return LOG_PATH


_NH_CACHE: dict[str, Any] = {"ts": 0.0, "data": None}
_NH_CACHE_TTL: float = 180.0  # 3 min -- NH endpoint takes ~6s, don't spam


def fetch_nh_usage(
    timeout: float = 10.0, force_refresh: bool = False
) -> dict[str, Any] | None:
    """Pull real Anthropic subscription %s from NH at localhost:8100.

    Returns None if NH isn't running or the call fails. Result cached
    for ``_NH_CACHE_TTL`` seconds because the endpoint takes ~6s per call
    and we invoke this from every planner-call hot path.

    NH counts raw output tokens against subscription rate-limit caps; the
    numbers are NOT identical to claude.ai/settings/usage (which weights
    by model price / compute / cache discount). NH's % is a CONSERVATIVE
    proxy -- typically reads 100% even when claude.ai shows 30-60%.
    Useful for visibility, NOT for hard-stop decisions.
    """
    now = time.time()
    if not force_refresh and (now - _NH_CACHE["ts"]) < _NH_CACHE_TTL:
        return _NH_CACHE["data"]

    try:
        import urllib.request
        req = urllib.request.Request(
            "http://localhost:8100/api/usage/limits",
            headers={"X-API-Key": "dev"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        result = {
            "session_percent": float(
                data.get("session", {}).get("percent_used", 0.0)
            ),
            "weekly_all_percent": float(
                data.get("weekly_all_models", {}).get("percent_used", 0.0)
            ),
            "weekly_sonnet_percent": float(
                data.get("weekly_sonnet", {}).get("percent_used", 0.0)
            ),
            "source": "nh_localhost_8100",
        }
    except Exception:
        result = None

    _NH_CACHE["ts"] = now
    _NH_CACHE["data"] = result
    return result


def today_total_usd(date_str: str | None = None) -> dict[str, Any]:
    """Sum cost across all records whose ts starts with ``date_str`` (YYYY-MM-DD).

    Defaults to today's UTC date.
    """
    if date_str is None:
        date_str = time.strftime("%Y-%m-%d", time.gmtime())
    total = 0.0
    by_provider: dict[str, float] = {}
    calls = 0
    for rec in _iter_records():
        ts = rec.get("ts", "")
        if not ts.startswith(date_str):
            continue
        calls += 1
        cost = float(rec.get("estimated_overage_usd") or 0.0)
        total += cost
        prov = str(rec.get("backend") or "").replace("_cli", "") or "unknown"
        by_provider[prov] = by_provider.get(prov, 0.0) + cost
    return {
        "date": date_str,
        "total_usd": round(total, 4),
        "total_eur": round(total * EUR_PER_USD, 4),
        "estimated_percent_all_models_weekly": round(
            total / USD_PER_PERCENT_ALL_MODELS_WEEKLY, 2
        ),
        "calls": calls,
        "by_provider": {k: round(v, 4) for k, v in by_provider.items()},
    }
