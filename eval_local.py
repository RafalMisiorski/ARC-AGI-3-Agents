"""Local eval harness for ARC-AGI-3 -- honors splits.yaml discipline.

Runs an agent sequentially across the envs of a chosen split (dev / train /
holdout) via subprocess.run(main.py --game=<id>). Parses the final scorecard
JSON from main.py's stdout to extract per-env ``levels_completed``. Writes a
JSONL summary to ``results/eval_<split>_<agent>_<timestamp>.jsonl`` plus
prints a readable table.

Per the plan (section "Decisions taken before Phase 0", point 5): max 3 live
submissions to the server before Phase 3. This script counts as ONE
submission per env it runs -- a dev sweep = 5 submissions when looking at it
from the server's side. Track usage.

Usage:
    uv run python eval_local.py --split=dev
    uv run python eval_local.py --split=train --agent=nharcbaselinev0
    uv run python eval_local.py --split=dev --timeout-per-env=900 --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
SPLITS_PATH = ROOT / "splits.yaml"
RESULTS_DIR = ROOT / "results"


def load_splits() -> dict[str, list[str]]:
    with SPLITS_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_scorecard_from_stdout(stdout: str) -> dict | None:
    """Find and parse the FINAL SCORECARD REPORT JSON block emitted by main.py."""
    marker = "--- FINAL SCORECARD REPORT ---"
    idx = stdout.find(marker)
    if idx < 0:
        return None
    # The JSON is logged across multiple lines after the marker.
    tail = stdout[idx + len(marker):]
    brace_start = tail.find("{")
    if brace_start < 0:
        return None
    # Track brace depth to find the matching close.
    depth = 0
    in_string = False
    escape = False
    end = -1
    for i, ch in enumerate(tail[brace_start:], start=brace_start):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return None
    raw = tail[brace_start:end]
    # Strip any logger prefixes like "2026-... | INFO |" at line starts.
    cleaned = re.sub(r"^\d{4}-\d{2}-\d{2}.*?\|.*?\|\s*", "", raw, flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def scorecard_url_from_stdout(stdout: str) -> str:
    m = re.search(r"View your scorecard online:\s*(\S+)", stdout)
    return m.group(1) if m else ""


def run_one_env(
    agent: str,
    env_id: str,
    timeout: int,
    extra_args: list[str] | None = None,
) -> dict:
    """Run main.py for a single env. Returns per-env result dict."""
    # sys.executable is the venv interpreter when this script runs under
    # `uv run`. The venv already has every dep main.py needs, so we invoke
    # main.py directly -- routing through `uv run` again would shell out
    # to the system Python which does NOT have uv as an importable module.
    cmd = [
        sys.executable, "main.py",
        "--agent", agent,
        "--game", env_id,
    ] + (extra_args or [])
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        elapsed = time.time() - t0
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
    except subprocess.TimeoutExpired as e:
        elapsed = time.time() - t0
        stdout = (e.stdout or b"").decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = (e.stderr or b"").decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")

    scorecard = parse_scorecard_from_stdout(stdout)
    levels = 0
    actions = 0
    state = "UNKNOWN"
    if scorecard:
        envs = scorecard.get("environments") or []
        if envs:
            levels = int(envs[0].get("levels_completed") or 0)
            actions = int(envs[0].get("actions") or 0)
            runs = envs[0].get("runs") or []
            if runs:
                state = runs[0].get("state") or "UNKNOWN"

    return {
        "env_id": env_id,
        "elapsed_seconds": round(elapsed, 1),
        "levels_completed": levels,
        "actions": actions,
        "state": state,
        "scorecard_url": scorecard_url_from_stdout(stdout),
        "scorecard_id": (scorecard or {}).get("card_id", ""),
        "parsed_scorecard": bool(scorecard),
    }


def _cost_for_game(env_id: str) -> tuple[float, dict[str, float]]:
    """Sum cost_burn.jsonl rows for one env. Returns (usd, by_provider).

    eval_local.py imports cost_tracker lazily so the module also works on
    boxes that don't have the tracker installed yet.
    """
    try:
        from agents.templates.cost_tracker import session_cost_summary
    except Exception:
        return 0.0, {}
    # game_id in cost_burn.jsonl is the FULL framework-issued id (e.g.
    # "ls20-9607627b"), not the bare env prefix "ls20". Fold both shapes.
    try:
        # Direct match on prefix-only key first.
        summary = session_cost_summary(env_id)
        if summary.get("calls", 0) > 0:
            return summary["total_usd"], summary["by_provider"]
    except Exception:
        pass
    # Fallback: scan log for any game_id starting with env_id-
    try:
        from agents.templates.cost_tracker import get_log_path
        import json as _json
        path = get_log_path()
        if not path.is_file():
            return 0.0, {}
        total = 0.0
        by_provider: dict[str, float] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                gid = rec.get("game_id", "")
                if gid != env_id and not gid.startswith(env_id + "-"):
                    continue
                cost = float(rec.get("estimated_overage_usd") or 0.0)
                total += cost
                prov = str(rec.get("backend") or "").replace("_cli", "") or "unknown"
                by_provider[prov] = by_provider.get(prov, 0.0) + cost
        return round(total, 4), {k: round(v, 4) for k, v in by_provider.items()}
    except Exception:
        return 0.0, {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "train", "holdout"], required=True)
    ap.add_argument("--agent", default="nharcbaselinev0")
    ap.add_argument(
        "--timeout-per-env", type=int, default=1800,
        help="Per-env wall-clock cap in seconds (default 1800 = 30 min). "
             "Day 1 used 900 and killed 4/5 envs mid-flight; agent's internal "
             "budget cap stops earlier when needed.",
    )
    ap.add_argument(
        "--cap-usd-per-env", type=float, default=5.0,
        help="Expected $/env hard cap (matches agent's BUDGET_HARD_USD). "
             "Sweep aborts early if cumulative spend exceeds 2x this times the "
             "number of envs scheduled.",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="List the envs that would run; do not invoke main.py.",
    )
    args = ap.parse_args()

    splits = load_splits()
    envs = splits.get(args.split) or []
    if not envs:
        print(f"ERROR: split {args.split!r} has no envs in splits.yaml", file=sys.stderr)
        sys.exit(2)

    if args.dry_run:
        print(f"[dry-run] split={args.split} agent={args.agent}")
        for e in envs:
            print(f"  - {e}")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_{args.split}_{args.agent}_{ts}.jsonl"

    sweep_cap_usd = args.cap_usd_per_env * len(envs)
    abort_threshold_usd = 2.0 * sweep_cap_usd

    print(
        f"split={args.split} agent={args.agent} envs={len(envs)} "
        f"timeout/env={args.timeout_per_env}s  per-env-cap=${args.cap_usd_per_env:.2f}  "
        f"sweep-cap=${sweep_cap_usd:.2f}  abort-at=${abort_threshold_usd:.2f}"
    )
    print(f"writing to {out_path}")
    print()

    results: list[dict] = []
    cumulative_cost_usd = 0.0
    aborted = False

    with out_path.open("w", encoding="utf-8") as f:
        for i, env_id in enumerate(envs, 1):
            print(f"[{i}/{len(envs)}] {env_id} ... ", end="", flush=True)
            r = run_one_env(args.agent, env_id, args.timeout_per_env)

            # Pull this env's cost from cost_burn.jsonl (written by agent).
            env_cost_usd, env_by_provider = _cost_for_game(env_id)
            r["cost_usd"] = env_cost_usd
            r["cost_by_provider"] = env_by_provider
            cumulative_cost_usd += env_cost_usd
            r["cumulative_cost_usd"] = round(cumulative_cost_usd, 4)

            results.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()

            print(
                f"levels={r['levels_completed']} actions={r['actions']} "
                f"state={r['state']} elapsed={r['elapsed_seconds']:.0f}s "
                f"cost=${env_cost_usd:.4f}  cumulative=${cumulative_cost_usd:.4f}"
            )
            if r["scorecard_url"]:
                print(f"        scorecard: {r['scorecard_url']}")

            if cumulative_cost_usd > abort_threshold_usd:
                aborted = True
                print(
                    f"\n[SWEEP ABORT] cumulative ${cumulative_cost_usd:.4f} "
                    f"> 2x sweep cap ${sweep_cap_usd:.2f}. Skipping remaining "
                    f"{len(envs) - i} env(s)."
                )
                break

    # Summary
    print()
    print("=" * 60)
    print(f"split={args.split} agent={args.agent}{'  (ABORTED)' if aborted else ''}")
    print(f"envs run:           {len(results)} of {len(envs)}")
    completed = [r["levels_completed"] for r in results if r["parsed_scorecard"]]
    if completed:
        mean = sum(completed) / len(completed)
        print(f"mean levels_completed: {mean:.2f}")
        print(f"per env levels:        {dict(zip([r['env_id'] for r in results], completed))}")
    else:
        print("No parseable scorecards -- check stdout for failures.")
    print(f"total wall-clock:      {sum(r['elapsed_seconds'] for r in results):.0f}s")
    print(f"total cost:            ${cumulative_cost_usd:.4f} "
          f"(~EUR {cumulative_cost_usd * 0.88:.2f})")
    print(f"per env cost:          {dict(zip([r['env_id'] for r in results], [r['cost_usd'] for r in results]))}")
    print(f"results jsonl:         {out_path}")
    print(f"cost detail jsonl:     logs/cost_burn.jsonl")


if __name__ == "__main__":
    main()
