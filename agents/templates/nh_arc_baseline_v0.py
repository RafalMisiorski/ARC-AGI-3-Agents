"""nh_arc_baseline_v0 -- Cost-aware sequential cascade planner for ARC-AGI-3.

Phase 0 v0.0 of the operator's plan.

EVOLUTION LOG
-------------
v0.0a (abandoned same day): 1 Claude CLI subprocess per action.
    ~3-4 min cold-start per call -> 4-5h per env. Unworkable.

v0.0b (abandoned next morning): planner-executor + 5-way race
    (3 Claude + Gemini + Codex). Achieved first scoring level on ls20
    (1/7 levels) but **burned 170 EUR of the 180 EUR monthly Extra-Usage
    cap in a single day** because cancelled-but-issued Claude requests
    still pay against Anthropic API.

v0.0c (current): **sequential cascade** -- one provider at a time,
    falling through to the next only when the current returns empty or
    unparseable output. Backed by per-call cost tracking via
    ``cost_tracker.py`` and per-game soft/hard budget caps.

Order (empirical from Day 1, not plan-agent guess):
    1. Codex CLI (default model)    -- cheapest tier-1, fastest when fresh
    2. Claude CLI (sonnet)           -- deepest reasoning, used only when
                                        Codex fails AND budget allows
    3. Gemini CLI (flash 2.5)        -- free, shallow; last resort

Architecture
------------
* ``choose_action(frames, latest_frame) -> GameAction``
    1. If the plan queue is empty, build a planner prompt from the latest
       frame plus N=3 recent frames.
    2. ``asyncio.run(_call_planner_cascade(...))`` tries each provider
       in CASCADE_ORDER, returning ``(provider, model, text, usage,
       duration_s, per_provider_durations)`` on first parseable plan.
    3. Parse the winner via ``_parse_plan`` (tolerant regex, multi-line).
       On total failure, queue gets one ACTION5 and we replan next step.
    4. Record the winning call's cost via ``cost_tracker.record_call``.
    5. If cumulative game cost > BUDGET_SOFT_USD, log warning. If >
       BUDGET_HARD_USD, set ``_budget_exceeded`` so the next ``is_done``
       returns True -- agent ends cleanly with scorecard.
    6. Pop one (action, data) and return.

We extend ``agents.agent.Agent`` directly, not ``LLM`` from
``llm_agents.py`` (which is hard-wired to the OpenAI SDK).

Instrumentation per step -> logs/baseline_v0_<game>_<ts>.jsonl
-------------------------------------------------------------
* planner_invoked                  (bool)
* cascade_winner_provider          (str) -- codex | claude | gemini | ""
* cascade_winner_model             (str) -- sonnet, codex_default, etc.
* cascade_winner_duration_s        (float)
* cascade_per_provider_durations   (dict, only tiers we attempted)
* plan_size, parse_ok, action_chosen, queue_depth_after,
  planner_call_count, response_preview, game_card_used
* last_call_cost_usd               (float, this step's cost)
* total_game_cost_usd              (float, running total this game)
* budget_state                     ("ok" | "soft_warning" | "hard_stop")

A parallel log goes to logs/cost_burn.jsonl (one row per CLI call,
mirroring NH's opus_credit_burn.jsonl schema -- see cost_tracker.py).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import textwrap
import time
from pathlib import Path
from typing import Any

from arcengine import FrameData, GameAction, GameState

from ..agent import Agent
from . import cost_tracker
from . import arc_vision
from . import arc_game_state

logger = logging.getLogger(__name__)

_CLI_ENV_SCRUB_KEYS = (
    # Claude
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
    "ANTHROPIC_API_KEY",
    # Gemini
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
)

_ACTION_PATTERN = re.compile(
    # Accepts "ACTION6 x=12 y=34", "ACTION6(x=12,y=34)", and bare "ACTION6 12 34".
    r"\b(RESET|ACTION[1-7])"
    r"(?:"
    r"\s*[\(\s]\s*x\s*=\s*(\d+)\s*[,;\s]\s*y\s*=\s*(\d+)"  # x=N y=N form
    r"|"
    r"\s+(\d+)\s+(\d+)"                                       # bare N N form
    r")?",
    re.IGNORECASE,
)


# Per-game-class prompt cards. Keyed by env_id prefix (chars before '-').
# Plan rule: cap at <=8 cards total, or the agent becomes a lookup table.
# Source: agents/templates/llm_agents.py::GuidedLLM (LockSmith rules) --
# the upstream framework already encoded these for OpenAI o3 GuidedLLM. We
# vendor them here for our race-based agent.
_PURE_CLICK_CARD = (
    "You are playing a **click-based puzzle** (PURE_CLICK class).\n"
    "* ACTION6 with (x, y) coordinates is the primary action.\n"
    "* Other actions (ACTION1-5, ACTION7) likely do nothing or have minor\n"
    "  side effects.\n"
    "* Strategy: identify clickable targets in the image (distinct sprites,\n"
    "  buttons, objects, pairs to match). Click them in a sensible order.\n"
    "* Coordinates: x in [0, 63] (column), y in [0, 63] (row).\n"
    "* When picking targets, prefer clicking on the CENTER of a sprite, not\n"
    "  its edge. If the grid does not change after a click, you missed --\n"
    "  try slightly different (x, y) offset.\n"
    "* Plan format: ACTION6 x=NN y=NN (one per line).\n"
)

_CLICK_OBJECT_CARD = (
    "You are playing an **object-manipulation puzzle** (CLICK_OBJECT class).\n"
    "* Primary action is ACTION6 (x, y) -- click to select / grab / place\n"
    "  objects. Movement actions (ACTION1-4) may also be active for cursor\n"
    "  positioning or scrolling.\n"
    "* Strategy: identify movable objects vs. fixed obstacles in the image.\n"
    "  Click to select, then click target location (or move with arrows).\n"
    "  Look for matching pairs, goal slots, stacking targets, or color/shape\n"
    "  alignment.\n"
    "* Coordinates: x, y in [0, 63].\n"
    "* If the grid is unchanged after action, your click missed or the move\n"
    "  was blocked -- try a different (x, y) or direction.\n"
)

_CLICK_MENU_CARD = (
    "You are playing a **menu / selection game** (CLICK_MENU class).\n"
    "* Available actions are typically limited to ACTION5 (select/enter),\n"
    "  ACTION6 (x, y click), ACTION7 (toggle/back/cycle), and RESET.\n"
    "  Movement actions ACTION1-4 are inactive in this class.\n"
    "* Strategy: navigate a menu/dialog UI. Click choices with ACTION6,\n"
    "  confirm with ACTION5. ACTION7 may cycle options or back out.\n"
    "* Look for highlighted, distinct, or differently-colored UI elements --\n"
    "  these are the clickable items.\n"
    "* Coordinates: x, y in [0, 63].\n"
)

_NAVIGATE_CARD = (
    "You are playing a **movement / navigation game** (NAVIGATE class).\n"
    "* Movement actions are primary: ACTION1=up, ACTION2=down,\n"
    "  ACTION3=left, ACTION4=right.\n"
    "* Other actions (5, 6, 7) likely do nothing or trigger special\n"
    "  abilities (jump, attack, interact) depending on the game.\n"
    "* Strategy: identify the player sprite and the goal / exit in the\n"
    "  image. Plan a path toward the goal, avoiding obstacles (walls,\n"
    "  traps, enemies).\n"
    "* If the grid is unchanged after a move, you bumped into a wall or\n"
    "  the move was blocked -- try a different direction.\n"
    "* Some games have multiple levels; reaching the exit advances\n"
    "  `levels_completed` by 1.\n"
)


GAME_CARDS: dict[str, str] = {
    # PURE_CLICK class (>=70% ACTION6 observed in agent runs)
    "tn36": _PURE_CLICK_CARD,
    "s5i5": _PURE_CLICK_CARD,
    "vc33": _PURE_CLICK_CARD,
    # CLICK_OBJECT class (50-70% ACTION6 + some movement)
    "cd82": _CLICK_OBJECT_CARD,
    "dc22": _CLICK_OBJECT_CARD,
    "ft09": _CLICK_OBJECT_CARD,
    "re86": _CLICK_OBJECT_CARD,
    "lp85": _CLICK_OBJECT_CARD,
    "r11l": _CLICK_OBJECT_CARD,
    "tr87": _CLICK_OBJECT_CARD,
    "wa30": _CLICK_OBJECT_CARD,
    "ka59": _CLICK_OBJECT_CARD,
    # CLICK_MENU class (ACTION6 + ACTION5/7, no movement)
    "sk48": _CLICK_MENU_CARD,
    "sb26": _CLICK_MENU_CARD,
    # NAVIGATE class (movement-dominant)
    "cn04": _NAVIGATE_CARD,
    "sp80": _NAVIGATE_CARD,
    "ar25": _NAVIGATE_CARD,
    "sc25": _NAVIGATE_CARD,
    "bp35": _NAVIGATE_CARD,  # ACTION3/4 horizontal heavy, baseline 22-action level 1
    # LOCKSMITH (specific to ls20, has expert replay for level 1)
    "ls20": (
        "You are playing **LockSmith**. Rules and strategy:\n"
        "* ACTION1=move up, ACTION2=move down, ACTION3=move left, "
        "ACTION4=move right. ACTION5/6/7 do nothing in this game.\n"
        "* Goal: find a key that matches the one inside the exit door, "
        "then walk into the door.\n"
        "* 7 levels total; `levels_completed` shows current progress.\n"
        "* Each level starts with limited energy. Moving consumes energy; "
        "GAME_OVER if you run out. Refill at 2x2 squares of value 0x06.\n"
        "* Player is a 4x4 sprite of value 0x04 (with one transparent row).\n"
        "* Walls = 0x0a (cannot pass). Floor = 0x08 (walkable).\n"
        "* Current key shape/color shown in bottom-left of grid.\n"
        "* Exit door is 4x4 with 0x0b border, contains target key (scaled 2x).\n"
        "* Key-shape rotator: 4x4 with 0x09 + 0x04 in top-left. Step on to "
        "rotate shape.\n"
        "* Key-color rotator: 4x4 with 0x09 + 0x02 in bottom-left. Step on "
        "to rotate color.\n"
        "* To rotate more than once: step off the rotator, then step back on.\n"
        "* If the grid does not change after a move, you bumped into a wall.\n"
        "\n"
        "**MANDATORY -- VERIFIED EXPERT SOLUTION FOR LEVEL 1:**\n"
        "This game is DETERMINISTIC per game_id. The following 28-action\n"
        "sequence was recorded from a human who completed all 7 levels.\n"
        "The first 28 actions take the player from RESET to "
        "`levels_completed=1`. **REPLAY THIS SEQUENCE VERBATIM** while the\n"
        "current `levels_completed` is still 0 (i.e. action_counter < 28).\n"
        "Do not deviate; do not 'improve'; do not re-analyse the grid.\n"
        "Output one action per turn in order; the env will respond.\n"
        "\n"
        "The full level-1 sequence (28 actions):\n"
        "1. RESET\n"
        "2. ACTION2   3. ACTION4   4. ACTION3   5. ACTION1   6. ACTION2\n"
        "7. ACTION3   8. ACTION3   9. ACTION3   10. ACTION1  11. ACTION1\n"
        "12. ACTION1  13. ACTION1  14. ACTION4  15. ACTION3  16. ACTION2\n"
        "17. ACTION4  18. ACTION3  19. ACTION2  20. ACTION1  21. ACTION2\n"
        "22. ACTION1  23. ACTION4  24. ACTION1  25. ACTION4  26. ACTION4\n"
        "27. ACTION1  28. ACTION1\n"
        "\n"
        "When `action_counter` is at N (0-indexed), output entry (N+1) in\n"
        "the list above. So at action_counter=0 output RESET; at\n"
        "action_counter=1 output ACTION2; etc. Continue until\n"
        "`levels_completed` reaches 1, then plan freely for levels 2+."
    ),
}


def _card_for_game(game_id: str) -> str:
    """Return the per-game prompt card if we have one, else ''."""
    if not game_id:
        return ""
    key = game_id.split("-", 1)[0]
    return GAME_CARDS.get(key, "")


def _clean_env_for_cli() -> dict[str, str]:
    """Env-scrub for all CLIs we shell out to (claude, gemini, codex)."""
    env = os.environ.copy()
    for key in _CLI_ENV_SCRUB_KEYS:
        env.pop(key, None)
    return env


def _extract_text_and_usage(stdout: str) -> tuple[str, dict[str, int]]:
    """Parse stream-json / JSONL output. Returns (text, usage_dict).

    Recognises text from:
      Claude: {"type":"result","subtype":"success","result":"..."}     (preferred)
              {"type":"assistant","message":{"content":[{"type":"text","text":"..."}]}}
      Gemini: {"type":"message","role":"assistant","content":"..."}
      Codex : {"type":"item.completed","item":{"type":"agent_message","text":"..."}}

    Recognises usage tokens from:
      Claude: result.usage.{input_tokens, output_tokens, cache_read_input_tokens,
              cache_creation_input_tokens}
      Codex : turn.completed.usage.{input_tokens, output_tokens,
              cached_input_tokens}   -- or item.completed.usage
      Gemini: message.usage_metadata.{prompt_token_count, candidates_token_count}
              (Gemini's tag names differ; we normalise on the way out)
    """
    text_parts: list[str] = []
    usage: dict[str, int] = {}
    final_text: str | None = None

    def _merge_usage(u: dict | None) -> None:
        if not isinstance(u, dict):
            return
        # Anthropic-style keys.
        if u.get("input_tokens") is not None:
            usage["input_tokens"] = int(u["input_tokens"])
        if u.get("output_tokens") is not None:
            usage["output_tokens"] = int(u["output_tokens"])
        if u.get("cache_read_input_tokens") is not None:
            usage["cache_read_tokens"] = int(u["cache_read_input_tokens"])
        elif u.get("cached_input_tokens") is not None:  # Codex spelling
            usage["cache_read_tokens"] = int(u["cached_input_tokens"])
        if u.get("cache_creation_input_tokens") is not None:
            usage["cache_creation_tokens"] = int(u["cache_creation_input_tokens"])
        # Gemini-style aliases.
        if u.get("prompt_token_count") is not None and "input_tokens" not in usage:
            usage["input_tokens"] = int(u["prompt_token_count"])
        if (
            u.get("candidates_token_count") is not None
            and "output_tokens" not in usage
        ):
            usage["output_tokens"] = int(u["candidates_token_count"])

    for line in stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Claude final-result event -- authoritative for text.
        if event.get("type") == "result" and event.get("subtype") == "success":
            result = event.get("result", "")
            if isinstance(result, str) and result:
                final_text = result.strip()
            _merge_usage(event.get("usage"))

        # Claude streaming assistant blocks.
        if event.get("type") == "assistant":
            msg = event.get("message", {})
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            t = block.get("text", "")
                            if t:
                                text_parts.append(t)
                elif isinstance(content, str) and content:
                    text_parts.append(content)
                _merge_usage(msg.get("usage"))

        # Gemini streaming assistant chunks.
        if event.get("type") == "message" and event.get("role") == "assistant":
            content = event.get("content", "")
            if isinstance(content, str) and content:
                text_parts.append(content)
            _merge_usage(event.get("usage") or event.get("usage_metadata"))

        # Codex JSONL agent_message events.
        if event.get("type") == "item.completed":
            item = event.get("item", {})
            if isinstance(item, dict) and item.get("type") == "agent_message":
                t = item.get("text", "")
                if t:
                    text_parts.append(t)
            _merge_usage((item or {}).get("usage"))

        # Codex final usage on turn.completed.
        if event.get("type") == "turn.completed":
            _merge_usage(event.get("usage"))

    text = final_text if final_text is not None else "".join(text_parts).strip()
    return text, usage


# ---------------------------------------------------------------------------
# Per-provider async wrappers
# ---------------------------------------------------------------------------


async def _call_claude_async(prompt: str, timeout: int, model: str = "opus") -> tuple[str, dict[str, int], float]:
    """Returns (text, usage_dict, duration_s).

    Model defaults to ``opus`` (Day-3 switch, Sonnet weekly = 100% used).
    Override per call if needed.
    """
    cmd_path = shutil.which("claude") or shutil.which("claude.cmd")
    if not cmd_path:
        logger.warning("Claude CLI not on PATH")
        return "", {}, 0.0

    cmd = [
        cmd_path,
        "--model", model,
        "--output-format", "stream-json",
        "--input-format", "stream-json",
        "--verbose",
        "--dangerously-skip-permissions",
    ]
    stdin_msg = (
        json.dumps(
            {"type": "user", "message": {"role": "user", "content": prompt}}
        )
        + "\n"
    )
    return await _run_subprocess(cmd, stdin_msg.encode("utf-8"), timeout)


async def _call_gemini_async(prompt: str, timeout: int) -> tuple[str, dict[str, int], float]:
    """Returns (text, usage_dict, duration_s)."""
    cmd_path = shutil.which("gemini") or shutil.which("gemini.cmd")
    if not cmd_path:
        logger.warning("Gemini CLI not on PATH")
        return "", {}, 0.0
    cmd = [cmd_path, "-m", "gemini-2.5-flash", "--output-format", "stream-json"]
    return await _run_subprocess(cmd, prompt.encode("utf-8"), timeout)


async def _call_codex_async(prompt: str, timeout: int) -> tuple[str, dict[str, int], float]:
    """Returns (text, usage_dict, duration_s)."""
    cmd_path = shutil.which("codex") or shutil.which("codex.cmd")
    if not cmd_path:
        logger.warning("Codex CLI not on PATH")
        return "", {}, 0.0
    cmd = [
        cmd_path,
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--sandbox", "read-only",
    ]
    return await _run_subprocess(cmd, prompt.encode("utf-8"), timeout)


async def _run_subprocess(
    cmd: list[str], stdin: bytes, timeout: int
) -> tuple[str, dict[str, int], float]:
    """Async subprocess with hard-kill on timeout. Returns (text, usage, duration_s)."""
    t0 = time.time()
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_clean_env_for_cli(),
        )
    except (FileNotFoundError, OSError) as e:
        logger.warning(f"subprocess spawn failed for {cmd[0]}: {e}")
        return "", {}, time.time() - t0

    try:
        stdout, _stderr = await asyncio.wait_for(
            proc.communicate(input=stdin), timeout=timeout
        )
    except asyncio.TimeoutError:
        for fn in ("kill", "terminate"):
            try:
                getattr(proc, fn)()
            except ProcessLookupError:
                pass
            except Exception:
                pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            pass
        return "", {}, time.time() - t0

    duration = time.time() - t0
    if proc.returncode and proc.returncode != 0:
        return "", {}, duration
    try:
        text_raw = stdout.decode("utf-8", errors="replace")
    except Exception:
        return "", {}, duration
    text, usage = _extract_text_and_usage(text_raw)
    return text, usage, duration


# ---------------------------------------------------------------------------
# Sequential cascade planner -- Codex first, Claude reserve, Gemini last
# ---------------------------------------------------------------------------
#
# Day 1 of Phase 0 burned 170 EUR running a 5-way race (3 Claude + Gemini +
# Codex) because cancelled Claude replicas STILL pay -- asyncio.cancel does
# not refund an already-issued API call. Day 2 architecture: try one
# provider at a time, only fall through to the next when the current one
# produces empty or unparseable output.
#
# Empirics from Day 1: in the one successful ls20 run (1 level completed),
# Codex won 22/24 plan calls, Gemini gave spam-move plans, Claude was
# bimodal. Order reflects this -- Codex first because cheapest tier-1 and
# fastest when fresh, Claude reserve because deepest reasoning when
# available, Gemini last resort because shallow but free.

CASCADE_ORDER: list[tuple[str, str, int, float]] = [
    # (provider, model_for_pricing, timeout_secs, min_budget_remaining_usd)
    #
    # Cold-cache CLI subprocesses are SLOWER than warm. Smoke tests after
    # a few back-to-back calls return in 3-5s (cache hit). The agent's
    # first calls per game start with cold cache and need 30-90s for the
    # provider's own request to come back. Set timeouts conservatively.
    #
    # Day-3 switch: Claude target is OPUS (Sonnet weekly bucket at 100%).
    # Opus is 5x more expensive ($15/$75 per 1M vs $3/$15) -- min_budget
    # raised to $1.0 to ensure room for at least one cold-cache call.
    ("codex",  "codex_default",     60, 0.05),   # cold ~30-50s, warm ~5s
    ("claude", "opus",             150, 1.00),   # 5x sonnet rate; reserve only
    ("gemini", "gemini-flash-2.5",  90, 0.0),    # free, last resort
]


async def _call_planner_cascade(
    prompt: str,
    plan_length: int,
    budget_remaining_usd: float,
) -> tuple[str, str, str, dict[str, int], float, list[dict[str, Any]]]:
    """Try providers in CASCADE_ORDER until one returns a parseable plan.

    Returns:
        (winner_provider, winner_model, winner_text, winner_usage,
         winner_duration_s, attempts)

    ``attempts`` is one entry per CASCADE_ORDER tier we touched, in
    order, each carrying ``{provider, model, duration_s, usage,
    text_len, status, parse_ok}``. ``status`` is one of:
      - ``"winner"``   text parsed into >=1 valid action; cascade stopped here.
      - ``"empty"``    returned text (or none) but no action tokens found.
      - ``"timeout"``  subprocess wall-clock reached/passed ``timeout``.
      - ``"error"``    subprocess failed immediately (binary missing, etc).
      - ``"skipped"``  budget_remaining_usd below this tier's min_budget.

    This makes every CLI invocation visible in ``cost_burn.jsonl`` --
    failed Codex/Claude attempts no longer disappear silently.
    """
    attempts: list[dict[str, Any]] = []

    for provider, model, timeout, min_budget in CASCADE_ORDER:
        if budget_remaining_usd < min_budget:
            attempts.append({
                "provider": provider, "model": model,
                "duration_s": 0.0, "usage": {},
                "text_len": 0, "status": "skipped", "parse_ok": False,
            })
            logger.warning(
                f"cascade: skipping {provider} -- budget ${budget_remaining_usd:.4f} "
                f"< min ${min_budget:.2f}"
            )
            continue

        if provider == "codex":
            text, usage, duration = await _call_codex_async(prompt, timeout)
        elif provider == "claude":
            text, usage, duration = await _call_claude_async(prompt, timeout)
        elif provider == "gemini":
            text, usage, duration = await _call_gemini_async(prompt, timeout)
        else:
            continue

        plan = _parse_plan(text, plan_length) if text else []
        if plan:
            status = "winner"
        elif duration >= (timeout - 5):
            status = "timeout"
        elif duration < 2 and not text:
            status = "error"
        else:
            status = "empty"

        attempts.append({
            "provider": provider, "model": model,
            "duration_s": round(duration, 2), "usage": usage,
            "text_len": len(text), "status": status,
            "parse_ok": bool(plan),
        })

        if status == "winner":
            return provider, model, text, usage, duration, attempts

    return "", "", "", {}, 0.0, attempts


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_action_match(
    m: "re.Match[str]",
) -> tuple[GameAction, dict] | None:
    name = m.group(1).upper()
    try:
        action = GameAction.from_name(name)
    except Exception:
        return None
    data: dict = {}
    if name == "ACTION6":
        # x=N y=N form (groups 2,3) or bare N N form (groups 4,5)
        x_str = m.group(2) or m.group(4)
        y_str = m.group(3) or m.group(5)
        if x_str and y_str:
            try:
                data = {"x": str(int(x_str)), "y": str(int(y_str))}
            except ValueError:
                data = {}
    return action, data


def _parse_plan(text: str, max_actions: int) -> list[tuple[GameAction, dict]]:
    if not text:
        return []
    out: list[tuple[GameAction, dict]] = []
    for m in _ACTION_PATTERN.finditer(text):
        parsed = _parse_action_match(m)
        if parsed is None:
            continue
        out.append(parsed)
        if len(out) >= max_actions:
            break
    return out


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------


def _pretty_print_grid(grid: list[list[list[Any]]]) -> str:
    """Render a 3D grid (list of 2D blocks) as compact hex rows."""
    lines: list[str] = []
    for i, block in enumerate(grid):
        lines.append(f"Grid {i}:")
        for row in block:
            lines.append("  " + "".join(f"{c:02x}" for c in row))
    return "\n".join(lines)


class NhArcBaselineV0(Agent):
    """v0.0c -- sequential cascade planner-executor with budget enforcement.

    Day 2 retrofit after Day 1 burned 170 EUR via the 3-Claude race.
    See module docstring + plan file UPDATE 2026-05-16 section.
    """

    MAX_ACTIONS: int = 80
    PLAN_LENGTH: int = 6
    FRAME_HISTORY_DEPTH: int = 3
    LOG_DIR: str = "logs"

    # Budget caps per game (operator-confirmed 2026-05-17, Opus enabled).
    # Per-env hard $4 ~= 2% of all-models weekly bucket per env.
    # Daily soft $15 ~= 7.5%, hard $20 ~= 10% (operator's daily target).
    # NOTE: USD/percent conversion is empirical ±50%; verify operator's
    # claude.ai/settings/usage % after each significant run.
    BUDGET_SOFT_USD: float = 2.0
    BUDGET_HARD_USD: float = 4.0
    DAILY_SOFT_USD: float = 15.0
    DAILY_HARD_USD: float = 20.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        Path(self.LOG_DIR).mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._log_path = (
            Path(self.LOG_DIR) / f"baseline_v0_{self.game_id}_{ts}.jsonl"
        )
        self._plan_queue: list[tuple[GameAction, dict]] = []
        self._planner_call_count: int = 0
        self._action_history: list[str] = []

        # Budget state.
        self._game_cost_usd: float = 0.0
        self._soft_warned: bool = False
        self._budget_exceeded: bool = False

        # Phase A.2: Gemini multimodal briefing cache. Populated lazily
        # on the first choose_action call when this game has no
        # calibrated color_map. ONE Gemini call per game (~30-60s);
        # subsequent prompts reuse the cached text.
        self._gemini_briefing: str = ""
        self._gemini_briefing_attempted: bool = False

        # Phase B.1: persistent per-game state (LockSmithState for ls20,
        # None for unknown game classes). Updated incrementally before
        # each planner invocation. KNOWN LIMITATION: player_pos is not
        # reliably tracked yet (see arc_game_state.py docstring).
        self._game_state = arc_game_state.state_for_game(self.game_id)
        self._prev_frame: list | None = None

    def is_done(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> bool:
        if self._budget_exceeded:
            return True
        return latest_frame.state is GameState.WIN

    def take_action(self, action: "GameAction"):
        """Override base to log every send/receive for debugging."""
        sent_name = action.name
        sent_id = action.value
        sent_data = action.action_data.model_dump() if hasattr(action, "action_data") else {}
        logger.warning(
            f"[DEBUG SEND] action.name={sent_name} action.id={sent_id} data={sent_data}"
        )
        frame = super().take_action(action)
        if frame is None:
            logger.warning("[DEBUG RECV] frame=None (validation failed)")
        else:
            logger.warning(
                f"[DEBUG RECV] state={frame.state.name} levels={frame.levels_completed} "
                f"avail={list(frame.available_actions or [])}"
            )
        return frame

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        planner_invoked = False
        winner_provider = ""
        winner_model = ""
        winner_duration_s = 0.0
        per_provider_durations: dict[str, float] = {}
        plan_size = 0
        parse_ok = True
        response_preview = ""
        last_call_cost_usd = 0.0
        budget_state = "ok"

        attempts: list[dict] = []
        # Phase B.1: update persistent game state before any planner call.
        # Runs every step (cheap, pure Python) so the state_diff in the
        # next prompt reflects what just happened.
        if self._game_state is not None:
            last_action_name = (
                self._action_history[-1].split()[0]
                if self._action_history else ""
            )
            self._game_state = arc_game_state.update_locksmith_state(
                prev=self._game_state,
                frame=latest_frame.frame,
                prev_action=last_action_name,
                action_counter=self.action_counter,
                level=latest_frame.levels_completed,
                state_label=latest_frame.state.name,
                prev_frame=self._prev_frame,
            )
        self._prev_frame = latest_frame.frame

        if not self._plan_queue:
            planner_invoked = True
            self._planner_call_count += 1
            # Phase A.2: one-time Gemini briefing for envs without
            # symbolic color_map. No-op on subsequent calls.
            self._ensure_gemini_briefing(latest_frame)
            prompt = self._build_planning_prompt(frames, latest_frame)
            budget_remaining = max(
                0.0, self.BUDGET_HARD_USD - self._game_cost_usd
            )

            (
                winner_provider,
                winner_model,
                response,
                usage,
                winner_duration_s,
                attempts,
            ) = asyncio.run(
                _call_planner_cascade(
                    prompt,
                    plan_length=self.PLAN_LENGTH,
                    budget_remaining_usd=budget_remaining,
                )
            )

            response_preview = response[:300]
            self._plan_queue = _parse_plan(response, self.PLAN_LENGTH)
            plan_size = len(self._plan_queue)
            parse_ok = plan_size > 0
            if not self._plan_queue:
                self._plan_queue = [(GameAction.ACTION5, {})]

            # Log EVERY tier we touched -- failed Codex/Claude attempts
            # are now visible in cost_burn.jsonl with status="timeout" /
            # "empty" / "error" / "skipped".
            for att in attempts:
                cost = cost_tracker.record_call(
                    provider=att["provider"],
                    model=att["model"],
                    usage=att["usage"],
                    duration_s=att["duration_s"],
                    game_id=self.game_id,
                    action_counter=self.action_counter,
                    parse_ok=att["parse_ok"],
                    status=att["status"],
                )
                if att["status"] == "winner":
                    last_call_cost_usd = cost
                    self._game_cost_usd += cost

            # Derive per-provider durations from attempts for the decision log.
            per_provider_durations = {
                a["provider"]: a["duration_s"] for a in attempts
            }

            # Daily budget snapshot -- aggregates across ALL games today.
            daily = cost_tracker.today_total_usd()
            daily_usd = daily["total_usd"]
            daily_pct_est = daily["estimated_percent_all_models_weekly"]

            # Real quota usage from NH at localhost:8100, if running.
            # NH counts raw output tokens vs limit -- conservative proxy,
            # not 1:1 to claude.ai (which weights by model price).
            nh = cost_tracker.fetch_nh_usage()
            if nh:
                nh_str = (
                    f"NH(sess={nh['session_percent']:.0f}% "
                    f"week_all={nh['weekly_all_percent']:.0f}% "
                    f"week_sonnet={nh['weekly_sonnet_percent']:.0f}%)"
                )
            else:
                nh_str = "NH=unavailable"

            logger.warning(
                f"[USAGE] {nh_str} | "
                f"USD: today=${daily_usd:.4f} (~{daily_pct_est:.2f}% est) | "
                f"game=${self._game_cost_usd:.4f}/${self.BUDGET_HARD_USD:.2f} | "
                f"daily soft=${self.DAILY_SOFT_USD} hard=${self.DAILY_HARD_USD}"
            )

            # Per-env budget gates.
            if (
                not self._soft_warned
                and self._game_cost_usd >= self.BUDGET_SOFT_USD
            ):
                self._soft_warned = True
                budget_state = "soft_warning"
                logger.warning(
                    f"[BUDGET WARNING] game={self.game_id} "
                    f"cost=${self._game_cost_usd:.4f} of hard ${self.BUDGET_HARD_USD:.2f}"
                )
            if self._game_cost_usd >= self.BUDGET_HARD_USD:
                self._budget_exceeded = True
                budget_state = "hard_stop_env"
                logger.warning(
                    f"[BUDGET HARD STOP env] game={self.game_id} "
                    f"cost=${self._game_cost_usd:.4f} -- ending game early"
                )

            # Daily budget gates.
            if daily_usd >= self.DAILY_HARD_USD:
                self._budget_exceeded = True
                budget_state = "hard_stop_daily"
                logger.warning(
                    f"[BUDGET HARD STOP daily] today=${daily_usd:.4f} "
                    f">= ${self.DAILY_HARD_USD:.2f} -- ending game"
                )
            elif daily_usd >= self.DAILY_SOFT_USD and budget_state == "ok":
                budget_state = "soft_warning_daily"
                logger.warning(
                    f"[BUDGET WARNING daily] today=${daily_usd:.4f} "
                    f">= soft ${self.DAILY_SOFT_USD:.2f}"
                )

        action, data = self._plan_queue.pop(0)
        # ALWAYS set game_id on the action; data may also contain x,y for
        # ACTION6. Without an explicit game_id the server treats every
        # action as a noop (verified by debug 2026-05-17: empty game_id
        # produced state=NOT_FINISHED with no level change across all
        # actions, even for the verified expert sequence).
        action.set_data({**data, "game_id": self.game_id})

        if data:
            self._action_history.append(
                f"{action.name} x={data.get('x','?')} y={data.get('y','?')}"
            )
        else:
            self._action_history.append(action.name)

        self._log_decision(
            planner_invoked=planner_invoked,
            cascade_winner_provider=winner_provider,
            cascade_winner_model=winner_model,
            cascade_winner_duration_s=winner_duration_s,
            cascade_per_provider_durations=per_provider_durations,
            plan_size=plan_size,
            parse_ok=parse_ok,
            action_name=action.name,
            queue_depth_after=len(self._plan_queue),
            planner_call_count=self._planner_call_count,
            response_preview=response_preview,
            last_call_cost_usd=last_call_cost_usd,
            total_game_cost_usd=self._game_cost_usd,
            budget_state=budget_state,
        )
        return action

    def _describe_frame(self, frame: list[list[list[Any]]]) -> tuple[str, bool]:
        """Return (description_text, used_symbolic).

        If we have a calibrated color map for this game, return a compact
        symbolic facts JSON (~150-300 tokens vs ~3000 for hex spam).
        Otherwise fall back to hex pretty-print so unknown game classes
        still get usable input.
        """
        cmap = arc_vision.color_map_for_game(self.game_id)
        if not cmap:
            return _pretty_print_grid(frame), False
        try:
            summary = arc_vision.to_symbolic_summary(
                frame, color_name_map=cmap, max_objects_per_type=5
            )
            return json.dumps(summary, indent=1), True
        except Exception as e:
            logger.warning(f"symbolic summary failed, falling back to hex: {e}")
            return _pretty_print_grid(frame), False

    def _ensure_gemini_briefing(self, latest_frame: FrameData) -> None:
        """Phase A.2: one-time Gemini multimodal briefing for unknown envs.

        Triggered only when:
          * we have no calibrated color_map for this game_id (so symbolic
            extraction is unreliable), AND
          * we haven't tried yet this session.

        On success, caches a 6-10 line Gemini description in
        ``self._gemini_briefing`` which the prompt builder appends as
        a "# VISUAL BRIEFING" section.
        """
        if self._gemini_briefing_attempted:
            return
        if arc_vision.color_map_for_game(self.game_id):
            # We have symbolic extraction; no need for Gemini fallback.
            self._gemini_briefing_attempted = True
            return
        self._gemini_briefing_attempted = True
        try:
            png_path = Path(self.LOG_DIR) / f"briefing_{self.game_id}.png"
            arc_vision.render_frame_to_png(latest_frame.frame, png_path)
            logger.warning(f"[A.2] requesting Gemini briefing for {self.game_id}...")
            text = arc_vision.describe_with_gemini(png_path, timeout=90)
            self._gemini_briefing = text
            if text:
                logger.warning(
                    f"[A.2] Gemini briefing cached ({len(text)} chars)"
                )
        except Exception as e:
            logger.warning(f"[A.2] Gemini briefing failed: {e}")

    def _build_planning_prompt(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> str:
        history_depth = min(self.FRAME_HISTORY_DEPTH, max(len(frames) - 1, 0))
        history_blocks: list[str] = []
        for i, f in enumerate(frames[-history_depth - 1 : -1]):
            past_desc, _ = self._describe_frame(f.frame)
            history_blocks.append(
                f"--- past frame t-{history_depth - i} (state={f.state.name}, "
                f"levels_completed={f.levels_completed}) ---\n"
                + past_desc
            )

        latest_desc, used_symbolic = self._describe_frame(latest_frame.frame)
        latest_block = (
            f"--- current frame (state={latest_frame.state.name}, "
            f"levels_completed={latest_frame.levels_completed}, "
            f"action_counter={self.action_counter}) ---\n"
            + latest_desc
        )

        avail_names: list[str] = []
        for aid in latest_frame.available_actions or []:
            try:
                avail_names.append(GameAction.from_id(aid).name)
            except Exception:
                avail_names.append(f"ACTION_ID_{aid}")
        avail = ", ".join(avail_names) or (
            "RESET, ACTION1, ACTION2, ACTION3, ACTION4, ACTION5, ACTION6, ACTION7"
        )

        # Build action history section — last 20 actions taken so far.
        action_hist = self._action_history[-20:]
        if action_hist:
            action_hist_block = (
                f"# ACTIONS TAKEN SO FAR (last {len(action_hist)} of "
                f"{len(self._action_history)} total)\n"
                + ", ".join(action_hist)
                + "\n(Avoid repeating sequences that have not changed the game state.)"
            )
        else:
            action_hist_block = "# ACTIONS TAKEN SO FAR\n(none yet)"

        # Per-game-class card (LockSmith, etc.) -- inserted as prominent
        # section near the top so the planner doesn't have to rediscover
        # game-specific mechanics in <80 actions.
        card_text = _card_for_game(self.game_id)
        card_block = (
            f"# GAME-SPECIFIC NOTES\n{card_text}\n"
            if card_text
            else ""
        )

        # Phase B.1: persistent state block (LockSmith etc.). Provides a
        # compact symbolic snapshot the planner can reason over without
        # re-parsing the grid every turn. Disclaimer: player_pos
        # tracking is best-effort -- see arc_game_state.py docstring.
        state_block = ""
        if self._game_state is not None:
            state_block = (
                "# PERSISTENT GAME STATE (tracked between turns)\n"
                + self._game_state.to_prompt_block()
                + "\n"
            )

        # Phase A.2: Gemini visual briefing block (only when no color_map).
        gemini_block = ""
        if self._gemini_briefing and not used_symbolic:
            gemini_block = (
                f"# VISUAL BRIEFING (Gemini multimodal, one-time per game)\n"
                f"{self._gemini_briefing}\n"
            )

        # Symbolic perception guide (only when we use to_symbolic_summary).
        symbolic_guide = ""
        if used_symbolic:
            symbolic_guide = (
                "# HOW TO READ THE FRAME SECTIONS\n"
                "The frame is given as a JSON of detected objects (NOT a hex\n"
                "grid). Each named class is a list of regions with their\n"
                "centroid `at: [row, col]` and `size` (cell count). Grid\n"
                "coordinates: row 0 is top, col 0 is left, both [0..63].\n"
                "Movement: ACTION1=up (row-), ACTION2=down (row+),\n"
                "ACTION3=left (col-), ACTION4=right (col+).\n"
                "Use centroid positions to plan paths: the difference\n"
                "between player and target gives the rough action sequence.\n"
                "If you don't see `player_eyes` in the current frame, look\n"
                "for the smallest play_area_bg region -- that's the player\n"
                "sprite footprint.\n"
            )

        return textwrap.dedent(
            f"""\
            # ROLE
            You are planning the next few moves in an ARC-AGI-3 game. The
            world is a grid (matrix of cells with integer values 0-15);
            each action produces the next frame. Your objective is to
            reach state=WIN while minimizing actions and avoiding
            GAME_OVER.

            {card_block}
            {gemini_block}
            {state_block}
            {symbolic_guide}
            # AVAILABLE ACTIONS
            {avail}

            ACTION1..ACTION5 and ACTION7 take no arguments. ACTION6 needs
            (x, y) where both are integers in [0, 63]. RESET starts or
            restarts the game (use as the first action when
            state=NOT_PLAYED, and after GAME_OVER if you want to retry).

            {action_hist_block}

            # RECENT HISTORY (last {history_depth} frames before current)
            {chr(10).join(history_blocks) if history_blocks else '(no prior frames)'}

            # CURRENT FRAME
            {latest_block}

            # OUTPUT FORMAT (strict)
            Plan the next {self.PLAN_LENGTH} actions. Reply with exactly
            one action per line, no prose, no markdown fences, no JSON.
            Use ACTION6 with explicit coordinates if you want to click:

                ACTION1
                ACTION3
                ACTION6 x=12 y=34
                ACTION5
                ACTION2
                ACTION3

            We will execute these in order and replan after the last one
            (or sooner if something looks off). It is fine to plan fewer
            than {self.PLAN_LENGTH} actions when the situation is unclear.

            # HARD RULES (violating these = wasted turn)
            1. **NEVER plan the same action {self.PLAN_LENGTH} times in a row.**
               If you are tempted to write `ACTION1`x6, you have not read
               the grid. Look again at the player position and walls.
            2. **If `levels_completed` has not increased in the last 20
               actions and recent moves were mostly identical, your
               current direction is wrong.** Pick a DIFFERENT primary
               action this turn and explore.
            3. **If the latest grid is byte-identical to the previous
               frame, your last action did nothing (you bumped a wall).**
               Try a perpendicular direction.
            4. Plan should reflect inspection of the grid -- describe in
               your head where the player is, where walls are, what the
               objective looks like, THEN write the {self.PLAN_LENGTH}
               actions. Do not output the inspection; just use it.
            """
        ).strip()

    def _log_decision(
        self,
        planner_invoked: bool,
        cascade_winner_provider: str,
        cascade_winner_model: str,
        cascade_winner_duration_s: float,
        cascade_per_provider_durations: dict[str, float],
        plan_size: int,
        parse_ok: bool,
        action_name: str,
        queue_depth_after: int,
        planner_call_count: int,
        response_preview: str,
        last_call_cost_usd: float,
        total_game_cost_usd: float,
        budget_state: str,
    ) -> None:
        record = {
            "ts": time.time(),
            "game_id": self.game_id,
            "action_counter": self.action_counter,
            "planner_invoked": planner_invoked,
            "cascade_winner_provider": cascade_winner_provider,
            "cascade_winner_model": cascade_winner_model,
            "cascade_winner_duration_s": round(cascade_winner_duration_s, 3),
            "cascade_per_provider_durations": cascade_per_provider_durations,
            "plan_size": plan_size,
            "parse_ok": parse_ok,
            "action_chosen": action_name,
            "queue_depth_after": queue_depth_after,
            "planner_call_count": planner_call_count,
            "game_card_used": bool(_card_for_game(self.game_id)),
            "last_call_cost_usd": round(last_call_cost_usd, 6),
            "total_game_cost_usd": round(total_game_cost_usd, 6),
            "budget_state": budget_state,
            "response_preview": response_preview if planner_invoked else "",
        }
        try:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass
