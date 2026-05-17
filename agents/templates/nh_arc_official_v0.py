"""nh_arc_official_v0 -- minimal "official-style" agent for ARC-AGI-3.

Background (from ARC Prize Technical Report, page 14-15):

* "The official leaderboard will not use a harness to report official
  scores."
* Official system prompt (page 15):
    "You are playing a game. Your goal is to win. Reply with the exact
     action you want to take. The final action in your reply will be
     executed next turn. Your entire reply will be carried to the next
     turn."
* Scoring is RHAE (Relative Human Action Efficiency), not raw level
  completion: per-level score = (human_baseline / agent_actions)^2,
  capped at 0% if agent takes >5x human baseline. Frontier models
  currently score 0.25-0.37%.

This agent is the COUNTER-EXPERIMENT to nh_arc_baseline_v0 (cascade
harness): single Claude Opus call per action, minimal prompt, no game
cards, no action history, no provider race. Goal: see what we get
under official leaderboard rules.

If this scores meaningfully on >=1 level, the harness in v0 was hurting
us. If it scores 0, the bug is environmental (operator's manual play
works -> API protocol differs from UI play).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any

from arcengine import FrameData, GameAction, GameState

from ..agent import Agent
from . import cost_tracker

logger = logging.getLogger(__name__)

# Official system prompt verbatim from ARC Prize Technical Report page 15.
_OFFICIAL_SYSTEM_PROMPT = (
    "You are playing a game. Your goal is to win. Reply with the exact "
    "action you want to take. The final action in your reply will be "
    "executed next turn. Your entire reply will be carried to the next turn."
)

_ACTION_PATTERN = re.compile(
    r"\b(RESET|ACTION[1-7])"
    r"(?:"
    r"\s*[\(\s]\s*x\s*=\s*(\d+)\s*[,;\s]\s*y\s*=\s*(\d+)"
    r"|"
    r"\s+(\d+)\s+(\d+)"
    r")?",
    re.IGNORECASE,
)

_CLI_ENV_SCRUB_KEYS = (
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
    "ANTHROPIC_API_KEY",
)


def _clean_env() -> dict[str, str]:
    import os as _os
    env = _os.environ.copy()
    for k in _CLI_ENV_SCRUB_KEYS:
        env.pop(k, None)
    return env


def _grid_to_text(grid: list[list[list[Any]]]) -> str:
    lines: list[str] = []
    for i, block in enumerate(grid):
        lines.append(f"Grid {i}:")
        for row in block:
            lines.append("  " + "".join(f"{c:02x}" for c in row))
    return "\n".join(lines)


async def _call_opus(prompt: str, timeout: int = 180) -> tuple[str, dict[str, int], float]:
    """Single Opus call, returns (text, usage, duration_s)."""
    cmd_path = shutil.which("claude") or shutil.which("claude.cmd")
    if not cmd_path:
        return "", {}, 0.0
    cmd = [
        cmd_path,
        "--model", "opus",
        "--output-format", "stream-json",
        "--input-format", "stream-json",
        "--verbose",
        "--dangerously-skip-permissions",
        "--append-system-prompt", _OFFICIAL_SYSTEM_PROMPT,
    ]
    stdin_msg = (
        json.dumps({"type": "user", "message": {"role": "user", "content": prompt}})
        + "\n"
    )
    t0 = time.time()
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_clean_env(),
        )
        stdout, _ = await asyncio.wait_for(
            proc.communicate(input=stdin_msg.encode("utf-8")), timeout=timeout
        )
    except (asyncio.TimeoutError, Exception) as e:
        logger.warning(f"Opus call failed: {e}")
        return "", {}, time.time() - t0

    text_raw = stdout.decode("utf-8", errors="replace")
    duration = time.time() - t0

    # Parse stream-json: prefer result.usage for usage, result.result for text.
    text_parts: list[str] = []
    usage: dict[str, int] = {}
    final_text = ""
    for line in text_raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "result" and event.get("subtype") == "success":
            r = event.get("result")
            if isinstance(r, str) and r:
                final_text = r.strip()
            u = event.get("usage") or {}
            if isinstance(u, dict):
                usage = {
                    "input_tokens": int(u.get("input_tokens") or 0),
                    "output_tokens": int(u.get("output_tokens") or 0),
                    "cache_read_tokens": int(u.get("cache_read_input_tokens") or 0),
                    "cache_creation_tokens": int(u.get("cache_creation_input_tokens") or 0),
                }
        if event.get("type") == "assistant":
            msg = event.get("message", {})
            if isinstance(msg, dict):
                c = msg.get("content", "")
                if isinstance(c, list):
                    for blk in c:
                        if isinstance(blk, dict) and blk.get("type") == "text":
                            t = blk.get("text", "")
                            if t:
                                text_parts.append(t)

    text = final_text if final_text else "".join(text_parts).strip()
    return text, usage, duration


def _parse_last_action(text: str) -> tuple[GameAction, dict]:
    """Per official prompt: 'The final action in your reply will be executed'.

    Scan all action tokens, take the LAST one. Fall back to ACTION5 on parse
    failure.
    """
    last_action = GameAction.ACTION5
    last_data: dict = {}
    last_match_pos = -1
    for m in _ACTION_PATTERN.finditer(text or ""):
        if m.start() <= last_match_pos:
            continue
        name = m.group(1).upper()
        try:
            action = GameAction.from_name(name)
        except Exception:
            continue
        data: dict = {}
        if name == "ACTION6":
            x = m.group(2) or m.group(4)
            y = m.group(3) or m.group(5)
            if x and y:
                try:
                    data = {"x": str(int(x)), "y": str(int(y))}
                except ValueError:
                    pass
        last_action = action
        last_data = data
        last_match_pos = m.start()
    return last_action, last_data


class NhArcOfficialV0(Agent):
    """Single-Opus, no-harness, official-prompt agent."""

    MAX_ACTIONS: int = 30  # cost containment for first test
    FRAME_HISTORY_DEPTH: int = 2  # very short -- official prompt is minimal
    LOG_DIR: str = "logs"

    # Per-env hard cap. Opus is 5x Sonnet; with cache hits ~$0.05/call,
    # 30 actions worst-case cold cache ~$1.5-3.
    BUDGET_HARD_USD: float = 5.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        Path(self.LOG_DIR).mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._log_path = (
            Path(self.LOG_DIR) / f"official_v0_{self.game_id}_{ts}.jsonl"
        )
        self._game_cost_usd: float = 0.0
        self._budget_exceeded: bool = False

    def is_done(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> bool:
        if self._budget_exceeded:
            return True
        return latest_frame.state is GameState.WIN

    def take_action(self, action: GameAction):
        # Force game_id (was the placebo bug from baseline_v0 -- harmless
        # to set even though SDK ignores it).
        action.set_data({"game_id": self.game_id})
        sent = (action.name, action.value)
        logger.warning(f"[OFFICIAL SEND] {sent[0]} (id={sent[1]})")
        frame = super().take_action(action)
        if frame is None:
            logger.warning("[OFFICIAL RECV] frame=None")
        else:
            logger.warning(
                f"[OFFICIAL RECV] state={frame.state.name} "
                f"levels={frame.levels_completed} "
                f"avail={list(frame.available_actions or [])}"
            )
        return frame

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        prompt = self._build_prompt(frames, latest_frame)
        response, usage, duration = asyncio.run(_call_opus(prompt, timeout=180))

        cost = cost_tracker.record_call(
            provider="claude",
            model="opus",
            usage=usage,
            duration_s=duration,
            game_id=self.game_id,
            action_counter=self.action_counter,
            parse_ok=bool(response),
            status="winner" if response else "empty",
            purpose="arc_planner_official",
        )
        self._game_cost_usd += cost
        if self._game_cost_usd >= self.BUDGET_HARD_USD:
            self._budget_exceeded = True
            logger.warning(
                f"[OFFICIAL BUDGET STOP] game={self.game_id} "
                f"cost=${self._game_cost_usd:.4f}"
            )

        action, data = _parse_last_action(response)
        # game_id is set in take_action override; keep coords for ACTION6.
        if data:
            action.set_data(data)

        self._log_step(
            response=response, usage=usage, duration_s=duration,
            cost_usd=cost, action_name=action.name,
        )
        return action

    def _build_prompt(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> str:
        """Minimal prompt -- frame state only, no extra scaffolding.

        Official prompt deliberately doesn't give the model lots of
        guidance. We feed the grid + state + available_actions and let
        Opus produce its own reply. Per official: "Your entire reply will
        be carried to the next turn" -- the model can self-track context.
        """
        avail_names: list[str] = []
        for aid in latest_frame.available_actions or []:
            try:
                avail_names.append(GameAction.from_id(aid).name)
            except Exception:
                pass
        avail = ", ".join(avail_names) or "RESET, ACTION1..7"

        # Just the latest frame; optionally one prior for diff.
        latest_block = _grid_to_text(latest_frame.frame)

        return (
            f"Game state: {latest_frame.state.name}\n"
            f"Levels completed: {latest_frame.levels_completed}/{latest_frame.win_levels}\n"
            f"Available actions: {avail}\n"
            f"Action counter: {self.action_counter}\n\n"
            f"Current grid:\n{latest_block}\n"
        )

    def _log_step(
        self,
        response: str,
        usage: dict[str, int],
        duration_s: float,
        cost_usd: float,
        action_name: str,
    ) -> None:
        record = {
            "ts": time.time(),
            "game_id": self.game_id,
            "action_counter": self.action_counter,
            "duration_s": round(duration_s, 3),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "cost_usd": round(cost_usd, 6),
            "total_game_cost_usd": round(self._game_cost_usd, 6),
            "action_chosen": action_name,
            "response_preview": response[:300],
        }
        try:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass
