"""nh_arc_vision_v0 -- per-action Claude vision multimodal agent.

Day 5 jump after random sweep proved SDK works (states transition to
GAME_OVER) and 0 levels across all agents was due to **hex-grid encoding
limiting LLM spatial reasoning**, not protocol bugs.

Smoke test (2026-05-19):
* Claude CLI vision (sonnet) on a real ls20 frame PNG: **18.4s**, output
  identified player ("black cursor with blue dot"), walls, door/exit,
  chest/key tile. Quality: domain-specific (knows LockSmith mechanics).
* Gemini CLI (flash 2.5): 33.5s, generic ("yellow maze walls").

Claude is 2x faster AND substantially higher quality. Per-action vision
call becomes economically viable: 18s x 14 planner calls = ~4 min vision
work per env. Total ~10 min/env including parsing and execution. Dev
sweep ~50 min + ~$0.50-1 (vision Claude tokens are cheap when image
encoding is ~2000 tokens).

Architecture
------------
* Planner-executor split (PLAN_LENGTH=6 actions per Claude call).
* Each planner call:
    1. Render current frame to PNG via ``arc_vision.render_frame_to_png``.
    2. Call ``claude -p`` with prompt + ``@<absolute_png_path>`` inline.
    3. Parse 6-action plan via ``_parse_plan`` (shared with baseline_v0).
    4. Pop one action per ``choose_action``; replan when queue empty.
* Fallback: Gemini vision (33s, lower quality) when Claude returns empty
  or times out.
* Cost-tracked via ``cost_tracker.record_call``.
* Budget hard stop at $5/env (configurable).

Frame PNGs saved to ``logs/vision_frames/<game_id>/<NNN>.png`` for
debugging / replay analysis.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from arcengine import FrameData, GameAction, GameState

from ..agent import Agent
from . import arc_vision
from . import cost_tracker

logger = logging.getLogger(__name__)

_CLI_ENV_SCRUB_KEYS = (
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
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


def _clean_env_for_cli() -> dict[str, str]:
    env = os.environ.copy()
    for k in _CLI_ENV_SCRUB_KEYS:
        env.pop(k, None)
    return env


def _parse_plan(text: str, max_actions: int) -> list[tuple[GameAction, dict]]:
    """Tolerant multi-action parser (shared with baseline_v0)."""
    if not text:
        return []
    out: list[tuple[GameAction, dict]] = []
    for m in _ACTION_PATTERN.finditer(text):
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
                    data = {}
        out.append((action, data))
        if len(out) >= max_actions:
            break
    return out


def _call_claude_vision(png_path: Path, prompt: str, timeout: int = 60) -> tuple[str, float]:
    """Claude CLI -p with inline @<abs_path> image reference.

    Runs from a temp cwd to bypass CLAUDE.md auto-discovery (which would
    otherwise make Claude think the prompt is a CC interactive request
    and reply "what would you like to work on?" instead of executing the
    task). Returns (text, latency_s).
    """
    cmd_path = shutil.which("claude") or shutil.which("claude.cmd")
    if not cmd_path:
        return "", 0.0
    abs_path = str(png_path.resolve())
    full_prompt = f"{prompt}\n\nFrame image: @{abs_path}"
    import tempfile
    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory(prefix="nh_arc_vision_") as tmp_cwd:
            proc = subprocess.run(
                [cmd_path, "-p", full_prompt, "--model", "sonnet"],
                cwd=tmp_cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
                env=_clean_env_for_cli(),
            )
        return (proc.stdout or "").strip(), time.time() - t0
    except subprocess.TimeoutExpired:
        return "", time.time() - t0
    except Exception as e:
        logger.warning(f"claude vision call failed: {e}")
        return "", time.time() - t0


def _call_gemini_vision(png_path: Path, prompt: str, timeout: int = 90) -> tuple[str, float]:
    """Gemini CLI -p fallback for vision. Slower (~33s) and weaker than Claude.

    Runs from Path.home() instead of a tempdir -- Gemini interprets empty
    dirs as "new project, I'm a code helper" and ignores the actual prompt
    (returns its intro instead of an action plan). Home dir has the user's
    normal context and behaves predictably.
    """
    cmd_path = shutil.which("gemini") or shutil.which("gemini.cmd")
    if not cmd_path:
        return "", 0.0
    abs_path = str(png_path.resolve()).replace("\\", "/")
    full_prompt = f"{prompt}\n\nImage: @{abs_path}"
    t0 = time.time()
    try:
        proc = subprocess.run(
            [cmd_path, "-p", full_prompt, "-m", "gemini-2.5-flash"],
            cwd=str(Path.home()),
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
            env=_clean_env_for_cli(),
        )
        return (proc.stdout or "").strip(), time.time() - t0
    except subprocess.TimeoutExpired:
        return "", time.time() - t0
    except Exception as e:
        logger.warning(f"gemini vision call failed: {e}")
        return "", time.time() - t0


# Per-game-class hint cards (from baseline_v0 GAME_CARDS). When the env_id
# prefix matches, this is injected into the vision planner prompt as a
# "# GAME-SPECIFIC NOTES" section -- gives the LLM concrete mechanics it
# would otherwise have to rediscover within the 80-action budget.
GAME_CARDS: dict[str, str] = {
    "ls20": (
        "You are playing **LockSmith**. Rules and strategy:\n"
        "* ACTION1=move up, ACTION2=move down, ACTION3=move left, "
        "ACTION4=move right. ACTION5/6/7 do nothing in this game.\n"
        "* Goal: find a key that matches the one inside the exit door, "
        "then walk into the door.\n"
        "* 6 levels total; `levels_completed` shows current progress.\n"
        "* Each level starts with limited energy. Moving consumes energy; "
        "GAME_OVER if you run out. Refill at 2x2 squares of energy pills.\n"
        "* Walls block movement. If the grid does not change after a move, "
        "you bumped into a wall -- pick a different direction.\n"
        "* Look for key-shape rotators and color rotators in the corners. "
        "Step on/off them to cycle key shape and color until they match "
        "the target key shown in the exit door.\n"
    ),
}


def _card_for_game(game_id: str) -> str:
    if not game_id:
        return ""
    key = game_id.split("-", 1)[0]
    return GAME_CARDS.get(key, "")


class NhArcVisionV0(Agent):
    """Per-action multimodal Claude vision agent."""

    MAX_ACTIONS: int = 80
    PLAN_LENGTH: int = 6
    VISION_TIMEOUT_SECS: int = 60
    GEMINI_FALLBACK_TIMEOUT_SECS: int = 90
    BUDGET_HARD_USD: float = 5.0
    BUDGET_SOFT_USD: float = 2.0
    LOG_DIR: str = "logs"
    FRAMES_DIR: str = "logs/vision_frames"

    # Conservative pricing assumption (Anthropic API rates).
    EST_COST_PER_CALL_CLAUDE_USD: float = 0.02  # ~2400 tokens in + 100 out at sonnet
    EST_COST_PER_CALL_GEMINI_USD: float = 0.0   # free subscription

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        Path(self.LOG_DIR).mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._log_path = (
            Path(self.LOG_DIR) / f"vision_v0_{self.game_id}_{ts}.jsonl"
        )
        self._frames_dir = Path(self.FRAMES_DIR) / self.game_id
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        self._plan_queue: list[tuple[GameAction, dict]] = []
        self._planner_call_count: int = 0
        self._game_cost_usd: float = 0.0
        self._budget_exceeded: bool = False
        self._budget_soft_warned: bool = False

    def is_done(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> bool:
        if self._budget_exceeded:
            return True
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        planner_invoked = False
        provider = ""
        latency = 0.0
        png_path_str = ""
        plan_size = 0
        parse_ok = True
        response_preview = ""
        call_cost = 0.0

        if not self._plan_queue:
            planner_invoked = True
            self._planner_call_count += 1

            # 1. Render current frame to PNG
            png_path = self._frames_dir / f"{self.action_counter:03d}.png"
            try:
                arc_vision.render_frame_to_png(latest_frame.frame, png_path)
                png_path_str = str(png_path.relative_to(Path.cwd())) if png_path.is_absolute() else str(png_path)
            except Exception as e:
                logger.warning(f"render failed at step {self.action_counter}: {e}")
                self._plan_queue = [(GameAction.ACTION5, {})]
                parse_ok = False

            if not self._plan_queue:
                # 2. Build prompt + call vision planner
                prompt = self._build_planning_prompt(latest_frame)
                response, latency = _call_claude_vision(
                    png_path, prompt, timeout=self.VISION_TIMEOUT_SECS
                )
                provider = "claude_vision"
                call_cost = self.EST_COST_PER_CALL_CLAUDE_USD if response else 0.0

                plan = _parse_plan(response, self.PLAN_LENGTH)
                if not plan:
                    # Fallback Gemini
                    response, latency_g = _call_gemini_vision(
                        png_path, prompt, timeout=self.GEMINI_FALLBACK_TIMEOUT_SECS
                    )
                    provider = "gemini_vision"
                    latency = latency_g
                    call_cost = self.EST_COST_PER_CALL_GEMINI_USD
                    plan = _parse_plan(response, self.PLAN_LENGTH)

                response_preview = response[:300]
                self._plan_queue = plan
                plan_size = len(self._plan_queue)
                parse_ok = plan_size > 0
                if not self._plan_queue:
                    self._plan_queue = [(GameAction.ACTION5, {})]
                    provider = provider or "none"

            # 3. Cost tracking + budget enforcement
            self._game_cost_usd += call_cost
            try:
                cost_tracker.record_call(
                    provider=provider,
                    model="sonnet" if provider == "claude_vision" else "gemini-flash-2.5",
                    usage={
                        "input_tokens": 0,  # CLI doesn't expose, we estimate
                        "output_tokens": 0,
                        "cache_read_tokens": 0,
                    },
                    duration_s=latency,
                    game_id=self.game_id,
                    action_counter=self.action_counter,
                    parse_ok=parse_ok,
                    status="winner" if parse_ok else ("empty" if not response_preview else "error"),
                    caller_module="nh_arc_vision_v0",
                    purpose="arc_vision_planner",
                )
            except Exception as e:
                logger.debug(f"cost_tracker.record_call failed: {e}")

            if not self._budget_soft_warned and self._game_cost_usd >= self.BUDGET_SOFT_USD:
                self._budget_soft_warned = True
                logger.warning(
                    f"[BUDGET WARNING] game {self.game_id} reached "
                    f"${self._game_cost_usd:.2f} (soft cap ${self.BUDGET_SOFT_USD})"
                )
            if self._game_cost_usd >= self.BUDGET_HARD_USD:
                self._budget_exceeded = True
                logger.warning(
                    f"[BUDGET HARD STOP] game {self.game_id} hit "
                    f"${self._game_cost_usd:.2f}, ending run"
                )

        action, data = self._plan_queue.pop(0)
        if data:
            action.set_data({**data, "game_id": self.game_id})

        self._log_decision(
            planner_invoked=planner_invoked,
            provider=provider,
            latency_seconds=latency,
            png_path=png_path_str,
            plan_size=plan_size,
            parse_ok=parse_ok,
            action_name=action.name,
            queue_depth_after=len(self._plan_queue),
            planner_call_count=self._planner_call_count,
            game_cost_usd=self._game_cost_usd,
            response_preview=response_preview,
        )
        return action

    def _build_planning_prompt(self, latest_frame: FrameData) -> str:
        avail_names: list[str] = []
        for aid in latest_frame.available_actions or []:
            try:
                avail_names.append(GameAction.from_id(aid).name)
            except Exception:
                avail_names.append(f"ACTION_ID_{aid}")
        avail = ", ".join(avail_names) or (
            "RESET, ACTION1, ACTION2, ACTION3, ACTION4, ACTION5, ACTION6, ACTION7"
        )

        card = _card_for_game(self.game_id)
        card_block = f"\n# GAME-SPECIFIC NOTES\n{card}\n" if card else ""

        return (
            "# ROLE\n"
            "You are planning moves in an ARC-AGI-3 game. You see one image\n"
            "(the current game frame). Plan the next "
            f"{self.PLAN_LENGTH} actions to progress toward winning a level\n"
            "while avoiding GAME_OVER.\n"
            f"{card_block}"
            "\n"
            "# AVAILABLE ACTIONS\n"
            f"{avail}\n"
            "\n"
            "ACTION1=up, ACTION2=down, ACTION3=left, ACTION4=right.\n"
            "ACTION5=enter/select. ACTION6 needs (x, y) coordinates in [0, 63].\n"
            "ACTION7=optional auxiliary. RESET=restart (use only if NOT_PLAYED\n"
            "or GAME_OVER and you want to retry from scratch).\n"
            "\n"
            "# GAME STATE\n"
            f"State: {latest_frame.state.name}\n"
            f"Levels completed: {latest_frame.levels_completed}\n"
            f"Action counter: {self.action_counter}/{self.MAX_ACTIONS}\n"
            "\n"
            "# OUTPUT FORMAT (strict)\n"
            f"Reply with EXACTLY {self.PLAN_LENGTH} lines, one action per\n"
            "line, no prose, no markdown fences, no JSON.\n"
            "\n"
            "Example:\n"
            "ACTION1\n"
            "ACTION3\n"
            "ACTION6 x=12 y=34\n"
            "ACTION5\n"
            "ACTION2\n"
            "ACTION4\n"
        )

    def _log_decision(
        self,
        planner_invoked: bool,
        provider: str,
        latency_seconds: float,
        png_path: str,
        plan_size: int,
        parse_ok: bool,
        action_name: str,
        queue_depth_after: int,
        planner_call_count: int,
        game_cost_usd: float,
        response_preview: str,
    ) -> None:
        record = {
            "ts": time.time(),
            "game_id": self.game_id,
            "action_counter": self.action_counter,
            "planner_invoked": planner_invoked,
            "provider": provider,
            "latency_seconds": round(latency_seconds, 3),
            "png_path": png_path if planner_invoked else "",
            "plan_size": plan_size,
            "parse_ok": parse_ok,
            "action_chosen": action_name,
            "queue_depth_after": queue_depth_after,
            "planner_call_count": planner_call_count,
            "game_cost_usd": round(game_cost_usd, 4),
            "response_preview": response_preview if planner_invoked else "",
        }
        try:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass
