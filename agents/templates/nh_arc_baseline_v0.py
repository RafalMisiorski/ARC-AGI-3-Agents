"""nh_arc_baseline_v0 -- Claude CLI baseline agent for ARC-AGI-3.

Phase 0 v0.0 of the operator's plan. One Claude CLI call per action.
This is intentionally slow (~10-15s per action -> ~15min per env) -- the
goal of v0.0 is to *confirm the Claude-CLI integration path works
end-to-end*, NOT to be efficient. Once v0.0 prints a non-zero levels_completed
on at least one env, v0.1 will introduce the planner-executor split that
the plan calls out as non-optional for Phase 1.

We extend ``agents.agent.Agent`` directly (NOT ``LLM`` from
``llm_agents.py``), because that template is hard-wired to the OpenAI SDK
(``openai.OpenAIClient``, OpenAI function-calling format, ``o3``/``o4-mini``
models). Vendor-swapping it for Anthropic is a deeper rewrite than just
implementing the ABC contract fresh.

Architecture
------------
- ``choose_action(frames, latest_frame) -> GameAction``
    1. Build a textual prompt from the latest frame (pretty-printed grid)
       plus the last N=3 frames as short context.
    2. Call Claude CLI via ``_call_claude_sync`` (subprocess.run, stream-json
       I/O, env-scrubbed) -- this mirrors the canonical NH pattern in
       ``scripts/llm_call.py::call_claude`` without importing from NH.
    3. Parse the response with a tolerant regex looking for
       ``RESET`` | ``ACTION1`` .. ``ACTION7`` [+ optional ``x=NN y=NN`` for
       ACTION6]. Fall back to ``ACTION5`` on parse failure.
    4. Log latency, response length, and chosen action per call.

Day-1 instrumentation
---------------------
Every ``choose_action`` writes one line to ``logs/baseline_v0_<game>_<ts>.jsonl``
with: timestamp, action_counter, latency_ms, response_chars, action_chosen,
parse_ok. This is the raw data behind the Phase 0 decision gate
("dev levels_completed >= random + 0.3") and the latency tracking the plan
flags as the single most important risk.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any

from arcengine import FrameData, GameAction, GameState

from ..agent import Agent

logger = logging.getLogger(__name__)

_CLAUDE_ENV_SCRUB_KEYS = (
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
    "ANTHROPIC_API_KEY",
)

_ACTION_PATTERN = re.compile(
    r"\b(RESET|ACTION[1-7])(?:\s*[\(\s]\s*x\s*=\s*(\d+)\s*[,;\s]\s*y\s*=\s*(\d+))?",
    re.IGNORECASE,
)


def _clean_env_for_claude() -> dict[str, str]:
    env = os.environ.copy()
    for key in _CLAUDE_ENV_SCRUB_KEYS:
        env.pop(key, None)
    return env


def _extract_text_from_stream(stdout: str) -> str:
    """Parse Claude CLI stream-json output, prefer the final result event."""
    text_parts: list[str] = []
    for line in stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "result" and event.get("subtype") == "success":
            result = event.get("result", "")
            if isinstance(result, str) and result:
                return result.strip()
        if event.get("type") == "assistant":
            msg = event.get("message", {})
            content = msg.get("content", "") if isinstance(msg, dict) else ""
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
            elif isinstance(content, str):
                text_parts.append(content)
    return "\n".join(p for p in text_parts if p).strip()


def _call_claude_sync(
    prompt: str,
    model: str = "sonnet",
    timeout: int = 60,
) -> tuple[str, float]:
    """Synchronous Claude CLI call. Returns (response_text, latency_seconds)."""
    claude_cmd = shutil.which("claude") or shutil.which("claude.cmd")
    if not claude_cmd:
        logger.warning("Claude CLI not on PATH; returning empty")
        return "", 0.0

    cmd = [
        claude_cmd,
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

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            input=stdin_msg,
            capture_output=True,
            text=True,
            env=_clean_env_for_claude(),
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        latency = time.time() - t0
        return _extract_text_from_stream(proc.stdout), latency
    except subprocess.TimeoutExpired:
        latency = time.time() - t0
        logger.warning(f"Claude CLI timeout after {timeout}s")
        return "", latency
    except Exception as e:
        latency = time.time() - t0
        logger.error(f"Claude CLI error: {e}")
        return "", latency


def _pretty_print_grid(grid: list[list[list[Any]]]) -> str:
    """Render a 3D grid (list of 2D blocks) as compact text."""
    lines: list[str] = []
    for i, block in enumerate(grid):
        lines.append(f"Grid {i}:")
        for row in block:
            lines.append("  " + "".join(f"{c:02x}" for c in row))
    return "\n".join(lines)


def _parse_action(text: str) -> tuple[GameAction, dict, bool]:
    """Parse a Claude response into (GameAction, data, parse_ok).

    Tolerant: looks for the first action token anywhere in the response.
    Falls back to ACTION5 on failure (a generally-safe neutral action).
    """
    m = _ACTION_PATTERN.search(text or "")
    if not m:
        return GameAction.ACTION5, {}, False
    name = m.group(1).upper()
    try:
        action = GameAction.from_name(name)
    except Exception:
        return GameAction.ACTION5, {}, False
    data: dict = {}
    if name == "ACTION6" and m.group(2) and m.group(3):
        try:
            data = {"x": str(int(m.group(2))), "y": str(int(m.group(3)))}
        except ValueError:
            data = {}
    return action, data, True


class NhArcBaselineV0(Agent):
    """Phase 0 v0.0 baseline. One Claude CLI call per action.

    Naming maps to ``--agent=nharcbaselinev0`` on the main.py CLI (auto-
    derived lowercase classname via ``Agent.__subclasses__()`` in
    ``agents/__init__.py``).
    """

    MAX_ACTIONS: int = 80
    MODEL: str = "sonnet"
    CALL_TIMEOUT_SECS: int = 60
    FRAME_HISTORY_DEPTH: int = 3
    LOG_DIR: str = "logs"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        Path(self.LOG_DIR).mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._log_path = (
            Path(self.LOG_DIR) / f"baseline_v0_{self.game_id}_{ts}.jsonl"
        )

    def is_done(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        prompt = self._build_prompt(frames, latest_frame)
        response, latency = _call_claude_sync(
            prompt, model=self.MODEL, timeout=self.CALL_TIMEOUT_SECS
        )
        action, data, parse_ok = _parse_action(response)
        if data:
            action.set_data({**data, "game_id": self.game_id})
        self._log_decision(
            latency=latency,
            response_chars=len(response),
            action_name=action.name,
            parse_ok=parse_ok,
            response_preview=response[:200],
        )
        return action

    def _build_prompt(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> str:
        history_depth = min(self.FRAME_HISTORY_DEPTH, max(len(frames) - 1, 0))
        history_blocks: list[str] = []
        for i, f in enumerate(frames[-history_depth - 1 : -1]):
            history_blocks.append(
                f"--- past frame t-{history_depth - i} (state={f.state.name}, "
                f"levels_completed={f.levels_completed}) ---\n"
                + _pretty_print_grid(f.frame)
            )

        latest_block = (
            f"--- current frame (state={latest_frame.state.name}, "
            f"levels_completed={latest_frame.levels_completed}, "
            f"action_counter={self.action_counter}) ---\n"
            + _pretty_print_grid(latest_frame.frame)
        )

        avail = ", ".join(
            a.name for a in (latest_frame.available_actions or [])
        ) or "RESET, ACTION1, ACTION2, ACTION3, ACTION4, ACTION5, ACTION6, ACTION7"

        return textwrap.dedent(
            f"""\
            # ROLE
            You play a single ARC-AGI-3 game. You see a grid (matrix of
            cells with integer values 0-15) and decide one action per turn.
            Your objective is to reach state=WIN while minimizing actions
            and avoiding GAME_OVER.

            # AVAILABLE ACTIONS
            {avail}

            ACTION1..ACTION5 take no arguments. ACTION6 needs (x, y) where
            both are integers in [0, 63]. RESET starts/restarts the game
            (must be your first action if state=NOT_PLAYED, and after
            GAME_OVER if you want to retry).

            # RECENT HISTORY (last {history_depth} frames before current)
            {chr(10).join(history_blocks) if history_blocks else '(no prior frames)'}

            # CURRENT FRAME
            {latest_block}

            # OUTPUT FORMAT (strict)
            Reply with EXACTLY one line of the form:

                ACTION_NAME
            or  ACTION6 x=NN y=NN

            No extra prose, no markdown fences, no JSON. Just the action.
            """
        ).strip()

    def _log_decision(
        self,
        latency: float,
        response_chars: int,
        action_name: str,
        parse_ok: bool,
        response_preview: str,
    ) -> None:
        record = {
            "ts": time.time(),
            "game_id": self.game_id,
            "action_counter": self.action_counter,
            "latency_seconds": round(latency, 3),
            "response_chars": response_chars,
            "action_chosen": action_name,
            "parse_ok": parse_ok,
            "response_preview": response_preview,
        }
        try:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass
