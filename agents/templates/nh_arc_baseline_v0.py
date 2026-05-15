"""nh_arc_baseline_v0 -- Claude CLI baseline agent for ARC-AGI-3.

Phase 0 v0.0 of the operator's plan.

ORIGINAL design: one Claude CLI call per action. ABANDONED on Phase 0 day 1
because measured cold-start latency is ~3-4 minutes per Claude CLI
subprocess invocation (NOT the 5-15s the plan assumed). At 80 actions
that's 4-5h per env, which makes any iteration impossible.

CURRENT design: **planner-executor split**, brought forward from v0.1.
One Claude CLI call produces a plan of ``PLAN_LENGTH`` actions; the
executor pops one per ``choose_action`` invocation; when the queue
empties (or stuck-state is detected), the planner is re-invoked. This
trades plan quality for throughput -- 80/6 = ~14 planner calls per env
at ~3min each = ~40min/env, vs ~4h with the per-action approach.

We extend ``agents.agent.Agent`` directly (NOT ``LLM`` from
``llm_agents.py``), because that template is hard-wired to the OpenAI SDK
(``openai.OpenAIClient``, OpenAI function-calling format, ``o3``/``o4-mini``
models). Vendor-swapping it for Anthropic is a deeper rewrite than just
implementing the ABC contract fresh.

Architecture
------------
- ``choose_action(frames, latest_frame) -> GameAction``
    1. If the plan queue is empty, build a planning prompt (latest frame
       pretty-printed + last N=3 frames as context) asking Claude for a
       list of up to ``PLAN_LENGTH`` actions to execute next.
    2. Call Claude CLI via ``_call_claude_sync`` (subprocess.run,
       stream-json I/O, env-scrubbed) -- mirrors NH's
       ``scripts/llm_call.py::call_claude`` pattern, no NH imports.
    3. Parse the response via ``_parse_plan`` (one action per line, tolerant
       of prose). Fall back to a single ACTION5 on parse failure.
    4. Pop one action from the queue and return it. Subsequent
       ``choose_action`` calls drain the queue without invoking Claude
       until empty.

Day-1 instrumentation
---------------------
Each ``choose_action`` writes one line to
``logs/baseline_v0_<game>_<ts>.jsonl``:

* ``planner_invoked`` -- True iff the queue was empty and we called Claude
  on this step.
* ``planner_latency_seconds`` / ``plan_size`` / ``parse_ok`` -- only when
  planner_invoked.
* ``action_chosen`` / ``queue_depth_after`` -- on every step.

This is the raw data behind the Phase 0 decision gate
("dev levels_completed >= random + 0.3") and the latency / planner-cadence
tracking flagged as the single most important risk in the plan.
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


def _parse_action_match(
    m: "re.Match[str]",
) -> tuple[GameAction, dict] | None:
    name = m.group(1).upper()
    try:
        action = GameAction.from_name(name)
    except Exception:
        return None
    data: dict = {}
    if name == "ACTION6" and m.group(2) and m.group(3):
        try:
            data = {"x": str(int(m.group(2))), "y": str(int(m.group(3)))}
        except ValueError:
            data = {}
    return action, data


def _parse_plan(text: str, max_actions: int) -> list[tuple[GameAction, dict]]:
    """Parse a Claude response into an ordered list of (action, data) pairs.

    Tolerant: scans the entire response for action tokens, takes the first
    ``max_actions`` valid ones, ignores intervening prose. Returns empty list
    on total failure (caller falls back to ACTION5).
    """
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


class NhArcBaselineV0(Agent):
    """Phase 0 v0.0 baseline. One Claude CLI call per action.

    Naming maps to ``--agent=nharcbaselinev0`` on the main.py CLI (auto-
    derived lowercase classname via ``Agent.__subclasses__()`` in
    ``agents/__init__.py``).
    """

    MAX_ACTIONS: int = 80
    MODEL: str = "sonnet"
    CALL_TIMEOUT_SECS: int = 180  # raised from 60 -- cold-start CLI takes ~3min
    FRAME_HISTORY_DEPTH: int = 3
    PLAN_LENGTH: int = 6  # actions per Claude planner call
    LOG_DIR: str = "logs"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        Path(self.LOG_DIR).mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._log_path = (
            Path(self.LOG_DIR) / f"baseline_v0_{self.game_id}_{ts}.jsonl"
        )
        self._plan_queue: list[tuple[GameAction, dict]] = []
        self._planner_call_count: int = 0

    def is_done(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        planner_invoked = False
        planner_latency = 0.0
        plan_size = 0
        parse_ok = True
        response_preview = ""

        if not self._plan_queue:
            planner_invoked = True
            self._planner_call_count += 1
            prompt = self._build_planning_prompt(frames, latest_frame)
            response, planner_latency = _call_claude_sync(
                prompt, model=self.MODEL, timeout=self.CALL_TIMEOUT_SECS
            )
            response_preview = response[:300]
            self._plan_queue = _parse_plan(response, self.PLAN_LENGTH)
            plan_size = len(self._plan_queue)
            parse_ok = plan_size > 0
            if not self._plan_queue:
                # Total parse failure -- emit one safe ACTION5 and re-plan
                # on the next step.
                self._plan_queue = [(GameAction.ACTION5, {})]

        action, data = self._plan_queue.pop(0)
        if data:
            action.set_data({**data, "game_id": self.game_id})

        self._log_decision(
            planner_invoked=planner_invoked,
            planner_latency_seconds=planner_latency,
            plan_size=plan_size,
            parse_ok=parse_ok,
            action_name=action.name,
            queue_depth_after=len(self._plan_queue),
            planner_call_count=self._planner_call_count,
            response_preview=response_preview,
        )
        return action

    def _build_planning_prompt(
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

        # available_actions is list[int] of action ids -- resolve via GameAction.from_id.
        avail_names: list[str] = []
        for aid in latest_frame.available_actions or []:
            try:
                avail_names.append(GameAction.from_id(aid).name)
            except Exception:
                avail_names.append(f"ACTION_ID_{aid}")
        avail = ", ".join(avail_names) or (
            "RESET, ACTION1, ACTION2, ACTION3, ACTION4, ACTION5, ACTION6, ACTION7"
        )

        return textwrap.dedent(
            f"""\
            # ROLE
            You are planning the next few moves in an ARC-AGI-3 game. The
            world is a grid (matrix of cells with integer values 0-15);
            each action produces the next frame. Your objective is to
            reach state=WIN while minimizing actions and avoiding
            GAME_OVER.

            # AVAILABLE ACTIONS
            {avail}

            ACTION1..ACTION5 and ACTION7 take no arguments. ACTION6 needs
            (x, y) where both are integers in [0, 63]. RESET starts or
            restarts the game (use as the first action when
            state=NOT_PLAYED, and after GAME_OVER if you want to retry).

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
            """
        ).strip()

    def _log_decision(
        self,
        planner_invoked: bool,
        planner_latency_seconds: float,
        plan_size: int,
        parse_ok: bool,
        action_name: str,
        queue_depth_after: int,
        planner_call_count: int,
        response_preview: str,
    ) -> None:
        record = {
            "ts": time.time(),
            "game_id": self.game_id,
            "action_counter": self.action_counter,
            "planner_invoked": planner_invoked,
            "planner_latency_seconds": round(planner_latency_seconds, 3),
            "plan_size": plan_size,
            "parse_ok": parse_ok,
            "action_chosen": action_name,
            "queue_depth_after": queue_depth_after,
            "planner_call_count": planner_call_count,
            "response_preview": response_preview if planner_invoked else "",
        }
        try:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass
