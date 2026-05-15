"""nh_arc_baseline_v0 -- Multi-provider CLI race baseline for ARC-AGI-3.

Phase 0 v0.0 of the operator's plan.

EVOLUTION LOG
-------------
v0.0a (abandoned same day): 1 Claude CLI subprocess per action.
    Measured cold start ~80s, but second+ planner calls **time out at
    180s** -- Claude Max 20x throttles after a rapid first call. ~25%
    success rate, levels_completed=0 after 23 actions.

v0.0b (current): planner-executor split brought forward from v0.1, and
    the single Claude planner replaced by a **5-way race**:

        3 x Claude CLI  (sonnet)   -- 3 separate processes, separate
                                       Max session slots per NH router
        1 x Gemini CLI  (flash 2.5) -- 1M context, free subscription
        1 x Codex CLI   (default)   -- subscription-backed

    Each planner call spawns all 5 processes in parallel. The first one
    to return a response that parses into >=1 valid action wins; the
    others are cancelled. This routes around per-provider throttling
    (any one of 5 finishing fast is enough) and amortises Windows
    subprocess cold-start across providers.

Architecture
------------
* ``choose_action(frames, latest_frame) -> GameAction``
    1. If the plan queue is empty, build a planner prompt from the latest
       frame plus N=3 recent frames.
    2. ``asyncio.run(_call_planner_race(...))`` launches 5 subprocesses,
       FIRST_COMPLETED-style. Winner returned with (response, provider,
       latency).
    3. Parse the winner via ``_parse_plan`` (tolerant regex, multi-line).
       On total failure, queue gets one ACTION5 and we replan next step.
    4. Pop one (action, data) and return.

We extend ``agents.agent.Agent`` directly, not ``LLM`` from
``llm_agents.py`` (which is hard-wired to the OpenAI SDK).

Instrumentation per step -> logs/baseline_v0_<game>_<ts>.jsonl
-------------------------------------------------------------
* planner_invoked       (bool) -- True iff queue was empty this step
* race_winner_provider  (str)  -- claude_0/1/2 | gemini | codex | ""
* race_winner_latency_s (float)
* race_competitor_latencies (dict provider->float, only those that
                              finished before the winner was picked)
* plan_size, parse_ok, action_chosen, queue_depth_after,
  planner_call_count, response_preview
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
GAME_CARDS: dict[str, str] = {
    "ls20": (
        "You are playing **LockSmith**. Rules and strategy:\n"
        "* ACTION1=move up, ACTION2=move down, ACTION3=move left, "
        "ACTION4=move right. ACTION5/6/7 do nothing in this game.\n"
        "* Goal: find a key that matches the one inside the exit door, "
        "then walk into the door.\n"
        "* 6 levels total; `levels_completed` shows current progress.\n"
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
        "* If the grid does not change after a move, you bumped into a wall."
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


def _extract_text_from_stream(stdout: str) -> str:
    """Parse stream-json / JSONL output from any of the 3 CLI shapes.

    Recognises:
      Claude: {"type":"result","subtype":"success","result":"..."}     (preferred)
              {"type":"assistant","message":{"content":[{"type":"text","text":"..."}]}}
      Gemini: {"type":"message","role":"assistant","content":"..."}
      Codex : {"type":"item.completed","item":{"type":"agent_message","text":"..."}}
    """
    text_parts: list[str] = []
    for line in stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Claude final-result event -- authoritative, return immediately.
        if event.get("type") == "result" and event.get("subtype") == "success":
            result = event.get("result", "")
            if isinstance(result, str) and result:
                return result.strip()

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

        # Gemini streaming assistant chunks.
        if event.get("type") == "message" and event.get("role") == "assistant":
            content = event.get("content", "")
            if isinstance(content, str) and content:
                text_parts.append(content)

        # Codex JSONL agent_message events.
        if event.get("type") == "item.completed":
            item = event.get("item", {})
            if isinstance(item, dict) and item.get("type") == "agent_message":
                t = item.get("text", "")
                if t:
                    text_parts.append(t)

    return "".join(text_parts).strip()


# ---------------------------------------------------------------------------
# Per-provider async wrappers
# ---------------------------------------------------------------------------


async def _call_claude_async(prompt: str, timeout: int) -> str:
    cmd_path = shutil.which("claude") or shutil.which("claude.cmd")
    if not cmd_path:
        logger.warning("Claude CLI not on PATH")
        return ""

    cmd = [
        cmd_path,
        "--model", "sonnet",
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


async def _call_gemini_async(prompt: str, timeout: int) -> str:
    cmd_path = shutil.which("gemini") or shutil.which("gemini.cmd")
    if not cmd_path:
        logger.warning("Gemini CLI not on PATH")
        return ""
    cmd = [cmd_path, "-m", "gemini-2.5-flash", "--output-format", "stream-json"]
    return await _run_subprocess(cmd, prompt.encode("utf-8"), timeout)


async def _call_codex_async(prompt: str, timeout: int) -> str:
    cmd_path = shutil.which("codex") or shutil.which("codex.cmd")
    if not cmd_path:
        logger.warning("Codex CLI not on PATH")
        return ""
    cmd = [
        cmd_path,
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--sandbox", "read-only",
    ]
    return await _run_subprocess(cmd, prompt.encode("utf-8"), timeout)


async def _run_subprocess(cmd: list[str], stdin: bytes, timeout: int) -> str:
    """Async subprocess with hard-kill on timeout."""
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
        return ""

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
        # Drain to release handles.
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            pass
        return ""

    if proc.returncode and proc.returncode != 0:
        return ""
    try:
        text = stdout.decode("utf-8", errors="replace")
    except Exception:
        return ""
    return _extract_text_from_stream(text)


# ---------------------------------------------------------------------------
# Multi-provider race
# ---------------------------------------------------------------------------


async def _call_planner_race(
    prompt: str,
    timeout: int,
    plan_length: int,
    claude_replicas: int = 3,
) -> tuple[str, str, float, dict[str, float]]:
    """Race ``claude_replicas`` Claude + 1 Gemini + 1 Codex.

    Returns (winning_text, winning_provider, winning_latency_seconds,
             completed_competitor_latencies_dict).

    "Winner" = first task that returns text parsing into >=1 valid action.
    All other tasks are cancelled when a winner is picked. If no provider
    produces a usable plan within ``timeout``, returns ("", "", elapsed, {}).
    """
    t0 = time.time()

    providers: list[tuple[str, Any]] = []
    for i in range(claude_replicas):
        providers.append((f"claude_{i}", _call_claude_async(prompt, timeout)))
    providers.append(("gemini", _call_gemini_async(prompt, timeout)))
    providers.append(("codex", _call_codex_async(prompt, timeout)))

    tasks_by_name: dict[str, asyncio.Task[str]] = {}
    for name, coro in providers:
        tasks_by_name[name] = asyncio.create_task(coro, name=name)

    pending = set(tasks_by_name.values())
    completed_latencies: dict[str, float] = {}
    winner_text = ""
    winner_provider = ""

    try:
        while pending:
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
                timeout=max(1.0, timeout - (time.time() - t0)),
            )
            if not done:
                # outer timeout expired
                break
            for t in done:
                name = t.get_name()
                completed_latencies[name] = round(time.time() - t0, 2)
                try:
                    text = t.result()
                except Exception as e:
                    logger.debug(f"provider {name} raised: {e}")
                    text = ""
                if winner_text:
                    continue
                if not text:
                    continue
                if _parse_plan(text, plan_length):
                    winner_text = text
                    winner_provider = name
                    break
            if winner_text:
                break
    finally:
        for t in pending:
            t.cancel()
        # Drain cancellations so we don't leak orphan subprocesses.
        if pending:
            try:
                await asyncio.wait(pending, timeout=2.0)
            except Exception:
                pass

    winning_latency = completed_latencies.get(winner_provider, round(time.time() - t0, 2))
    return winner_text, winner_provider, winning_latency, completed_latencies


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
    """v0.0b -- 5-way CLI race planner-executor agent."""

    MAX_ACTIONS: int = 80
    PLAN_LENGTH: int = 6
    PLANNER_TIMEOUT_SECS: int = 240
    CLAUDE_REPLICAS: int = 3
    FRAME_HISTORY_DEPTH: int = 3
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
        self._action_history: list[str] = []  # all actions taken, in order

    def is_done(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        planner_invoked = False
        winner_provider = ""
        winner_latency = 0.0
        competitor_latencies: dict[str, float] = {}
        plan_size = 0
        parse_ok = True
        response_preview = ""

        if not self._plan_queue:
            planner_invoked = True
            self._planner_call_count += 1
            prompt = self._build_planning_prompt(frames, latest_frame)
            response, winner_provider, winner_latency, competitor_latencies = (
                asyncio.run(
                    _call_planner_race(
                        prompt,
                        timeout=self.PLANNER_TIMEOUT_SECS,
                        plan_length=self.PLAN_LENGTH,
                        claude_replicas=self.CLAUDE_REPLICAS,
                    )
                )
            )
            response_preview = response[:300]
            self._plan_queue = _parse_plan(response, self.PLAN_LENGTH)
            plan_size = len(self._plan_queue)
            parse_ok = plan_size > 0
            if not self._plan_queue:
                self._plan_queue = [(GameAction.ACTION5, {})]

        action, data = self._plan_queue.pop(0)
        if data:
            action.set_data({**data, "game_id": self.game_id})

        if data:
            self._action_history.append(
                f"{action.name} x={data.get('x','?')} y={data.get('y','?')}"
            )
        else:
            self._action_history.append(action.name)

        self._log_decision(
            planner_invoked=planner_invoked,
            race_winner_provider=winner_provider,
            race_winner_latency_seconds=winner_latency,
            race_competitor_latencies=competitor_latencies,
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

        return textwrap.dedent(
            f"""\
            # ROLE
            You are planning the next few moves in an ARC-AGI-3 game. The
            world is a grid (matrix of cells with integer values 0-15);
            each action produces the next frame. Your objective is to
            reach state=WIN while minimizing actions and avoiding
            GAME_OVER.

            {card_block}
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
            """
        ).strip()

    def _log_decision(
        self,
        planner_invoked: bool,
        race_winner_provider: str,
        race_winner_latency_seconds: float,
        race_competitor_latencies: dict[str, float],
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
            "race_winner_provider": race_winner_provider,
            "race_winner_latency_seconds": race_winner_latency_seconds,
            "race_competitor_latencies": race_competitor_latencies,
            "plan_size": plan_size,
            "parse_ok": parse_ok,
            "action_chosen": action_name,
            "queue_depth_after": queue_depth_after,
            "planner_call_count": planner_call_count,
            "game_card_used": bool(_card_for_game(self.game_id)),
            "response_preview": response_preview if planner_invoked else "",
        }
        try:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass
