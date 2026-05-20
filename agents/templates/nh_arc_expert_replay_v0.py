"""nh_arc_expert_replay_v0 -- deterministic hardcoded-sequence agent.

Zero LLM, zero state tracking, zero heuristics. Just a hardcoded action
queue per game_id prefix. Sends actions verbatim via the framework SDK
and logs what the server returns after each step.

PURPOSE
-------
Definitive SDK protocol validator. Day 3 commit said "ls20 expert-injection
run: 14 plans, 28/28 expert moves replicated verbatim, cost $2.76, 0
levels" -- but that run went through the LLM cascade, so the LLM *could*
have deviated despite the "MANDATORY VERBATIM" instruction in GAME_CARDS.
This agent removes the LLM entirely. If 28 actions verbatim still yields
0 levels, the SDK has a real bug. If it yields 1 level, the previous
runs failed because the LLM didn't actually replay verbatim.

OUTCOME INTERPRETATION
----------------------
* levels_completed == 1 after action 28 -> SDK works. The 0-levels we've
  been seeing are a prompt-engineering / LLM-fidelity problem. Fix is
  to hardcode action queues for known winning sequences and skip the
  LLM for the recorded prefix.
* levels_completed == 0 after action 28 -> SDK or game-engine has a
  subtle bug (timing, data field, env_id variant). Worth opening an
  issue on arcprize/ARC-AGI-3-Agents with this reproducer.
* state == GAME_OVER mid-sequence -> the recorded sequence is wrong
  (energy depleted, walked into wall) OR the game state has changed
  since the recording (different env_id hash).

Per-step JSONL log captures the data needed for either diagnosis.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from arcengine import FrameData, GameAction, GameState

from ..agent import Agent

logger = logging.getLogger(__name__)


# Hardcoded expert sequences per env_id prefix.
# Source: operator's manual REPL playthrough on 2026-05-20 via
# nh_arc_manual_v0 (game_id ls20-9607627b, scorecard 96edc65e-...).
# Reached levels_completed=1 at step 14 (15 actions total, vs human
# baseline 22). The legacy 28-action sequence in GAME_CARDS was stale
# for the current env hash and yielded 0 levels.
EXPERT_SEQUENCES: dict[str, list[str]] = {
    "ls20": [
        # Steps 0-4
        "ACTION1", "ACTION2", "ACTION3", "ACTION3", "ACTION3",
        # Steps 5-9
        "ACTION1", "ACTION1", "ACTION1", "ACTION1", "ACTION4",
        # Steps 10-14 (level 1 achieved at the final ACTION1)
        "ACTION4", "ACTION4", "ACTION1", "ACTION1", "ACTION1",
    ],
}

# Where to look for agent-side action logs written by nh_arc_manual_v0.
# Format: one JSON per line with {action_name, action_data, action_counter}.
# Latest matching <prefix>_*.jsonl wins if no hardcoded entry exists.
_ACTIONS_DIR = Path("logs/manual_actions")


def _load_sequence_from_jsonl(game_id_prefix: str) -> list[tuple[str, dict]]:
    """Find newest manual_actions/<prefix>_*.jsonl and parse action sequence."""
    if not _ACTIONS_DIR.is_dir():
        return []
    candidates = sorted(
        _ACTIONS_DIR.glob(f"{game_id_prefix}*.jsonl"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        return []
    out: list[tuple[str, dict]] = []
    for line in candidates[-1].read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            name = rec.get("action_name")
            data = rec.get("action_data") or {}
            if isinstance(name, str) and name:
                out.append((name, dict(data)))
        except json.JSONDecodeError:
            continue
    return out


class NhArcExpertReplayV0(Agent):
    """Deterministic action-queue agent. Replays a hardcoded sequence."""

    MAX_ACTIONS: int = 200  # high cap; sequence length is the real bound
    LOG_DIR: str = "logs"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        Path(self.LOG_DIR).mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._log_path = (
            Path(self.LOG_DIR) / f"expert_replay_v0_{self.game_id}_{ts}.jsonl"
        )

        key = self.game_id.split("-", 1)[0]
        # Priority:
        # 1. Hardcoded EXPERT_SEQUENCES (curated, fast)
        # 2. Latest agent-side JSONL from manual_v0 playthrough (auto-loaded)
        # 3. Empty -> agent idles
        self._sequence: list[tuple[str, dict]] = []
        self._source: str = "none"
        if key in EXPERT_SEQUENCES:
            self._sequence = [(n, {}) for n in EXPERT_SEQUENCES[key]]
            self._source = "hardcoded"
        else:
            loaded = _load_sequence_from_jsonl(key)
            if loaded:
                self._sequence = loaded
                self._source = "jsonl"

        if not self._sequence:
            logger.warning(
                f"[EXPERT REPLAY] No sequence for game_id prefix {key!r} "
                f"(checked hardcoded + {_ACTIONS_DIR}). Agent will idle "
                f"on ACTION5."
            )
        else:
            logger.warning(
                f"[EXPERT REPLAY] Loaded {len(self._sequence)} actions "
                f"({self._source}) for prefix {key!r}. Replaying verbatim."
            )

        self._level_at_start: int = 0
        self._sequence_exhausted_at: int | None = None

    def is_done(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> bool:
        if latest_frame.state is GameState.WIN:
            return True
        # Stop a few steps past sequence end so we observe the final state.
        if self._sequence_exhausted_at is not None:
            if self.action_counter >= self._sequence_exhausted_at + 3:
                return True
        return False

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        data: dict = {}
        if self.action_counter < len(self._sequence):
            name, data = self._sequence[self.action_counter]
        else:
            if self._sequence_exhausted_at is None:
                self._sequence_exhausted_at = self.action_counter
                logger.warning(
                    f"[EXPERT REPLAY] Sequence exhausted at action "
                    f"{self.action_counter}. Padding ACTION5 for 3 more "
                    f"steps to observe terminal state."
                )
            name = "ACTION5"

        try:
            action = GameAction.from_name(name)
        except Exception:
            logger.warning(f"[EXPERT REPLAY] Bad action name {name!r}, "
                           f"falling back to ACTION5")
            action = GameAction.ACTION5

        if data:
            action.set_data({**data, "game_id": self.game_id})

        self._log_step(
            action_name=action.name,
            state_name=latest_frame.state.name,
            levels_completed=latest_frame.levels_completed,
            available=list(latest_frame.available_actions or []),
        )
        return action

    def _log_step(
        self,
        action_name: str,
        state_name: str,
        levels_completed: int,
        available: list[int],
    ) -> None:
        record = {
            "ts": time.time(),
            "game_id": self.game_id,
            "action_counter": self.action_counter,
            "action_sent": action_name,
            "state_before": state_name,
            "levels_completed_before": levels_completed,
            "available_actions": available,
            "sequence_position": self.action_counter,
            "sequence_total": len(self._sequence),
            "exhausted": self._sequence_exhausted_at is not None,
        }
        try:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass
