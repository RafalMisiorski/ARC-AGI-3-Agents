"""nh_arc_hierarchical_v0 -- LLM-free hierarchical planner agent.

Phase C.2 of the long-term roadmap. Combines:
* arc_vision (Phase A) -- symbolic object detection
* arc_game_state (Phase B) -- per-game state tracking with frame-diff
  player localisation
* arc_planner (Phase C.1) -- A* pathfinding + sub-goal decomposition

This MVP makes NO LLM calls -- it's pure Python. The goal is to validate
that vision + state + planner can actually solve a level (level 1 of
LockSmith in particular), since:
* Operator's manual replay solved level 1 in 28 actions.
* Phase B.2 frame-diff tracking gives reliable player_pos.
* Phase C.1 A* yields a 5-action optimistic plan from start to door
  via the nearest rotator.

If this scores >= 1 level on ls20, we've broken the persistent "0
levels" pattern WITHOUT spending any LLM budget. If it still scores 0,
the ARC API-protocol issue (Day 3 unfixed) is confirmed as the binding
constraint, not the agent's brain.

Phase C.3 (next) will add an LLM hook for high-level decisions (which
rotator to step on first when shape vs colour both need rotating) plus
a richer prompt context when stuck.
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any

from arcengine import FrameData, GameAction, GameState

from ..agent import Agent
from . import arc_game_state, arc_planner, arc_vision, cost_tracker

logger = logging.getLogger(__name__)

_FALLBACK_EXPLORE_ACTIONS = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]


class NhArcHierarchicalV0(Agent):
    """Vision + state + pathfinding. Zero LLM calls in MVP."""

    MAX_ACTIONS: int = 80
    STUCK_REPLAN_THRESHOLD: int = 2  # blocked actions before replan
    LOG_DIR: str = "logs"

    # Phase C.2 has no LLM, so no $ budget gate is needed; the field is
    # kept for telemetry parity with the cascade agent.
    BUDGET_HARD_USD: float = 0.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        Path(self.LOG_DIR).mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._log_path = (
            Path(self.LOG_DIR) / f"hierarchical_v0_{self.game_id}_{ts}.jsonl"
        )

        self._game_state = arc_game_state.state_for_game(self.game_id)
        self._prev_frame: list | None = None
        self._plan: arc_planner.HierarchicalPlan | None = None
        self._action_queue: list[str] = []
        self._stuck_counter: int = 0
        self._planner_decisions: int = 0  # count of plan generations
        self._rng = random.Random(42)  # deterministic explore tiebreaks

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def take_action(self, action: GameAction):
        # ALWAYS set game_id; harmless if SDK ignores it.
        action.set_data({"game_id": self.game_id})
        logger.warning(
            f"[HIER SEND] {action.name} (id={action.value})"
        )
        frame = super().take_action(action)
        if frame is None:
            logger.warning("[HIER RECV] frame=None")
        else:
            logger.warning(
                f"[HIER RECV] state={frame.state.name} "
                f"levels={frame.levels_completed} "
                f"avail={list(frame.available_actions or [])}"
            )
        return frame

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        # 1. Update state from latest frame.
        last_action = self._last_action_name()
        if self._game_state is not None:
            self._game_state = arc_game_state.update_locksmith_state(
                prev=self._game_state,
                frame=latest_frame.frame,
                prev_action=last_action,
                action_counter=self.action_counter,
                level=latest_frame.levels_completed,
                state_label=latest_frame.state.name,
                prev_frame=self._prev_frame,
            )
        self._prev_frame = latest_frame.frame

        # 2. Stuck-state detection: a blocked action means our path is
        # wrong (we just discovered a wall). Burst replan after a couple.
        if (
            self._game_state is not None
            and not self._game_state.last_action_succeeded
            and self._game_state.last_action  # ignore the very first step
        ):
            self._stuck_counter += 1
            if self._stuck_counter >= self.STUCK_REPLAN_THRESHOLD:
                logger.warning(
                    f"[HIER STUCK] {self._stuck_counter} blocked actions "
                    "-- discarding queue and replanning"
                )
                self._action_queue.clear()
                self._stuck_counter = 0
        else:
            self._stuck_counter = 0

        # 3. Level transition -> reset plan (level 2 is a new map).
        if (
            self._plan is not None
            and self._game_state is not None
            and self._game_state.level > 0
            and self._planner_decisions
            and self._plan.done
        ):
            self._plan = None
            self._action_queue.clear()

        # 4. Generate / advance plan as needed.
        if self._plan is None:
            self._plan = self._generate_plan()
            self._planner_decisions += 1
        elif (
            self._plan.current_goal is not None
            and self._game_state is not None
            and arc_planner.goal_reached(
                self._plan.current_goal, self._game_state.player_pos
            )
        ):
            self._plan.advance()
            self._action_queue.clear()

        # 5. Build action queue if empty.
        if not self._action_queue:
            self._refill_action_queue()

        # 6. Pop next action.
        if self._action_queue:
            action_name = self._action_queue.pop(0)
        else:
            action_name = self._explore_action()

        try:
            action = GameAction.from_name(action_name)
        except Exception:
            action = GameAction.ACTION5

        if self._plan is not None and self._plan.current_goal is not None:
            self._plan.actions_in_current += 1
            # Abandon a sub-goal that's burning its budget.
            if self._plan.actions_in_current >= self._plan.current_goal.max_actions:
                logger.warning(
                    f"[HIER GOAL TIMEOUT] {self._plan.current_goal.target_label} "
                    f"after {self._plan.actions_in_current} actions"
                )
                self._plan.advance()
                self._action_queue.clear()

        self._log_step(action_name)
        return action

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _last_action_name(self) -> str:
        if not hasattr(self, "_history"):
            self._history: list[str] = []
        return self._history[-1] if self._history else ""

    def _record_history(self, action_name: str) -> None:
        if not hasattr(self, "_history"):
            self._history = []
        self._history.append(action_name)

    def _generate_plan(self) -> arc_planner.HierarchicalPlan:
        if self._game_state is None:
            return arc_planner.HierarchicalPlan()
        return arc_planner.decompose_simple(
            player_pos=self._game_state.player_pos,
            door_pos=self._game_state.door_pos,
            rotators=self._game_state.rotators,
            level=self._game_state.level,
        )

    def _refill_action_queue(self) -> None:
        if self._plan is None or self._game_state is None:
            return
        goal = self._plan.current_goal
        if goal is None or goal.target_pos is None:
            return
        if self._game_state.player_pos is None:
            return
        actions = arc_planner.pathfind_to_actions(
            self._game_state.player_pos,
            goal.target_pos,
            walls=self._game_state.walls_seen,
        )
        if actions:
            # Take only the first 2-3 steps -- replan after we observe
            # what actually happened (frame-diff player tracking might
            # disagree with our model after wall bumps).
            self._action_queue = actions[: min(3, len(actions))]

    def _explore_action(self) -> str:
        """Fallback when planner has nothing to offer (no goal / no path)."""
        return self._rng.choice(_FALLBACK_EXPLORE_ACTIONS)

    def _log_step(self, action_name: str) -> None:
        self._record_history(action_name)
        record: dict[str, Any] = {
            "ts": time.time(),
            "game_id": self.game_id,
            "action_counter": self.action_counter,
            "action_chosen": action_name,
            "queue_depth_after": len(self._action_queue),
            "planner_decisions": self._planner_decisions,
            "stuck_counter": self._stuck_counter,
        }
        if self._game_state is not None:
            record["player_pos"] = self._game_state.player_pos
            record["level"] = self._game_state.level
            record["walls_seen"] = len(self._game_state.walls_seen)
            record["last_action_succeeded"] = self._game_state.last_action_succeeded
        if self._plan is not None and self._plan.current_goal is not None:
            cg = self._plan.current_goal
            record["current_goal"] = cg.target_label
            record["current_goal_pos"] = list(cg.target_pos) if cg.target_pos else None
            record["actions_in_current"] = self._plan.actions_in_current
        try:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass
