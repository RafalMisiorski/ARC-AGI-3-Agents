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
# Phase C.3 was attempted (LLM hook on stuck-state via vision_v0 wrappers)
# but reverted on 2026-05-20: subprocess-based vision calls hung 4-14 min
# per call regardless of cwd choice (tmp -> Gemini "empty workspace" intro,
# home -> Gemini analyses whole home dir before responding). Accept pure
# A* + symbolic state as the hierarchical track; focus shifts to baseline
# text-cascade with broader GAME_CARDS coverage.

logger = logging.getLogger(__name__)

_FALLBACK_EXPLORE_ACTIONS = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]


class NhArcHierarchicalV0(Agent):
    """Vision + state + pathfinding. Pure A* navigation, zero LLM calls.

    Phase C.2 design: symbolic perception (arc_vision) + game state
    tracking (arc_game_state, with frame-diff player localisation) +
    A* pathfinding (arc_planner). The MVP that validates whether the
    symbolic stack alone can clear a level.

    Phase C.3 vision-LLM hook was attempted and reverted: subprocess
    vision calls hung 4-14 min regardless of cwd, with empty/unparseable
    outputs in every fire. Pure A* remains the track here; LLM-driven
    planning lives in nh_arc_baseline_v0 (text cascade with GAME_CARDS).
    """

    MAX_ACTIONS: int = 80
    STUCK_REPLAN_THRESHOLD: int = 2  # blocked actions before replan
    LOG_DIR: str = "logs"

    BUDGET_HARD_USD: float = 0.0  # no LLM calls in this agent

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
                self._plan = None  # force regen + LLM hook check in step 4
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

    # ------------------------------------------------------------------
    # Phase C.3 vision LLM hook
    # ------------------------------------------------------------------

    def _maybe_invoke_llm_hook(self, latest_frame: FrameData) -> list[str]:
        """Fire vision LLM hook if any trigger matches and budget remains.

        Triggers (any):
          (a) stuck-replan trigger: consecutive A* replans without level
              progress >= LLM_TRIGGER_CONSECUTIVE_REPLANS.
          (b) cold-start periodic trigger: after PERIODIC_FIRST_CALL_AT
              actions, if no LLM call has fired and level == 0.
          (c) refresh trigger: every PERIODIC_REFRESH_EVERY actions after
              the last LLM call, if level == 0.

        Returns parsed plan as a list of action name strings, or [] if
        nothing should fire (or LLM returned nothing usable).
        """
        PERIODIC_FIRST_CALL_AT = 12   # cold-start: first hook after 12 actions
        PERIODIC_REFRESH_EVERY = 18   # refresh every 18 actions on stuck level

        if self._game_state is None:
            return []
        if self._llm_call_count >= self.LLM_HOOK_BUDGET:
            return []
        if self._budget_exceeded:
            return []

        trigger_a = (
            self._consecutive_replans_without_progress
            >= self.LLM_TRIGGER_CONSECUTIVE_REPLANS
        )
        trigger_b = (
            self._llm_call_count == 0
            and self.action_counter >= PERIODIC_FIRST_CALL_AT
            and self._game_state.level == 0
        )
        trigger_c = (
            self._llm_call_count > 0
            and self._game_state.level == 0
            and (self.action_counter - self._action_at_last_llm_call)
            >= PERIODIC_REFRESH_EVERY
        )

        if not (trigger_a or trigger_b or trigger_c):
            return []

        png_path = self._frames_dir / f"{self.action_counter:03d}_stuck.png"
        try:
            arc_vision.render_frame_to_png(latest_frame.frame, png_path)
        except Exception as e:
            logger.warning(f"[HIER LLM HOOK] render failed: {e}")
            return []

        prompt = self._build_stuck_prompt(latest_frame)
        text, latency = _call_claude_vision(
            png_path, prompt, timeout=self.VISION_TIMEOUT_SECS
        )
        provider = "claude_vision"
        cost = self.EST_COST_PER_CLAUDE_CALL_USD if text else 0.0

        plan = _parse_plan(text, self.LLM_PLAN_LENGTH)
        if not plan:
            text, latency = _call_gemini_vision(
                png_path, prompt, timeout=self.VISION_TIMEOUT_SECS * 2
            )
            provider = "gemini_vision"
            cost = 0.0
            plan = _parse_plan(text, self.LLM_PLAN_LENGTH)

        self._llm_call_count += 1
        self._action_at_last_llm_call = self.action_counter
        self._game_cost_usd += cost
        if self._game_cost_usd >= self.BUDGET_HARD_USD:
            self._budget_exceeded = True
            logger.warning(
                f"[HIER BUDGET HARD STOP] ${self._game_cost_usd:.2f}"
            )

        try:
            cost_tracker.record_call(
                provider=provider,
                model="sonnet" if provider == "claude_vision" else "gemini-flash-2.5",
                usage={"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0},
                duration_s=latency,
                game_id=self.game_id,
                action_counter=self.action_counter,
                parse_ok=bool(plan),
                status="winner" if plan else "empty",
                caller_module="nh_arc_hierarchical_v0",
                purpose="arc_stuck_hook",
            )
        except Exception:
            pass

        if not plan:
            logger.warning(
                f"[HIER LLM HOOK #{self._llm_call_count}] empty/unparseable "
                f"from {provider} (latency {latency:.1f}s)"
            )
            return []

        action_names: list[str] = []
        for ga, data in plan:
            if data:
                # Skip ACTION6 with x,y for MVP -- hierarchical action_queue
                # holds bare strings; data injection would need separate path.
                continue
            action_names.append(ga.name)

        if not action_names:
            logger.warning(
                f"[HIER LLM HOOK #{self._llm_call_count}] plan was ACTION6-only "
                f"from {provider}; dropped (MVP doesn't inject coords yet)"
            )
            return []

        logger.warning(
            f"[HIER LLM HOOK #{self._llm_call_count}/{self.LLM_HOOK_BUDGET}] "
            f"{provider} -> {len(action_names)}-action plan injected: "
            f"{action_names}"
        )
        return action_names

    def _build_stuck_prompt(self, latest_frame: FrameData) -> str:
        avail_names: list[str] = []
        for aid in latest_frame.available_actions or []:
            try:
                avail_names.append(GameAction.from_id(aid).name)
            except Exception:
                pass
        avail = ", ".join(avail_names) or (
            "RESET, ACTION1, ACTION2, ACTION3, ACTION4, ACTION5, ACTION6, ACTION7"
        )

        card = _card_for_game(self.game_id)
        card_block = f"\n# GAME-SPECIFIC NOTES\n{card}\n" if card else ""

        state_info = "(no symbolic state available)"
        if self._game_state is not None:
            state_info = (
                f"Player position: {self._game_state.player_pos}\n"
                f"Door position:   {self._game_state.door_pos}\n"
                f"Walls seen:      {len(self._game_state.walls_seen)}\n"
                f"Level:           {self._game_state.level}\n"
                f"Rotators:        {len(self._game_state.rotators)}\n"
                f"Last action:     {self._game_state.last_action or '(none)'}\n"
                f"Last succeeded:  {self._game_state.last_action_succeeded}\n"
            )

        return (
            "# ROLE\n"
            f"You see one frame of an ARC-AGI-3 game. The A* planner has been\n"
            f"stuck for {self._consecutive_replans_without_progress} replans without "
            "level progress.\n"
            f"Look at the image, account for the symbolic state estimate (below),\n"
            f"and suggest {self.LLM_PLAN_LENGTH} actions that might break the\n"
            "deadlock. Prefer simple movement (ACTION1-4) over clicks (ACTION6)\n"
            "for this hook -- coordinate clicks aren't wired in yet.\n"
            f"{card_block}"
            "\n"
            "# CURRENT SYMBOLIC STATE\n"
            f"{state_info}\n"
            "# AVAILABLE ACTIONS\n"
            f"{avail}\n"
            "ACTION1=up, ACTION2=down, ACTION3=left, ACTION4=right.\n"
            "\n"
            "# OUTPUT FORMAT (strict)\n"
            f"Reply with EXACTLY {self.LLM_PLAN_LENGTH} lines, one action token per\n"
            "line, no prose, no markdown, no JSON. Use ACTION1-5 or ACTION7.\n"
        )

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
