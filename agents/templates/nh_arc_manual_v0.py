"""nh_arc_manual_v0 -- interactive REPL agent for ground-truth playthroughs.

Operator types actions in the terminal, frames are rendered to PNG and
auto-opened (Windows ``os.startfile``). Recording is auto-saved by the
framework, so the resulting JSONL can be played back via the SDK's
``Playback`` agent to verify deterministic reproduction.

USE
---
    uv run main.py --agent=nharcmanualv0 --game=ls20

Then in the REPL:
    Action: ACTION1                # simple
    Action: ACTION6 12 34          # click at (12, 34)
    Action: ACTION6 x=12 y=34      # also accepted
    Action: RESET
    Action: QUIT                   # exit cleanly

OUTCOME
-------
* If you can reach ``levels_completed=1`` manually -> SDK works end to end
  on current env hash. Recording is saved in recordings/, can be replayed
  via Playback agent for any future automated verification.
* If you cannot reach level 1 manually despite knowing the mechanics ->
  SDK has a real bug between client and server-side level tracking.
  File an issue on arcprize/ARC-AGI-3-Agents with the recording.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from arcengine import FrameData, GameAction, GameState

from ..agent import Agent
from . import arc_vision

logger = logging.getLogger(__name__)


class NhArcManualV0(Agent):
    """Interactive REPL agent. Operator types each action; PNG auto-opens."""

    MAX_ACTIONS: int = 1000  # operator decides when to quit
    FRAMES_DIR: str = "logs/manual_frames"
    AUTO_OPEN_PNG: bool = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._frames_dir = Path(self.FRAMES_DIR) / self.game_id
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        self._quit = False
        self._last_opened_path: Path | None = None
        print("\n" + "=" * 70)
        print(f"MANUAL REPL  game={self.game_id}")
        print(f"  Frames saved to: {self._frames_dir.resolve()}")
        print(f"  Recording: framework auto-saves to recordings/")
        print(f"  Commands: ACTION1-7 [x=N y=N] | RESET | QUIT")
        print("=" * 70)

    def is_done(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> bool:
        return self._quit or latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        # 1. Render frame to PNG
        png_path = self._frames_dir / f"{self.action_counter:03d}.png"
        rendered = False
        try:
            arc_vision.render_frame_to_png(latest_frame.frame, png_path)
            rendered = True
        except Exception as e:
            print(f"[render failed: {e}]")

        # 2. Auto-open PNG (Windows: default image viewer)
        if rendered and self.AUTO_OPEN_PNG and self._last_opened_path != png_path:
            try:
                if sys.platform.startswith("win"):
                    os.startfile(str(png_path))
                self._last_opened_path = png_path
            except Exception as e:
                logger.debug(f"auto-open failed: {e}")

        # 3. Print status
        print("\n" + "-" * 70)
        print(f"Step {self.action_counter}/{self.MAX_ACTIONS}")
        print(f"  State: {latest_frame.state.name}")
        print(f"  Levels completed: {latest_frame.levels_completed}")
        print(f"  Available actions: {list(latest_frame.available_actions or [])}")
        if rendered:
            print(f"  PNG: {png_path.resolve()}")

        # 4. Read action from operator
        while True:
            try:
                raw = input(">> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("[KeyboardInterrupt -> QUIT]")
                self._quit = True
                return GameAction.ACTION5

            if not raw:
                continue
            upper = raw.upper()

            if upper in ("QUIT", "Q", "EXIT"):
                self._quit = True
                return GameAction.ACTION5  # placeholder; is_done returns True next

            parts = upper.split()
            name = parts[0]
            try:
                action = GameAction.from_name(name)
            except Exception:
                print(f"  Invalid action name: {name!r}. Try ACTION1-7 or RESET.")
                continue

            data: dict = {}
            if name == "ACTION6":
                xy = self._parse_xy(parts[1:])
                if xy is None:
                    print(f"  ACTION6 needs x,y in [0,63]. Examples:")
                    print(f"    ACTION6 12 34")
                    print(f"    ACTION6 x=12 y=34")
                    continue
                data = {"x": str(xy[0]), "y": str(xy[1])}

            # Always set game_id (some env wrappers need it)
            action.set_data({**data, "game_id": self.game_id})
            return action

    @staticmethod
    def _parse_xy(tokens: list[str]) -> tuple[int, int] | None:
        """Accept '12 34', 'x=12 y=34', 'X=12,Y=34', etc."""
        cleaned = " ".join(tokens).replace(",", " ").replace(";", " ")
        cleaned = cleaned.replace("x=", "").replace("X=", "")
        cleaned = cleaned.replace("y=", "").replace("Y=", "")
        nums = cleaned.split()
        if len(nums) < 2:
            return None
        try:
            x = int(nums[0])
            y = int(nums[1])
        except ValueError:
            return None
        if not (0 <= x <= 63 and 0 <= y <= 63):
            return None
        return x, y
