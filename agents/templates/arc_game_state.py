"""arc_game_state.py -- per-game-class symbolic state tracking.

Phase B.2 (current): player_pos tracked via FRAME-DIFF. The region that
CHANGES between consecutive frames IS the player. Validation on
operator's 621-action replay: player moves coherently step-to-step,
e.g. (38, 30) -> (38, 31) -> (39, 31) after right+down moves.

Phase B.1 MVP attempted single-frame heuristics ("smallest region in
size range", "closest to previous position"). Both failed because
ls20's yellow (0x04) appears in many static regions (arena background,
UI panels) AND the moving sprite -- picking from frame alone we
inevitably grabbed a static UI region.

What works:
* Level transition detection (step 28 registers 0 -> 1)
* Door / rotator / key_indicator centroids (static, single-frame OK)
* Frame-diff player tracking (Phase B.2)
* Wall discovery via blocked-move detection


Phase B.1 of the long-term plan. Replaces "LLM rediscovers everything
from the grid every turn" with structured state that gets incrementally
updated from each frame.

For LockSmith (ls20): track player position, current key shape/color,
energy, level, door position, rotators seen, walls discovered. The
agent's prompt then gets a compact "STATE" block + "WHAT JUST HAPPENED"
delta block per turn -- far more useful than 64x64 grid spam.

Reusable for new game classes: subclass GameState, override
``update_from_frame(...)``.

Public API:
* ``LockSmithState`` -- dataclass for ls20
* ``update_state(prev, frame, prev_action, color_map)`` -> ``GameState``
* ``state_diff(prev, curr)`` -> dict of changes
* ``state_for_game(game_id)`` -> a fresh state instance, or None
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
from skimage import measure

from . import arc_vision


def _find_player_via_diff(
    prev_frame: list[list[list[Any]]] | None,
    curr_frame: list[list[list[Any]]],
    prev_player_pos: tuple[float, float] | None,
) -> tuple[tuple[float, float] | None, int]:
    """Identify the player as the region of cells that changed between
    consecutive frames.

    Phase B.2: connected-component analysis over the diff mask.
    * Player typically occupies ~12-30 changed cells per move (4x4 sprite
      partially overlapping with old position).
    * We prefer regions in [4, 60] cells (filters out 1-cell noise and
      huge UI repaints) and, when prev_player_pos is known, pick the
      closest such region.

    Returns (None, 0) when there's no prev_frame (first observation)
    or no motion -- caller falls back to prev_player_pos.
    """
    if prev_frame is None or not curr_frame:
        return None, 0
    try:
        f0 = np.array(prev_frame[0])
        f1 = np.array(curr_frame[0])
    except Exception:
        return None, 0
    if f0.shape != f1.shape:
        return None, 0

    diff = f0 != f1
    if not diff.any():
        return None, 0  # no motion this step

    labels = measure.label(diff, connectivity=2)
    regions = measure.regionprops(labels)
    if not regions:
        return None, 0

    # Filter to plausible sprite sizes.
    candidates = [r for r in regions if 4 <= r.area <= 60]
    if not candidates:
        # No "sprite-sized" change; might be a level transition with
        # many cells repainted. Pick the smallest changed region as a
        # rough proxy (rarely correct, but bounded).
        candidates = sorted(regions, key=lambda r: r.area)[:3]

    if prev_player_pos is not None:
        def _dist(r: Any) -> float:
            return (r.centroid[0] - prev_player_pos[0]) ** 2 + (
                r.centroid[1] - prev_player_pos[1]
            ) ** 2
        best = min(candidates, key=_dist)
    else:
        # First observed motion -- pick the largest plausible region
        # (player sprite usually dominates the diff on first move).
        best = max(candidates, key=lambda r: r.area)

    return (round(float(best.centroid[0]), 1),
            round(float(best.centroid[1]), 1)), int(best.area)


@dataclass
class LockSmithState:
    """Snapshot of a LockSmith (ls20) game at one frame."""

    game_id: str = ""
    action_counter: int = 0
    level: int = 0
    state_label: str = "NOT_FINISHED"  # NOT_PLAYED / NOT_FINISHED / WIN / GAME_OVER

    player_pos: tuple[float, float] | None = None  # (row, col) centroid
    player_size: int = 0  # cell area of detected player region

    door_pos: tuple[float, float] | None = None
    door_size: int = 0

    # Rotators: list of (row, col, size). Two kinds (shape vs color) per card,
    # but at this layer we just track positions; semantic kind is inferred by
    # the planner from card knowledge.
    rotators: list[tuple[float, float, int]] = field(default_factory=list)

    # Key indicator -- small region adjacent to player showing current key.
    key_indicator_pos: tuple[float, float] | None = None
    key_indicator_size: int = 0

    # Cumulative "walls discovered" -- cells where the player attempted
    # to move and failed (state didn't change). Set of (row, col) tuples.
    walls_seen: set[tuple[int, int]] = field(default_factory=set)

    # Last action attempted (filled by tracker after frame arrives).
    last_action: str = ""
    last_action_succeeded: bool = True  # False if frame is byte-identical to prev

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Sets aren't JSON-serializable; lists are.
        d["walls_seen"] = sorted(list(self.walls_seen))
        return d

    def to_prompt_block(self) -> str:
        """Render as a compact block for LLM prompts."""
        lines = [
            f"  level: {self.level}",
            f"  state: {self.state_label}",
            f"  player_at: {self.player_pos}",
            f"  door_at: {self.door_pos}",
            f"  key_indicator_at: {self.key_indicator_pos}",
        ]
        if self.rotators:
            rs = ", ".join(f"({r:.0f},{c:.0f})" for r, c, _ in self.rotators[:6])
            lines.append(f"  rotators: [{rs}]")
        if self.walls_seen:
            lines.append(f"  walls_discovered: {len(self.walls_seen)} cells")
        lines.append(f"  last_action: {self.last_action} "
                     f"(succeeded={self.last_action_succeeded})")
        return "\n".join(lines)


def _extract_centroid(
    objs: dict[str, list[dict[str, Any]]],
    label: str,
    prefer_smaller: bool = False,
    near: tuple[float, float] | None = None,
    size_range: tuple[int, int] | None = None,
) -> tuple[tuple[float, float] | None, int]:
    """Pick the most-likely instance of ``label``.

    Args:
        prefer_smaller: pick smallest region (e.g. player sprite vs arena bg).
        near: if given, prefer region whose centroid is closest to this point.
        size_range: filter regions to (min, max) area inclusive before picking.
    """
    instances = objs.get(label, [])
    if not instances:
        return None, 0
    pool = instances
    if size_range is not None:
        lo, hi = size_range
        pool = [o for o in instances if lo <= o["area"] <= hi] or instances
    if near is not None:
        def _d(o: dict[str, Any]) -> float:
            r, c = o["centroid"]
            return (r - near[0]) ** 2 + (c - near[1]) ** 2
        best = min(pool, key=_d)
    elif prefer_smaller:
        small = [o for o in pool if o["area"] < 200]
        pool2 = small if small else pool
        best = min(pool2, key=lambda o: o["area"])
    else:
        best = max(pool, key=lambda o: o["area"])
    c = best["centroid"]
    return (round(float(c[0]), 1), round(float(c[1]), 1)), int(best["area"])


def update_locksmith_state(
    prev: LockSmithState,
    frame: list[list[list[Any]]],
    prev_action: str,
    action_counter: int,
    level: int,
    state_label: str,
    prev_frame: list[list[list[Any]]] | None = None,
) -> LockSmithState:
    """Build a new state from the current frame + the action that produced it."""
    objs = arc_vision.detect_objects(
        frame,
        color_name_map=arc_vision.LOCKSMITH_COLOR_NAMES,
        min_size=2,
    )

    # Phase B.2 player tracking: frame-diff detection.
    # The region of cells that CHANGED between prev_frame and frame IS
    # the player (it's the only thing moving in a single action step).
    # When no motion (frame-equal => blocked) or first observation
    # (no prev_frame), retain prev.player_pos as the best guess.
    diff_pos, diff_size = _find_player_via_diff(
        prev_frame, frame, prev.player_pos
    )
    if diff_pos is not None:
        player_pos, player_size = diff_pos, diff_size
    else:
        # No motion / no prior frame -- keep last known position.
        # For the very first frame (prev.player_pos is None too), fall
        # back to picking the smallest play_area_bg region as a rough
        # initial guess (will lock in once the player actually moves).
        if prev.player_pos is not None:
            player_pos, player_size = prev.player_pos, 0
        else:
            player_pos, player_size = _extract_centroid(
                objs, "play_area_bg", prefer_smaller=True,
                size_range=(8, 60),
            )

    door_pos, door_size = _extract_centroid(objs, "door_border")
    key_pos, key_size = _extract_centroid(objs, "key_indicator")

    rotators = [
        (round(float(o["centroid"][0]), 1),
         round(float(o["centroid"][1]), 1),
         int(o["area"]))
        for o in sorted(objs.get("rotator_marker", []),
                        key=lambda o: -o["area"])[:6]
    ]

    # Did the last action succeed? Frame-equal-to-previous means we bumped
    # a wall (or rotator rotation has no visible effect on grid -- harder
    # case, skip for now).
    succeeded = True
    walls_seen = set(prev.walls_seen)
    if prev_frame is not None and prev_action in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
        f0 = np.array(prev_frame[0]) if prev_frame else None
        f1 = np.array(frame[0]) if frame else None
        if f0 is not None and f1 is not None and f0.shape == f1.shape:
            if np.array_equal(f0, f1):
                succeeded = False
                # Record wall adjacent to player in direction of attempted action.
                if prev.player_pos is not None:
                    pr, pc = prev.player_pos
                    # Player sprite is ~4x4; record wall cells 1 step beyond the centroid
                    # in the action's direction.
                    dr, dc = {
                        "ACTION1": (-3, 0),  # up
                        "ACTION2": (3, 0),   # down
                        "ACTION3": (0, -3),  # left
                        "ACTION4": (0, 3),   # right
                    }[prev_action]
                    wr, wc = int(round(pr + dr)), int(round(pc + dc))
                    if 0 <= wr < 64 and 0 <= wc < 64:
                        walls_seen.add((wr, wc))

    return LockSmithState(
        game_id=prev.game_id,
        action_counter=action_counter,
        level=level,
        state_label=state_label,
        player_pos=player_pos,
        player_size=player_size,
        door_pos=door_pos,
        door_size=door_size,
        rotators=rotators,
        key_indicator_pos=key_pos,
        key_indicator_size=key_size,
        walls_seen=walls_seen,
        last_action=prev_action,
        last_action_succeeded=succeeded,
    )


def state_diff(prev: LockSmithState, curr: LockSmithState) -> dict[str, Any]:
    """Compact diff for LLM prompt -- only fields that changed."""
    out: dict[str, Any] = {}
    if prev.level != curr.level:
        out["level"] = f"{prev.level} -> {curr.level}"
    if prev.state_label != curr.state_label:
        out["state"] = f"{prev.state_label} -> {curr.state_label}"
    if prev.player_pos != curr.player_pos:
        out["player_moved"] = f"{prev.player_pos} -> {curr.player_pos}"
    if prev.key_indicator_pos != curr.key_indicator_pos:
        out["key_changed"] = f"{prev.key_indicator_pos} -> {curr.key_indicator_pos}"
    if len(prev.walls_seen) != len(curr.walls_seen):
        new_walls = curr.walls_seen - prev.walls_seen
        if new_walls:
            out["new_walls"] = sorted(list(new_walls))
    if not curr.last_action_succeeded and curr.last_action:
        out["action_blocked"] = curr.last_action
    return out


def state_for_game(game_id: str) -> LockSmithState | None:
    """Return a fresh state instance for the game prefix, or None if unknown."""
    if not game_id:
        return None
    prefix = game_id.split("-", 1)[0]
    if prefix == "ls20":
        return LockSmithState(game_id=game_id)
    return None
