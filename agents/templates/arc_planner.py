"""arc_planner.py -- hierarchical planning building blocks.

Phase C.1 of the long-term plan: cheap, deterministic primitives that
turn "navigate to (row, col)" into a concrete action sequence WITHOUT
invoking an LLM. The LLM stays for high-level goal selection
("which rotator? key match achieved?") -- everything tactical runs
on Python here.

Public API:
* ``SubGoal`` -- dataclass describing one tactical objective
* ``HierarchicalPlan`` -- ordered list of sub-goals + progress cursor
* ``astar_pathfind(start, goal, walls, grid_size, step)`` -> list[(r,c)]
* ``pathfind_to_actions(start, goal, walls, step)`` -> list[str]
* ``decompose_simple(state)`` -> ``HierarchicalPlan`` (heuristic, no LLM)

Conventions:
* Grid is 64x64. Row 0 is top, col 0 is left.
* Movement (verified empirically on ls20 replay Phase B.2):
    ACTION1 = up    -> row -= step (~5)
    ACTION2 = down  -> row += step (~5)
    ACTION3 = left  -> col -= step (~5)
    ACTION4 = right -> col += step (~5)
* Walls are blocked cells the agent has *discovered* (see
  ``LockSmithState.walls_seen``). We can't know walls we haven't
  bumped into, so pathfinding is optimistic.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Iterable


# Action -> (delta_row, delta_col) per step.
ACTION_DELTAS: dict[str, tuple[int, int]] = {
    "ACTION1": (-1, 0),  # up
    "ACTION2": (1, 0),   # down
    "ACTION3": (0, -1),  # left
    "ACTION4": (0, 1),   # right
}

# Default per-action step in cells. From Phase B.2 empirical observation
# on ls20 (player_pos jumps ~5 cells per movement action). Override per
# game class if mechanics differ.
DEFAULT_STEP = 5


@dataclass
class SubGoal:
    """One tactical objective in a hierarchical plan."""

    name: str  # "navigate_to" | "step_on" | "wait" | "reset" | "custom"
    target_pos: tuple[float, float] | None = None  # for navigate / step_on
    target_label: str = ""  # human-readable: "door", "rotator_0", "energy_pill"
    max_actions: int = 20  # safety cap; abandon goal if exceeded


@dataclass
class HierarchicalPlan:
    """An ordered list of sub-goals plus a cursor + per-goal action budget."""

    sub_goals: list[SubGoal] = field(default_factory=list)
    current_idx: int = 0
    actions_in_current: int = 0

    @property
    def current_goal(self) -> SubGoal | None:
        if 0 <= self.current_idx < len(self.sub_goals):
            return self.sub_goals[self.current_idx]
        return None

    @property
    def done(self) -> bool:
        return self.current_idx >= len(self.sub_goals)

    def advance(self) -> None:
        self.current_idx += 1
        self.actions_in_current = 0


def _quantize(pos: tuple[float, float], step: int) -> tuple[int, int]:
    """Round a (row, col) centroid to the nearest step-grid cell."""
    return (int(round(pos[0] / step)) * step,
            int(round(pos[1] / step)) * step)


def astar_pathfind(
    start: tuple[float, float],
    goal: tuple[float, float],
    walls: Iterable[tuple[int, int]] = (),
    grid_size: int = 64,
    step: int = DEFAULT_STEP,
    wall_radius: int = 3,
) -> list[tuple[int, int]]:
    """A* over the coarse step-grid. Returns a list of waypoints from
    quantized start to quantized goal (inclusive on both ends), or [] if
    no path found.

    ``walls`` are cells the agent has already bumped into; we treat
    any cell within ``wall_radius`` as blocked too (player sprite is
    ~4x4, so a wall at (10,10) effectively blocks neighbours).
    """
    wall_set: set[tuple[int, int]] = set()
    for wr, wc in walls:
        for dr in range(-wall_radius, wall_radius + 1):
            for dc in range(-wall_radius, wall_radius + 1):
                wall_set.add((wr + dr, wc + dc))

    s = _quantize(start, step)
    g = _quantize(goal, step)

    def _in_bounds(p: tuple[int, int]) -> bool:
        r, c = p
        return 0 <= r < grid_size and 0 <= c < grid_size

    def _heur(p: tuple[int, int]) -> float:
        return (abs(p[0] - g[0]) + abs(p[1] - g[1])) / step

    open_heap: list[tuple[float, tuple[int, int]]] = [(_heur(s), s)]
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {s: None}
    g_score: dict[tuple[int, int], float] = {s: 0.0}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == g:
            # Reconstruct
            path: list[tuple[int, int]] = []
            node: tuple[int, int] | None = current
            while node is not None:
                path.append(node)
                node = came_from[node]
            return list(reversed(path))

        for dr, dc in ACTION_DELTAS.values():
            nr, nc = current[0] + dr * step, current[1] + dc * step
            neighbour = (nr, nc)
            if not _in_bounds(neighbour):
                continue
            if neighbour in wall_set:
                continue
            tentative = g_score[current] + 1
            if tentative < g_score.get(neighbour, float("inf")):
                came_from[neighbour] = current
                g_score[neighbour] = tentative
                heapq.heappush(open_heap, (tentative + _heur(neighbour), neighbour))

    return []


def pathfind_to_actions(
    start: tuple[float, float],
    goal: tuple[float, float],
    walls: Iterable[tuple[int, int]] = (),
    grid_size: int = 64,
    step: int = DEFAULT_STEP,
) -> list[str]:
    """Find a path and translate it into ARC action names.

    Returns [] if no path exists. Each waypoint transition maps to one
    movement action (ACTION1..4) -- the agent executes one per turn.
    """
    path = astar_pathfind(start, goal, walls, grid_size, step)
    if len(path) < 2:
        return []
    actions: list[str] = []
    for i in range(1, len(path)):
        dr = path[i][0] - path[i - 1][0]
        dc = path[i][1] - path[i - 1][1]
        nrm_dr = (dr // step) if dr else 0
        nrm_dc = (dc // step) if dc else 0
        for name, (adr, adc) in ACTION_DELTAS.items():
            if (adr, adc) == (nrm_dr, nrm_dc):
                actions.append(name)
                break
    return actions


def decompose_simple(
    player_pos: tuple[float, float] | None,
    door_pos: tuple[float, float] | None,
    rotators: list[tuple[float, float, int]],
    level: int,
) -> HierarchicalPlan:
    """Heuristic goal decomposition without LLM.

    Strategy for LockSmith-style games:
      1. If we have a door but no rotator interaction yet, step on the
         NEAREST rotator first (to set key shape/color).
      2. Then navigate to the door.
      3. If the level just changed, restart from the new state.

    Returns an empty plan when we lack enough info (caller falls back
    to a simple explore behaviour).
    """
    plan = HierarchicalPlan()
    if player_pos is None:
        return plan

    if rotators:
        # Find nearest rotator.
        nearest = min(
            rotators,
            key=lambda r: (r[0] - player_pos[0]) ** 2
            + (r[1] - player_pos[1]) ** 2,
        )
        plan.sub_goals.append(SubGoal(
            name="step_on",
            target_pos=(nearest[0], nearest[1]),
            target_label="rotator",
            max_actions=15,
        ))

    if door_pos is not None:
        plan.sub_goals.append(SubGoal(
            name="navigate_to",
            target_pos=door_pos,
            target_label="door",
            max_actions=25,
        ))

    return plan


def goal_reached(
    goal: SubGoal,
    player_pos: tuple[float, float] | None,
    tolerance: int = DEFAULT_STEP,
) -> bool:
    """Has the agent arrived at the sub-goal's target?"""
    if goal.target_pos is None or player_pos is None:
        return False
    dr = abs(goal.target_pos[0] - player_pos[0])
    dc = abs(goal.target_pos[1] - player_pos[1])
    return dr <= tolerance and dc <= tolerance
