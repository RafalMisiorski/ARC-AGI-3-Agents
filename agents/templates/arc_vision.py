"""arc_vision.py -- pure-Python perception layer for ARC-AGI-3 grids.

Phase A.1 of the long-term plan. Pre-processes 64x64 hex grids into:
1. PNG image (for multimodal LLM consumption or human inspection)
2. Symbolic object dictionary (for prompt injection, replaces hex spam)

Replaces the agent's hex-pretty-print prompt with structured facts like
``{"player": {"at": [30,32], "size": 16}, "walls": [...], "rotator_marker": [...]}``.

Cost per call: zero LLM tokens. Pure numpy + scikit-image. Deterministic.

Public API:
* ``render_frame_to_png(frame, output_path)`` -> Path
* ``detect_objects(frame, color_name_map=None, min_size=1)`` -> dict
* ``to_symbolic_summary(frame, color_name_map=None, max_objects=5)`` -> dict
* ``LOCKSMITH_COLOR_NAMES`` -- per-game-class color mapping for ls20

Add new game classes by extending the COLOR_MAP_BY_GAME registry below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage import measure

# ARC standard 16-color palette (RGB). Approximate values matching
# typical ARC visualizations.
PALETTE: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),         # black (background)
    1: (30, 147, 255),    # blue
    2: (255, 65, 81),     # red
    3: (78, 204, 75),     # green
    4: (255, 220, 0),     # yellow
    5: (153, 153, 153),   # gray
    6: (245, 53, 178),    # magenta/pink
    7: (255, 130, 50),    # orange
    8: (170, 220, 255),   # sky blue
    9: (140, 80, 30),     # brown
    10: (90, 90, 90),     # dark gray
    11: (255, 150, 180),  # pink
    12: (200, 100, 100),  # rust
    13: (100, 100, 200),  # purple
    14: (200, 200, 100),  # khaki
    15: (100, 200, 200),  # teal
}

# Per-game-class semantic mapping.
# Note: original mapping was lifted from agents/templates/llm_agents.py's
# GuidedLLM prompt, but those colors are WRONG for the current ls20
# environment (frame[0] dominant colors: 0x04 yellow bg, 0x03 green floor,
# 0x05 gray walls, 0x0b pink door, 0x09 brown rotators).
# Calibrated empirically from operator's 621-action replay on 2026-05-17.
LOCKSMITH_COLOR_NAMES: dict[int, str] = {
    0:  "player_eyes",    # black cells inside 4x4 player sprite
    3:  "floor",          # green; most frequently entered/exited by player
    4:  "play_area_bg",   # yellow; player sprite "transparent" + arena bg
    5:  "wall",           # gray; static large regions blocking movement
    9:  "rotator_marker", # brown; player steps on -> key rotates
    11: "door_border",    # pink; exit door frame
    12: "key_indicator",  # small cells next to player showing current key
}

# Registry indexed by env_id prefix (3-4 chars). Add as we discover.
COLOR_MAP_BY_GAME: dict[str, dict[int, str]] = {
    "ls20": LOCKSMITH_COLOR_NAMES,
}


def color_map_for_game(game_id: str) -> dict[int, str]:
    """Lookup semantic color map for a game_id prefix; empty dict if unknown."""
    if not game_id:
        return {}
    prefix = game_id.split("-", 1)[0]
    return COLOR_MAP_BY_GAME.get(prefix, {})


def render_frame_to_png(
    frame: list[list[list[int]]],
    output_path: Path,
    scale: int = 8,
) -> Path:
    """Render a 3D frame (list of 2D grids) as a PNG.

    Each cell becomes ``scale x scale`` pixels (default 8x8 -> 512x512 image
    for one 64x64 sub-grid). Multi-grid frames stack vertically with a 1px
    black separator.
    """
    if not frame:
        raise ValueError("Empty frame")

    scaled_grids: list[np.ndarray] = []
    for grid in frame:
        if not grid:
            continue
        g = np.array(grid, dtype=int)
        h, w = g.shape
        arr = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)
        for r in range(h):
            for c in range(w):
                color = PALETTE.get(int(g[r, c]), (255, 255, 255))
                arr[r * scale:(r + 1) * scale, c * scale:(c + 1) * scale] = color
        scaled_grids.append(arr)

    if not scaled_grids:
        raise ValueError("No valid sub-grids in frame")

    # Stack vertically with 1px separator.
    total_h = sum(a.shape[0] for a in scaled_grids) + (len(scaled_grids) - 1)
    max_w = max(a.shape[1] for a in scaled_grids)
    combined = np.zeros((total_h, max_w, 3), dtype=np.uint8)
    row_off = 0
    for idx, a in enumerate(scaled_grids):
        if idx > 0:
            row_off += 1
        combined[row_off:row_off + a.shape[0], :a.shape[1]] = a
        row_off += a.shape[0]

    img = Image.fromarray(combined)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return output_path


def detect_objects(
    frame: list[list[list[int]]],
    color_name_map: dict[int, str] | None = None,
    min_size: int = 1,
    grid_idx: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Connected-components per unique cell value.

    Returns ``{"object_name": [{"bbox": [r0,c0,r1,c1], "area": int,
    "centroid": [r,c]}, ...]}``. Cells with unknown colors are labelled
    ``color_<hex>`` so nothing is silently lost.

    Args:
        frame: 3D list (sub-grids of cells).
        color_name_map: optional ``{value: name}``. Defaults to LockSmith.
        min_size: ignore regions smaller than this (in cells).
        grid_idx: which sub-grid to analyse.
    """
    name_map = color_name_map if color_name_map is not None else LOCKSMITH_COLOR_NAMES
    out: dict[str, list[dict[str, Any]]] = {}

    if not frame or grid_idx >= len(frame):
        return out

    grid = np.array(frame[grid_idx], dtype=int)
    unique_vals = np.unique(grid)

    for val in unique_vals:
        mask = (grid == val)
        labels = measure.label(mask, connectivity=2)  # 8-connectivity
        regions = measure.regionprops(labels)

        objects: list[dict[str, Any]] = []
        for r in regions:
            if r.area < min_size:
                continue
            objects.append({
                "bbox": [int(x) for x in r.bbox],
                "area": int(r.area),
                "centroid": [round(float(r.centroid[0]), 1),
                             round(float(r.centroid[1]), 1)],
            })

        if objects:
            label = name_map.get(int(val), f"color_{int(val):02x}")
            if label in out:
                out[label].extend(objects)
            else:
                out[label] = objects

    return out


def calibrate_from_replay(
    replay_path: Path,
    max_records: int = 50,
) -> dict[str, Any]:
    """Analyze a replay JSONL to infer color->object mapping.

    Reusable for new game classes: drop a manual replay into the file,
    run this, get a starting color_name_map proposal.

    Heuristic identification:
      * **player**: color whose cell-count drops most when player moves
        (cells where player WAS get replaced by underlying floor)
      * **floor**: color most frequently revealed by player motion
        (cells player vacates default to this)
      * **wall**: color of large static regions never entered
        (frame[0] color counts, excluding player + floor)
      * **rotator_marker / door_border**: small consistent regions
        (manually reviewed; we just surface counts as hints)

    Returns ``{"player": int, "floor": int, "wall": int, "other_counts":
    {color: count}, "evidence": {...}}``.
    """
    import collections
    import json as _json

    records: list[dict] = []
    with replay_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(_json.loads(line))
            except _json.JSONDecodeError:
                continue
    if not records:
        return {"error": "empty replay"}

    target_votes: collections.Counter = collections.Counter()
    source_votes: collections.Counter = collections.Counter()
    for i in range(min(max_records, len(records) - 1)):
        d0 = records[i].get("data", {})
        d1 = records[i + 1].get("data", {})
        f0 = d0.get("frame", [[[]]])
        f1 = d1.get("frame", [[[]]])
        if not f0 or not f1:
            continue
        a = d1.get("action_input", {}).get("id", "")
        if a not in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
            continue
        arr0 = np.array(f0[0])
        arr1 = np.array(f1[0])
        if arr0.shape != arr1.shape:
            continue
        diff_mask = (arr0 != arr1)
        for c in arr1[diff_mask]:
            target_votes[int(c)] += 1
        for c in arr0[diff_mask]:
            source_votes[int(c)] += 1

    # Frame[0] palette distribution
    frame0 = np.array(records[0]["data"]["frame"][0])
    color_counts = collections.Counter(int(c) for c in frame0.flatten())

    # Inference:
    # - floor: most cells gain this color (player moves away -> reveals floor)
    floor_color = target_votes.most_common(1)[0][0] if target_votes else None
    # - player: color where source > target (player cells become floor when player leaves)
    asymmetry = {
        c: source_votes[c] - target_votes[c]
        for c in set(source_votes) | set(target_votes)
    }
    player_candidates = sorted(asymmetry.items(), key=lambda x: -x[1])
    player_color = (
        player_candidates[0][0]
        if player_candidates and player_candidates[0][1] > 0
        else None
    )
    # - wall: large frame[0] regions that AREN'T player or floor
    wall_candidates = [
        (c, n)
        for c, n in color_counts.most_common()
        if c not in (player_color, floor_color)
    ]
    wall_color = wall_candidates[0][0] if wall_candidates else None

    return {
        "player": player_color,
        "floor": floor_color,
        "wall": wall_color,
        "target_votes": dict(target_votes.most_common(8)),
        "source_votes": dict(source_votes.most_common(8)),
        "frame0_counts": dict(color_counts.most_common(8)),
    }


def to_symbolic_summary(
    frame: list[list[list[int]]],
    color_name_map: dict[int, str] | None = None,
    max_objects_per_type: int = 5,
    grid_idx: int = 0,
) -> dict[str, Any]:
    """Compact summary suitable for LLM prompt injection.

    Replaces 4096-cell hex spam with a structured object list:
    ``{"player": [{"at": [30, 32], "size": 16}], "walls": [...], ...}``.

    Floor / background classes get a count instead of per-region lists
    to keep the output token-efficient.
    """
    objs = detect_objects(frame, color_name_map, grid_idx=grid_idx)
    summary: dict[str, Any] = {}
    for name, instances in objs.items():
        if name in ("floor", "color_00"):
            summary[name] = {
                "count": len(instances),
                "total_area": sum(o["area"] for o in instances),
            }
            continue
        sorted_insts = sorted(instances, key=lambda o: -o["area"])[:max_objects_per_type]
        summary[name] = [
            {"at": o["centroid"], "size": o["area"]} for o in sorted_insts
        ]
    return summary


def describe_with_gemini(
    png_path: Path,
    prompt: str | None = None,
    timeout: int = 90,
) -> str:
    """Send a frame PNG to Gemini CLI for multimodal description.

    Used as a fallback for game classes without a calibrated color_map
    (e.g. ar25, cn04). Free via Gemini CLI subscription.

    Returns empty string on failure (caller falls back to hex). Latency
    is ~30-60s per call -- callers should cache aggressively. Typical
    usage: ONE call per game (initial briefing), not per action.
    """
    import os as _os
    import shutil as _shutil
    import subprocess as _subprocess

    cmd_path = _shutil.which("gemini") or _shutil.which("gemini.cmd")
    if not cmd_path:
        return ""
    if not png_path.is_file():
        return ""

    if prompt is None:
        prompt = (
            "Look at the image and describe the game state in 6-10 short lines. "
            "Focus on: (1) player position (likely a small distinct sprite); "
            "(2) walls or obstacles; (3) any goal/door/exit; (4) special "
            "objects (rotators, items, indicators); (5) overall layout. "
            "Use approximate [row, col] grid coordinates where useful "
            "(grid is 64x64, row 0 = top, col 0 = left)."
        )

    # Gemini CLI references files in prompts via @path syntax. The path
    # MUST be absolute -- a relative path would be resolved against
    # gemini's cwd, which differs from ours.
    abs_path = png_path.resolve().as_posix()
    full_prompt = f"{prompt}\n\nImage: @{abs_path}"

    env = _os.environ.copy()
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        env.pop(k, None)

    try:
        proc = _subprocess.run(
            [cmd_path, "-p", full_prompt, "-m", "gemini-2.5-flash"],
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
    except _subprocess.TimeoutExpired:
        return ""
    except Exception:
        return ""

    text = (proc.stdout or "").strip()
    # Strip CLI noise lines.
    lines = [
        line for line in text.split("\n")
        if line.strip()
        and not line.startswith("Loaded cached credentials")
        and not line.startswith("Data collection")
    ]
    return "\n".join(lines).strip()
