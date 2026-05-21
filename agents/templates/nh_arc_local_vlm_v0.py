"""nh_arc_local_vlm_v0 -- offline Qwen2.5-VL-3B vision agent.

Kaggle submission requires offline inference (internet disabled in
notebook env). All CLI-based agents (baseline_v0, vision_v0,
expert_replay_v0) cannot work server-side. This template uses
HuggingFace's Qwen2.5-VL-3B-Instruct loaded locally on the GPU
(~7 GB VRAM in fp16), making per-action multimodal planning possible
without any network call.

Smoke-test results (2026-05-21, RTX 4070 Laptop 8 GB):
* Load: ~6 min first time (HF download cached at ~/.cache/huggingface)
* VRAM peak: 7249 MB (fits 8 GB GPU)
* Inference latency: ~10s/call, stable run-to-run

Plan length is set to 12 actions per call (vs 6 in CLI vision_v0) so
that for an 80-action env we make only ~7 planner calls, not 14 --
keeping total time under the Kaggle 9h budget when scaled to 110 games.

Singleton model: the model loads once at class level and is shared
across game instances spawned by the Swarm.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from arcengine import FrameData, GameAction, GameState

from ..agent import Agent
from . import arc_vision
from .nh_arc_baseline_v0 import GAME_CARDS, _card_for_game

logger = logging.getLogger(__name__)

# Default to 7B int4 (better reasoning than 3B fp16, fits 8 GB VRAM).
# Override via NH_VLM_MODEL_ID env var to test alternatives.
_MODEL_ID = os.environ.get(
    "NH_VLM_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct"
)
_USE_4BIT = os.environ.get("NH_VLM_4BIT", "1") != "0"

_ACTION_PATTERN = re.compile(
    r"\b(RESET|ACTION[1-7])"
    r"(?:"
    r"\s*[\(\s]\s*x\s*=\s*(\d+)\s*[,;\s]\s*y\s*=\s*(\d+)"
    r"|"
    r"\s+(\d+)\s+(\d+)"
    r")?",
    re.IGNORECASE,
)


def _parse_plan(text: str, max_actions: int) -> list[tuple[str, dict]]:
    if not text:
        return []
    out: list[tuple[str, dict]] = []
    for m in _ACTION_PATTERN.finditer(text):
        name = m.group(1).upper()
        data: dict = {}
        if name == "ACTION6":
            x = m.group(2) or m.group(4)
            y = m.group(3) or m.group(5)
            if x and y:
                try:
                    data = {"x": str(int(x)), "y": str(int(y))}
                except ValueError:
                    data = {}
        out.append((name, data))
        if len(out) >= max_actions:
            break
    return out


# Module-level singletons (lazy-loaded on first agent instantiation).
_MODEL: Any = None
_PROCESSOR: Any = None
_DEVICE: str = "cuda:0"


def _ensure_model_loaded() -> tuple[Any, Any]:
    global _MODEL, _PROCESSOR
    if _MODEL is not None and _PROCESSOR is not None:
        return _MODEL, _PROCESSOR

    logger.warning(
        f"[LOCAL VLM] Loading {_MODEL_ID} on {_DEVICE} "
        f"(4bit={_USE_4BIT})..."
    )
    t0 = time.time()

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. nh_arc_local_vlm_v0 requires a GPU. "
            "Run on a machine with NVIDIA GPU + CUDA-enabled PyTorch."
        )

    proc = AutoProcessor.from_pretrained(_MODEL_ID)

    load_kwargs: dict[str, Any] = {"device_map": _DEVICE}
    if _USE_4BIT:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = torch.float16

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        _MODEL_ID,
        **load_kwargs,
    )
    model.requires_grad_(False)
    load_s = time.time() - t0
    vram_mb = torch.cuda.memory_allocated(0) // (1024 ** 2)
    logger.warning(
        f"[LOCAL VLM] Loaded in {load_s:.1f}s, VRAM used: {vram_mb} MB"
    )
    _MODEL = model
    _PROCESSOR = proc
    return model, proc


def _vlm_generate(
    prompt: str,
    png_path: Path,
    max_new_tokens: int = 256,
    max_image_dim: int = 1024,
) -> tuple[str, float]:
    """Run Qwen2.5-VL inference on (prompt + image). Returns (text, latency_s).

    Game frames can be 512x3077 (multi-grid stacked: UI panels + main map),
    which is ~6x more vision tokens than the 512x512 smoke-test PNG and
    causes the 3B model to degenerate into "ACTION1 x 12" enumeration.
    We downsample any side over `max_image_dim` to keep tokens manageable.
    """
    model, processor = _ensure_model_loaded()

    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    img = Image.open(png_path).convert("RGB")
    if max(img.size) > max_image_dim:
        img.thumbnail((max_image_dim, max_image_dim), Image.LANCZOS)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    ).to(_DEVICE)

    t0 = time.time()
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    elapsed = time.time() - t0
    out_ids = generated_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
    return response.strip(), elapsed


class NhArcLocalVlmV0(Agent):
    """Local Qwen2.5-VL-3B vision agent, offline-first."""

    MAX_ACTIONS: int = 80
    PLAN_LENGTH: int = 12  # tuned for Kaggle 9h budget across 110 games
    LOG_DIR: str = "logs"
    FRAMES_DIR: str = "logs/local_vlm_frames"
    VLM_MAX_NEW_TOKENS: int = 256

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        Path(self.LOG_DIR).mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._log_path = (
            Path(self.LOG_DIR) / f"local_vlm_v0_{self.game_id}_{ts}.jsonl"
        )
        self._frames_dir = Path(self.FRAMES_DIR) / self.game_id
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        self._plan_queue: list[tuple[str, dict]] = []
        self._planner_call_count: int = 0
        self._action_history: list[str] = []

    def is_done(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        planner_invoked = False
        latency = 0.0
        plan_size = 0
        parse_ok = True
        response_preview = ""
        png_path_str = ""

        if not self._plan_queue:
            planner_invoked = True
            self._planner_call_count += 1

            png_path = self._frames_dir / f"{self.action_counter:03d}.png"
            try:
                arc_vision.render_frame_to_png(latest_frame.frame, png_path)
                png_path_str = str(png_path)
            except Exception as e:
                logger.warning(f"render failed at step {self.action_counter}: {e}")
                self._plan_queue = [("ACTION5", {})]
                parse_ok = False

            if not self._plan_queue:
                prompt = self._build_planning_prompt(latest_frame)
                try:
                    response, latency = _vlm_generate(
                        prompt, png_path,
                        max_new_tokens=self.VLM_MAX_NEW_TOKENS,
                    )
                except Exception as e:
                    logger.error(f"[LOCAL VLM] inference failed: {e}")
                    response = ""
                    latency = 0.0

                response_preview = response[:300]
                self._plan_queue = _parse_plan(response, self.PLAN_LENGTH)
                plan_size = len(self._plan_queue)
                parse_ok = plan_size > 0
                if not self._plan_queue:
                    self._plan_queue = [("ACTION5", {})]

        name, data = self._plan_queue.pop(0)
        try:
            action = GameAction.from_name(name)
        except Exception:
            action = GameAction.ACTION5
        if data:
            action.set_data({**data, "game_id": self.game_id})
        else:
            action.set_data({"game_id": self.game_id})

        self._action_history.append(action.name)

        self._log_decision(
            planner_invoked=planner_invoked,
            latency_seconds=latency,
            png_path=png_path_str,
            plan_size=plan_size,
            parse_ok=parse_ok,
            action_name=action.name,
            queue_depth_after=len(self._plan_queue),
            planner_call_count=self._planner_call_count,
            response_preview=response_preview,
        )
        return action

    def _build_planning_prompt(self, latest_frame: FrameData) -> str:
        avail_names: list[str] = []
        for aid in latest_frame.available_actions or []:
            try:
                avail_names.append(GameAction.from_id(aid).name)
            except Exception:
                avail_names.append(f"ACTION_ID_{aid}")
        avail = ", ".join(avail_names) or "RESET, ACTION1..7"

        card = _card_for_game(self.game_id)
        card_block = f"\n# GAME-SPECIFIC NOTES\n{card}\n" if card else ""

        history = self._action_history[-12:]
        history_block = (
            "Recent actions taken: " + ", ".join(history)
            if history
            else "Recent actions: (none yet)"
        )

        # Symbolic ground truth: real (row, col) positions extracted from
        # the frame array. Pre-trained VLM hallucinates coords from the
        # image; we hand them the actual numbers so reasoning is grounded.
        symbolic_block = ""
        try:
            cmap = arc_vision.color_map_for_game(self.game_id)
            summary = arc_vision.to_symbolic_summary(
                latest_frame.frame,
                color_name_map=cmap,
                max_objects_per_type=4,
                grid_idx=0,
            )
            if summary:
                lines = ["# OBSERVED OBJECTS (real coords, NOT hallucinated)"]
                lines.append("Grid is 64x64, row 0 = top, col 0 = left.")
                lines.append("Format: name -> list of {at: [row, col], size}")
                for name, val in summary.items():
                    if isinstance(val, dict) and "count" in val:
                        lines.append(f"  {name}: count={val['count']} area={val['total_area']}")
                    elif isinstance(val, list):
                        items = ", ".join(
                            f"at[{o['at'][0]:.1f},{o['at'][1]:.1f}] size={o['size']}"
                            for o in val[:4]
                        )
                        lines.append(f"  {name}: {items}")
                symbolic_block = "\n" + "\n".join(lines) + "\n"
        except Exception as e:
            logger.debug(f"symbolic_summary failed: {e}")

        return (
            "# ROLE\n"
            "You see one frame of an ARC-AGI-3 game. Plan the next "
            f"{self.PLAN_LENGTH} actions to make progress toward winning a\n"
            "level while avoiding GAME_OVER.\n"
            f"{card_block}\n"
            "# CURRENT STATE\n"
            f"Game state: {latest_frame.state.name}\n"
            f"Levels completed so far: {latest_frame.levels_completed}\n"
            f"Action counter: {self.action_counter}/{self.MAX_ACTIONS}\n"
            f"Available actions THIS FRAME: {avail}\n"
            f"{history_block}\n"
            f"{symbolic_block}"
            "\n"
            "# ACTION SEMANTICS (for LockSmith / movement-based games)\n"
            "* ACTION1 = move UP    (player row goes DOWN by 1, since row 0 = top)\n"
            "* ACTION2 = move DOWN  (row goes UP by 1)\n"
            "* ACTION3 = move LEFT  (column goes DOWN by 1)\n"
            "* ACTION4 = move RIGHT (column goes UP by 1)\n"
            "* ACTION5, ACTION7 = take no arguments, game-specific (often no-op in LockSmith)\n"
            "* ACTION6 ONLY = needs `x=NN y=NN` (click coordinate). All others take NO arguments.\n"
            "* RESET = restart current level / game.\n"
            "\n"
            "# REASONING -> PLAN\n"
            "Use the REAL [row, col] coords above (player_eyes, exit_door,\n"
            "rotators, etc.) -- DO NOT make up coordinates.\n"
            "\n"
            "Step 1: ONE line, plain text:\n"
            "  delta_row = target_row - player_row\n"
            "  delta_col = target_col - player_col\n"
            "If delta_row < 0 -> need ACTION1 (up) abs(delta_row) times.\n"
            "If delta_row > 0 -> need ACTION2 (down) abs(delta_row) times.\n"
            "If delta_col < 0 -> need ACTION3 (left) abs(delta_col) times.\n"
            "If delta_col > 0 -> need ACTION4 (right) abs(delta_col) times.\n"
            "\n"
            "Step 2: On a NEW LINE write `PLAN:` then list EXACTLY "
            f"{self.PLAN_LENGTH} actions to close those deltas.\n"
            "Mix directions if both row and col need to change.\n"
            "\n"
            "Do NOT append `x=NN y=NN` to ACTION1/2/3/4/5/7 (only ACTION6 needs it).\n"
            "Do NOT just enumerate ACTION1..ACTION4 in order -- compute deltas.\n"
        )

    def _log_decision(
        self,
        planner_invoked: bool,
        latency_seconds: float,
        png_path: str,
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
            "latency_seconds": round(latency_seconds, 3),
            "png_path": png_path if planner_invoked else "",
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
