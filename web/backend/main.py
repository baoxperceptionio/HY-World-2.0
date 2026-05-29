import asyncio
import base64
import hashlib
import json
import logging
import math
import mimetypes
import os
import signal
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from openai import OpenAI
from PIL import Image, UnidentifiedImageError


logger = logging.getLogger(__name__)
ROOT_DIR = Path(os.environ.get("HYWORLD_ROOT", Path(__file__).resolve().parents[2]))
WORLDGEN_DIR = ROOT_DIR / "hyworld2" / "worldgen"
OUTPUT_ROOT = ROOT_DIR / "outputs" / "web_runs"
JOBS_LOG = OUTPUT_ROOT / "jobs.jsonl"
TARGET_SIZE = (1920, 960)
DEFAULT_SPLIT_VIEW_NUM = 4
DEFAULT_TRAJECTORY_MODES = ["forward", "left-translation", "right-translation"]
DEFAULT_APPLY_NAV_TRAJ = False
MAX_TRAJECTORY_MODES = 8
DEFAULT_GS_MAX_STEPS = 8000
MIN_GS_MAX_STEPS = 100
MAX_GS_MAX_STEPS = 50000
DEFAULT_WORLD_NAV_ATTEMPTS = 3
MIN_WORLD_NAV_ATTEMPTS = 1
MAX_WORLD_NAV_ATTEMPTS = 20
AUTO_PROMPT_FALLBACK = (
    "A cinematic camera trajectory through the panoramic scene, preserving the visible "
    "architecture, terrain, lighting, materials, objects, and overall visual style."
)
AUTO_PROMPT_IMAGE_SIZE = (768, 384)
GENERIC_PROMPT_MARKERS = (
    "panoramic scene",
    "visible architecture, terrain, lighting, materials, objects",
    "architecture, terrain, lighting, materials, objects",
    "overall visual style",
    "major objects",
)
LEGACY_PROMPT_PLACEHOLDERS = {
    "uploaded panorama",
    "an ancient chinese palace",
    "a chinese palace",
}

TRAJECTORY_MODE_LABELS = {
    "right-rotation": ("right rotation", "Camera rotates right from this split view."),
    "left-rotation": ("left rotation", "Camera rotates left from this split view."),
    "up-right-aerial": ("up-right aerial", "Camera pitches upward and rotates toward the right."),
    "up-rotation": ("up rotation", "Camera pitches upward from this split view."),
    "forward": ("forward translation", "Camera translates forward from this split view."),
    "backward": ("backward translation", "Camera translates backward from this split view."),
    "right-translation": ("right translation", "Camera translates to the right from this split view."),
    "left-translation": ("left translation", "Camera translates to the left from this split view."),
}

ARTIFACTS = {
    "panorama.png": ("panorama.png", "image/png"),
    "point_cloud_7999.spz": ("gs_result/ply/point_cloud_7999.spz", "application/octet-stream"),
    "point_cloud_7999.ply": ("gs_result/ply/point_cloud_7999.ply", "application/octet-stream"),
    "point_cloud_7999_playcanvas.ply": ("gs_result/ply/point_cloud_7999_playcanvas.ply", "application/octet-stream"),
    "ckpt_7999_rank0.pt": ("gs_result/ckpts/ckpt_7999_rank0.pt", "application/octet-stream"),
    "pipeline.log": ("pipeline.log", "text/plain"),
    "logs": ("pipeline.log", "text/plain"),
}

PREVIEW_SPECS = [
    ("input", "Uploaded panorama", "Image after direct 1920x960 resize", ["panorama.png"]),
    ("trajectory", "Sky mask", "Automatically inferred sky mask", ["render_results/sky_mask.png"]),
    (
        "video",
        "Generation sample",
        "Generated image bank sample",
        [
            "render_results/generation_bank_worldstereo-memory-dmd/world_mirror_data/images/view0-traj0-0003.png",
            "render_results/generation_bank_worldstereo-memory-dmd/world_mirror_data/images/pano-0000.png",
        ],
    ),
    ("gs-data", "Pano bank image", "Rendered panorama bank sample", ["render_results/pano_bank/images/0000.png"]),
    ("gs-data", "Pano depth", "Panorama bank depth sample", ["render_results/pano_bank/depths/0000.png"]),
    ("gs-data", "GS image sample", "Image copied into Gaussian Splat training data", ["gs_data/images/view0-traj0_000000.png", "gs_data/images/panorama_L00_A0000.png"]),
    ("gs-data", "GS depth sample", "Depth copied into Gaussian Splat training data", ["gs_data/depths/view0-traj0_000003.png", "gs_data/depths/panorama_L00_A0000.png"]),
    ("gs-data", "GS normal sample", "Normal map copied into Gaussian Splat training data", ["gs_data/normals/view0-traj0_000003.png", "gs_data/normals/panorama_L00_A0000.png"]),
    ("training", "Validation render", "Final trainer validation render", ["gs_result/renders/val_step7999_0000.png"]),
]

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_task_params(
    split_view_num: int,
    trajectory_modes: list[str],
    indoor: bool,
    gs_max_steps: int,
    apply_nav_traj: bool = DEFAULT_APPLY_NAV_TRAJ,
    world_nav_attempts: int | None = DEFAULT_WORLD_NAV_ATTEMPTS,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "split_view_num": max(1, min(int(split_view_num), 8)),
        "trajectory_modes": sanitize_trajectory_modes(trajectory_modes),
        "indoor": bool(indoor),
        "gs_max_steps": sanitize_gs_max_steps(gs_max_steps),
        "apply_nav_traj": bool(apply_nav_traj),
    }
    selected_world_nav_attempts = sanitize_world_nav_attempts(world_nav_attempts)
    if apply_nav_traj:
        params["world_nav_attempts"] = selected_world_nav_attempts
    return params


def task_hash(
    data: bytes,
    split_view_num: int,
    trajectory_modes: list[str],
    indoor: bool,
    gs_max_steps: int,
    apply_nav_traj: bool = DEFAULT_APPLY_NAV_TRAJ,
    world_nav_attempts: int | None = DEFAULT_WORLD_NAV_ATTEMPTS,
) -> tuple[str, str, dict[str, Any]]:
    image_sha256 = hashlib.sha256(data).hexdigest()
    params = canonical_task_params(split_view_num, trajectory_modes, indoor, gs_max_steps, apply_nav_traj, world_nav_attempts)
    payload = {
        "version": 1,
        "image_sha256": image_sha256,
        "params": params,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return digest, image_sha256, params


def make_job_id(
    data: bytes,
    split_view_num: int,
    trajectory_modes: list[str],
    indoor: bool,
    gs_max_steps: int,
    apply_nav_traj: bool = DEFAULT_APPLY_NAV_TRAJ,
    world_nav_attempts: int | None = DEFAULT_WORLD_NAV_ATTEMPTS,
) -> str:
    digest, _, _ = task_hash(data, split_view_num, trajectory_modes, indoor, gs_max_steps, apply_nav_traj, world_nav_attempts)
    return f"task_{digest[:20]}"


def safe_original_name(filename: str | None) -> str:
    suffix = Path(filename or "upload").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
        suffix = ".img"
    return f"original_upload{suffix}"


def uploaded_image_data(run_dir: Path) -> bytes:
    candidates = sorted(
        path
        for path in run_dir.glob("original_upload.*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".img"}
    )
    panorama_path = run_dir / "panorama.png"
    if panorama_path.exists() and panorama_path.is_file():
        candidates.append(panorama_path)
    if not candidates:
        raise FileNotFoundError("Uploaded image is not available for prompt synthesis.")
    return candidates[0].read_bytes()


@dataclass(frozen=True)
class PromptSynthesis:
    prompt: str
    source: str
    error: str | None = None


def compact_prompt_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").replace("\r", " ").split()).strip(" \"'")


def normalize_generated_prompt(text: str) -> str:
    prompt = compact_prompt_text(text)
    if not prompt:
        return AUTO_PROMPT_FALLBACK
    return prompt[:700]


def is_generic_generated_prompt(prompt: str) -> bool:
    normalized = compact_prompt_text(prompt).lower()
    if not normalized:
        return True
    if normalized == AUTO_PROMPT_FALLBACK.lower():
        return True
    return any(marker in normalized for marker in GENERIC_PROMPT_MARKERS)


def is_placeholder_prompt(prompt: str) -> bool:
    normalized = compact_prompt_text(prompt).lower()
    return normalized in LEGACY_PROMPT_PLACEHOLDERS or is_generic_generated_prompt(prompt)


def fallback_prompt(source: str, error: str) -> PromptSynthesis:
    return PromptSynthesis(prompt=AUTO_PROMPT_FALLBACK, source=source, error=error)


def synthesize_prompt_details(data: bytes) -> PromptSynthesis:
    if os.environ.get("HYWORLD_AUTO_PROMPT", "1").strip().lower() in {"0", "false", "no", "off"}:
        return fallback_prompt("disabled", "Automatic LLM prompt generation is disabled.")

    try:
        with Image.open(BytesIO(data)) as img:
            img.load()
            preview = img.convert("RGB")
    except (UnidentifiedImageError, OSError):
        return fallback_prompt("invalid_image", "Upload must be a readable image file.")

    preview.thumbnail(AUTO_PROMPT_IMAGE_SIZE, Image.Resampling.LANCZOS)
    buffer = BytesIO()
    preview.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    base_url = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1").strip()
    api_key = os.environ.get("LLM_API_KEY", "EMPTY")
    model = os.environ.get("LLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct").strip() or "Qwen/Qwen3-VL-8B-Instruct"
    timeout = float(os.environ.get("HYWORLD_AUTO_PROMPT_TIMEOUT", "45"))

    try:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a visual scene describer for image-conditioned 3D world generation. "
                        "Write only what is directly visible in the image. Mention concrete objects, spatial layout, "
                        "environment type, colors, lighting, material cues, and style. Never use generic category lists."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Describe this panorama as one compact English prompt, 35 to 70 words. "
                                "It must include specific visible scene details such as what occupies the foreground, "
                                "middle distance, background, and lighting. Do not say uploaded image, panorama, "
                                "architecture/terrain/materials/objects as generic placeholders, or camera trajectory. "
                                "If you cannot inspect the image, output exactly CANNOT_DESCRIBE."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                    ],
                },
            ],
            max_tokens=160,
            temperature=0.15,
        )
        raw_prompt = compact_prompt_text(response.choices[0].message.content or "")
        if not raw_prompt or raw_prompt == "CANNOT_DESCRIBE":
            return fallback_prompt("empty_response", "LLM did not return a scene description.")
        prompt = normalize_generated_prompt(raw_prompt)
        if is_generic_generated_prompt(prompt):
            return fallback_prompt("generic_response", "LLM returned a generic prompt instead of visible scene details.")
        return PromptSynthesis(prompt=prompt, source="llm")
    except Exception as exc:
        message = f"LLM request failed: {exc}"
        logger.warning("Automatic prompt synthesis failed; using fallback prompt: %s", exc)
        return fallback_prompt("error", message)


def synthesize_prompt_from_image(data: bytes) -> str:
    return synthesize_prompt_details(data).prompt


def prepare_job_files(run_dir: Path, filename: str | None, data: bytes, prompt: str, indoor: bool = False) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    if not data:
        raise ValueError("Uploaded file is empty.")

    original_path = run_dir / safe_original_name(filename)
    original_path.write_bytes(data)

    try:
        with Image.open(BytesIO(data)) as img:
            img.load()
            panorama = img.convert("RGB").resize(TARGET_SIZE, Image.Resampling.LANCZOS)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Upload must be a readable image file.") from exc

    panorama.save(run_dir / "panorama.png")
    (run_dir / "meta_info.json").write_text(
        json.dumps({"scene_type": "indoor" if indoor else "outdoor", "prompt": prompt}, indent=2),
        encoding="utf-8",
    )
    (run_dir / "pipeline.log").touch()


def update_scene_type(run_dir: Path, prompt: str, indoor: bool) -> None:
    meta_path = run_dir / "meta_info.json"
    meta: dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["scene_type"] = "indoor" if indoor else "outdoor"
    meta["prompt"] = prompt
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def gs_final_step(gs_max_steps: int) -> int:
    return max(0, int(gs_max_steps) - 1)


def artifact_path(run_dir: Path, name: str, gs_max_steps: int = DEFAULT_GS_MAX_STEPS) -> tuple[Path, str]:
    if name not in ARTIFACTS:
        raise KeyError(name)
    rel_path, media_type = ARTIFACTS[name]
    final_step = gs_final_step(gs_max_steps)
    if name == "point_cloud_7999.spz":
        rel_path = f"gs_result/ply/point_cloud_{final_step}.spz"
    elif name == "point_cloud_7999.ply":
        rel_path = f"gs_result/ply/point_cloud_{final_step}.ply"
    elif name == "point_cloud_7999_playcanvas.ply":
        rel_path = f"gs_result/ply/point_cloud_{final_step}_playcanvas.ply"
    elif name == "ckpt_7999_rank0.pt":
        rel_path = f"gs_result/ckpts/ckpt_{final_step}_rank0.pt"
    return run_dir / rel_path, media_type


def safe_run_file(run_dir: Path, rel_path: str) -> Path:
    root = run_dir.resolve()
    candidate = (run_dir / rel_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("Path is outside this job run.")
    return candidate


def make_preview_item(
    job_id: str,
    run_dir: Path,
    item_id: str,
    stage: str,
    title: str,
    description: str,
    candidates: list[str],
    group_title: str | None = None,
    group_order: int | None = None,
) -> dict[str, Any]:
    selected: Path | None = None
    selected_rel: str | None = None
    for rel_path in candidates:
        candidate = safe_run_file(run_dir, rel_path)
        if candidate.exists() and candidate.is_file():
            selected = candidate
            selected_rel = rel_path
            break

    media_type = mimetypes.guess_type(selected.name)[0] if selected else None
    kind = "video" if media_type and media_type.startswith("video/") else "image"
    item = {
        "id": item_id,
        "stage": stage,
        "title": title,
        "description": description,
        "available": selected is not None,
        "kind": kind,
        "media_type": media_type,
        "path": selected_rel,
        "url": None,
        "updated_at": None,
        "group_title": group_title,
        "group_order": group_order,
    }
    if selected and selected_rel:
        stat = selected.stat()
        item["url"] = f"/api/jobs/{job_id}/preview-files/{quote(selected_rel, safe='/')}?v={stat.st_mtime_ns}"
        item["updated_at"] = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
    return item


def sanitize_trajectory_modes(value: str | list[str] | None) -> list[str]:
    if value is None:
        return DEFAULT_TRAJECTORY_MODES.copy()
    modes = value.split(",") if isinstance(value, str) else value
    cleaned: list[str] = []
    for mode in modes:
        mode = str(mode).strip()
        if not mode:
            continue
        if mode not in TRAJECTORY_MODE_LABELS:
            raise ValueError(f"Unknown trajectory mode: {mode}")
        cleaned.append(mode)
    if not cleaned:
        raise ValueError("At least one trajectory mode is required.")
    if len(cleaned) > MAX_TRAJECTORY_MODES:
        raise ValueError(f"At most {MAX_TRAJECTORY_MODES} trajectory modes are supported.")
    return cleaned


def sanitize_gs_max_steps(value: int) -> int:
    steps = int(value)
    if steps < MIN_GS_MAX_STEPS or steps > MAX_GS_MAX_STEPS:
        raise ValueError(f"gs_max_steps must be between {MIN_GS_MAX_STEPS} and {MAX_GS_MAX_STEPS}.")
    return steps


def sanitize_world_nav_attempts(value: int | str | None) -> int:
    if value is None or value == "":
        return DEFAULT_WORLD_NAV_ATTEMPTS
    attempts = int(value)
    if attempts < MIN_WORLD_NAV_ATTEMPTS or attempts > MAX_WORLD_NAV_ATTEMPTS:
        raise ValueError(f"world_nav_attempts must be between {MIN_WORLD_NAV_ATTEMPTS} and {MAX_WORLD_NAV_ATTEMPTS}.")
    return attempts


def split_view_preview_specs(split_view_num: int, trajectory_modes: list[str] | None = None) -> list[tuple[str, str, str, str, list[str]]]:
    specs: list[tuple[str, str, str, str, list[str]]] = []
    split_count = max(1, min(int(split_view_num), 8))
    modes = sanitize_trajectory_modes(trajectory_modes)
    for view_i in range(split_count):
        azimuth = round(view_i * 360.0 / split_count)
        specs.extend(
            [
                (
                    f"split-view-{view_i}-start",
                    "split-view",
                    f"View {view_i} crop",
                    f"Perspective crop from the panorama at about {azimuth} degrees; this is the start frame for this scene split.",
                    [f"render_results/view{view_i}/start_frame.png"],
                ),
                (
                    f"split-view-{view_i}-mask",
                    "split-view",
                    f"View {view_i} point mask",
                    "Visible projected panorama points for this split; sparse or broken areas here often become weak GS regions.",
                    [f"render_results/view{view_i}/point_mask.png"],
                ),
            ]
        )
        for traj_i, mode in enumerate(modes):
            label, caption = TRAJECTORY_MODE_LABELS.get(mode, (f"traj {traj_i}", "Generated camera trajectory."))
            specs.extend(
                [
                    (
                        f"view-{view_i}-traj-{traj_i}-render",
                        "trajectory",
                        f"View {view_i} {label} render",
                        f"Point-cloud render before video generation. {caption} Holes here show what WorldStereo must hallucinate.",
                        [f"render_results/view{view_i}/traj{traj_i}/render.mp4"],
                    ),
                    (
                        f"view-{view_i}-traj-{traj_i}-mask",
                        "trajectory",
                        f"View {view_i} {label} mask",
                        "Mask for the point-cloud trajectory render; bright regions are supported by projected source points.",
                        [f"render_results/view{view_i}/traj{traj_i}/render_mask.mp4"],
                    ),
                    (
                        f"view-{view_i}-traj-{traj_i}-generated",
                        "video",
                        f"View {view_i} {label} generated",
                        "WorldStereo output used as synthetic training views for Gaussian Splatting.",
                        [f"render_results/view{view_i}/traj{traj_i}/worldstereo-memory-dmd_result.mp4"],
                    ),
                ]
            )
    return specs


def _world_nav_sort_key(traj_dir: Path) -> tuple[int, str, int]:
    parent = traj_dir.parent.name
    family_order = 99
    for index, prefix in enumerate(("target", "wonder", "reconstruct")):
        if parent.startswith(prefix):
            family_order = index
            break
    try:
        traj_index = int(traj_dir.name.replace("traj", "", 1))
    except ValueError:
        traj_index = 0
    return family_order, parent, traj_index


def _world_nav_group_title(folder_name: str) -> str:
    if folder_name.startswith("target_"):
        label = folder_name[len("target_"):].replace("_", " ").strip()
        return f"WorldNav target {label}"
    if folder_name.startswith("wonder_"):
        label = folder_name[len("wonder_"):].replace("_", " ").strip()
        return f"WorldNav exploration {label}"
    if folder_name.startswith("reconstruct_"):
        label = folder_name[len("reconstruct_"):].replace("_", " ").strip()
        return f"WorldNav reconstruction {label}"
    return f"WorldNav {folder_name.replace('_', ' ')}"


def world_nav_preview_specs(run_dir: Path, apply_nav_traj: bool = DEFAULT_APPLY_NAV_TRAJ) -> list[tuple[str, str, str, str, list[str], str, int]]:
    if not apply_nav_traj:
        return []

    render_root = run_dir / "render_results"
    traj_dirs = sorted(
        {
            path
            for pattern in ("target*/traj*", "wonder*/traj*", "reconstruct*/traj*")
            for path in render_root.glob(pattern)
            if path.is_dir()
        },
        key=_world_nav_sort_key,
    )
    specs: list[tuple[str, str, str, str, list[str], str, int]] = []
    for group_index, traj_dir in enumerate(traj_dirs):
        folder_name = traj_dir.parent.name
        traj_name = traj_dir.name
        base_rel = traj_dir.relative_to(run_dir).as_posix()
        folder_rel = traj_dir.parent.relative_to(run_dir).as_posix()
        group_title = _world_nav_group_title(folder_name)
        item_prefix = f"worldnav-{folder_name}-{traj_name}"
        specs.extend(
            [
                (
                    f"{item_prefix}-start",
                    "world-nav",
                    f"{group_title} start",
                    "Start frame for this LLM-planned navigation route.",
                    [f"{folder_rel}/start_frame.png"],
                    group_title,
                    group_index,
                ),
                (
                    f"{item_prefix}-path",
                    "world-nav",
                    f"{group_title} path",
                    "WorldNav path visualization planned from scene analysis.",
                    [f"{base_rel}/traj_vis.png"],
                    group_title,
                    group_index,
                ),
                (
                    f"{item_prefix}-render",
                    "world-nav",
                    f"{group_title} render",
                    "Point-cloud render before video generation for this planned route.",
                    [f"{base_rel}/render.mp4"],
                    group_title,
                    group_index,
                ),
                (
                    f"{item_prefix}-mask",
                    "world-nav",
                    f"{group_title} mask",
                    "Mask for the planned route render; bright regions are supported by projected source points.",
                    [f"{base_rel}/render_mask.mp4"],
                    group_title,
                    group_index,
                ),
                (
                    f"{item_prefix}-generated",
                    "video",
                    f"{group_title} generated",
                    "WorldStereo output used as synthetic training views for this planned route.",
                    [f"{base_rel}/worldstereo-memory-dmd_result.mp4"],
                    group_title,
                    group_index,
                ),
            ]
        )
    return specs


def preview_items(
    job_id: str,
    run_dir: Path,
    split_view_num: int = DEFAULT_SPLIT_VIEW_NUM,
    trajectory_modes: list[str] | None = None,
    gs_max_steps: int = DEFAULT_GS_MAX_STEPS,
    indoor: bool = False,
    apply_nav_traj: bool = DEFAULT_APPLY_NAV_TRAJ,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, (stage, title, description, candidates) in enumerate(PREVIEW_SPECS[:1]):
        items.append(make_preview_item(job_id, run_dir, f"{stage}-{index}", stage, title, description, candidates))

    for item_id, stage, title, description, candidates in split_view_preview_specs(split_view_num, trajectory_modes):
        items.append(make_preview_item(job_id, run_dir, item_id, stage, title, description, candidates))

    for item_id, stage, title, description, candidates, group_title, group_order in world_nav_preview_specs(run_dir, apply_nav_traj):
        items.append(make_preview_item(job_id, run_dir, item_id, stage, title, description, candidates, group_title, group_order))

    final_step = gs_final_step(gs_max_steps)
    for index, (stage, title, description, candidates) in enumerate(PREVIEW_SPECS[1:], start=1):
        if indoor and title == "Sky mask":
            continue
        if title == "Validation render":
            candidates = [f"gs_result/renders/val_step{final_step}_0000.png", *candidates]
        items.append(make_preview_item(job_id, run_dir, f"{stage}-{index}", stage, title, description, candidates))
    return items


def _transpose_3x3(matrix: list[list[float]]) -> list[list[float]]:
    return [[matrix[row][col] for row in range(3)] for col in range(3)]


def _mat_vec_mul(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(matrix[row][col] * vector[col] for col in range(3)) for row in range(3)]


def _vec_add(a: list[float], b: list[float]) -> list[float]:
    return [a[i] + b[i] for i in range(3)]


def _vec_scale(vector: list[float], scale: float) -> list[float]:
    return [value * scale for value in vector]


def viewer_meta(run_dir: Path) -> dict[str, Any]:
    preview_meta_path = run_dir / "gs_result" / "ply" / "position_meta_info.json"
    if preview_meta_path.exists():
        preview_meta = json.loads(preview_meta_path.read_text(encoding="utf-8"))
        position = [float(value) for value in preview_meta["center_point"]]
        target = [float(value) for value in preview_meta["facing_direction"]]
        up = [float(value) for value in preview_meta["up_direction"]]
        return {
            "camera_key": "official-preview",
            "position": position,
            "target": target,
            "up": up,
            "fov": 70.0,
        }

    cameras_path = run_dir / "gs_data" / "cameras.json"
    if not cameras_path.exists():
        raise FileNotFoundError("Viewer camera metadata is not available yet.")

    cameras = json.loads(cameras_path.read_text(encoding="utf-8"))
    camera_keys = [key for key, value in cameras.items() if isinstance(value, dict) and "extrinsic" in value]
    if not camera_keys:
        raise FileNotFoundError("Viewer camera metadata does not contain any cameras.")

    key = "view0-traj0_000000" if "view0-traj0_000000" in cameras else sorted(camera_keys)[0]
    camera = cameras[key]
    extrinsic = camera["extrinsic"]
    intrinsic = camera.get("intrinsic") or [[240.0, 0.0, 416.0], [0.0, 240.0, 240.0], [0.0, 0.0, 1.0]]
    width = int(cameras.get("width") or camera.get("width") or 832)
    height = int(cameras.get("height") or camera.get("height") or 480)

    rotation = [[float(extrinsic[row][col]) for col in range(3)] for row in range(3)]
    translation = [float(extrinsic[row][3]) for row in range(3)]
    rotation_t = _transpose_3x3(rotation)
    position = _vec_scale(_mat_vec_mul(rotation_t, translation), -1.0)
    # HY-World camera extrinsics are OpenCV-style world-to-camera matrices:
    # camera +Z is forward and camera -Y is up in world space.
    forward = _mat_vec_mul(rotation_t, [0.0, 0.0, 1.0])
    up = _mat_vec_mul(rotation_t, [0.0, -1.0, 0.0])
    target = _vec_add(position, forward)
    fy = float(intrinsic[1][1]) if intrinsic and intrinsic[1][1] else 240.0
    fov = math.degrees(2.0 * math.atan(height / (2.0 * fy)))

    return {
        "camera_key": key,
        "position": position,
        "target": target,
        "up": up,
        "fov": fov,
        "width": width,
        "height": height,
    }


@dataclass
class Job:
    id: str
    state: str
    stage: str
    progress: str
    prompt: str
    run_dir: str
    created_at: str
    updated_at: str
    error: str | None = None
    split_view_num: int = DEFAULT_SPLIT_VIEW_NUM
    trajectory_modes: list[str] = field(default_factory=lambda: DEFAULT_TRAJECTORY_MODES.copy())
    indoor: bool = False
    gs_max_steps: int = DEFAULT_GS_MAX_STEPS
    apply_nav_traj: bool = DEFAULT_APPLY_NAV_TRAJ
    world_nav_attempts: int = DEFAULT_WORLD_NAV_ATTEMPTS
    prompt_source: str = "unknown"
    prompt_error: str | None = None
    input_hash: str | None = None
    input_image_sha256: str | None = None
    input_params: dict[str, Any] | None = None
    artifacts: dict[str, str] = field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        return asdict(self)


class JobStore:
    def __init__(self, path: Path):
        self.path = path
        self.jobs: dict[str, Job] = {}

    def load(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            return
        loaded_jobs: dict[str, Job] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            if "split_view_num" not in data:
                run_dir = Path(data.get("run_dir", ""))
                existing_views = [
                    path
                    for path in (run_dir / "render_results").glob("view*")
                    if path.is_dir() and (path / "start_frame.png").exists()
                ]
                if existing_views:
                    data["split_view_num"] = max(1, min(len(existing_views), 8))
            if "trajectory_modes" not in data:
                data["trajectory_modes"] = DEFAULT_TRAJECTORY_MODES.copy()
            else:
                try:
                    data["trajectory_modes"] = sanitize_trajectory_modes(data["trajectory_modes"])
                except ValueError:
                    data["trajectory_modes"] = DEFAULT_TRAJECTORY_MODES.copy()
            data.setdefault("indoor", False)
            data.setdefault("gs_max_steps", DEFAULT_GS_MAX_STEPS)
            data.setdefault("apply_nav_traj", DEFAULT_APPLY_NAV_TRAJ)
            data.setdefault("world_nav_attempts", DEFAULT_WORLD_NAV_ATTEMPTS)
            data.setdefault("prompt_source", "unknown")
            data.setdefault("prompt_error", None)
            data.setdefault("input_hash", None)
            data.setdefault("input_image_sha256", None)
            data.setdefault("input_params", None)
            if data["prompt_source"] == "unknown" and is_placeholder_prompt(str(data.get("prompt", ""))):
                data["prompt_source"] = "fallback"
                data["prompt_error"] = "This prompt matches a generic placeholder rather than an LLM scene description."
            try:
                data["gs_max_steps"] = sanitize_gs_max_steps(data["gs_max_steps"])
            except ValueError:
                data["gs_max_steps"] = DEFAULT_GS_MAX_STEPS
            try:
                data["world_nav_attempts"] = sanitize_world_nav_attempts(data["world_nav_attempts"])
            except (TypeError, ValueError):
                data["world_nav_attempts"] = DEFAULT_WORLD_NAV_ATTEMPTS
            job = Job(**data)
            loaded_jobs[job.id] = job
        self.jobs = loaded_jobs

        reconciled_jobs: list[Job] = []
        for job in list(self.jobs.values()):
            if job.state in {"queued", "running"}:
                spz_path, _ = artifact_path(Path(job.run_dir), "point_cloud_7999.spz", job.gs_max_steps)
                if spz_path.exists():
                    job.state = "succeeded"
                    job.stage = "complete"
                    job.progress = "SPZ export is ready."
                    job.error = None
                else:
                    job.state = "failed"
                    job.stage = "interrupted"
                    job.progress = "Backend restarted before this job finished."
                    job.error = job.progress
                job.updated_at = utc_now()
                reconciled_jobs.append(job)
        for job in reconciled_jobs:
            self.save(job)

    def save(self, job: Job) -> None:
        self.jobs[job.id] = job
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(job.public(), sort_keys=True) + "\n")

    def recent(self, limit: int = 25) -> list[Job]:
        return sorted(self.jobs.values(), key=lambda job: job.created_at, reverse=True)[:limit]


def make_job_record(
    *,
    job_id: str,
    run_dir: Path,
    split_view_num: int,
    trajectory_modes: list[str],
    indoor: bool,
    gs_max_steps: int,
    apply_nav_traj: bool,
    world_nav_attempts: int,
    input_hash_value: str | None,
    input_image_sha256: str | None,
    input_params: dict[str, Any] | None,
    prompt: str = "",
    prompt_source: str = "not_started",
    prompt_error: str | None = None,
    progress: str = "Panorama uploaded. Set split views and start.",
) -> Job:
    now = utc_now()
    return Job(
        id=job_id,
        state="ready",
        stage="uploaded",
        progress=progress,
        prompt=prompt,
        run_dir=str(run_dir),
        created_at=now,
        updated_at=now,
        split_view_num=split_view_num,
        trajectory_modes=trajectory_modes,
        indoor=indoor,
        gs_max_steps=gs_max_steps,
        apply_nav_traj=apply_nav_traj,
        world_nav_attempts=world_nav_attempts,
        prompt_source=prompt_source,
        prompt_error=prompt_error,
        input_hash=input_hash_value,
        input_image_sha256=input_image_sha256,
        input_params=input_params,
    )


class JobManager:
    def __init__(self, store: JobStore):
        self.store = store
        self.queue: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
        self.queue_tokens: dict[str, int] = {}
        self.subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = {}
        self.log_tails: dict[str, deque[str]] = {}
        self.preview_signatures: dict[str, set[str]] = {}
        self.active_process: asyncio.subprocess.Process | None = None
        self.active_job_id: str | None = None
        self.worker_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self.worker_task = asyncio.create_task(self.worker())

    async def stop(self) -> None:
        if self.active_process and self.active_process.returncode is None:
            await self.terminate_process(self.active_process)
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

    def add_job(self, job: Job) -> None:
        self.log_tails[job.id] = deque(maxlen=300)
        self.preview_signatures[job.id] = self.available_preview_signatures(job)
        self.store.save(job)
        self.publish(job.id, {"type": "status", "job": job.public()})

    def start_job(
        self,
        job: Job,
        split_view_num: int,
        trajectory_modes: list[str],
        indoor: bool,
        gs_max_steps: int,
        apply_nav_traj: bool,
        world_nav_attempts: int,
    ) -> None:
        job.split_view_num = split_view_num
        job.trajectory_modes = trajectory_modes
        job.indoor = indoor
        job.gs_max_steps = gs_max_steps
        job.apply_nav_traj = apply_nav_traj
        job.world_nav_attempts = world_nav_attempts
        job.error = None
        queue_token = self.queue_tokens.get(job.id, 0) + 1
        self.queue_tokens[job.id] = queue_token
        mode_text = ",".join(trajectory_modes)
        self.update(
            job,
            state="queued",
            stage="queued",
            progress=(
                f"Waiting for the GPU pipeline. split_view_num={split_view_num}, "
                f"trajectory_modes={mode_text}, apply_nav_traj={apply_nav_traj}, "
                f"world_nav_attempts={world_nav_attempts}, gs_max_steps={gs_max_steps}."
            ),
        )
        self.queue.put_nowait((job.id, queue_token))

    def update(self, job: Job, *, state: str | None = None, stage: str | None = None, progress: str | None = None, error: str | None = None) -> None:
        if state is not None:
            job.state = state
        if stage is not None:
            job.stage = stage
        if progress is not None:
            job.progress = progress
        if error is not None:
            job.error = error
        job.updated_at = utc_now()
        self.refresh_artifacts(job)
        self.store.save(job)
        self.publish(job.id, {"type": "status", "job": job.public()})

    def refresh_artifacts(self, job: Job) -> None:
        run_dir = Path(job.run_dir)
        artifacts: dict[str, str] = {}
        for name in (
            "panorama.png",
            "point_cloud_7999.spz",
            "point_cloud_7999.ply",
            "point_cloud_7999_playcanvas.ply",
            "ckpt_7999_rank0.pt",
            "pipeline.log",
        ):
            path, _ = artifact_path(run_dir, name, job.gs_max_steps)
            if path.exists():
                artifacts[name] = f"/api/jobs/{job.id}/artifacts/{name}"
        job.artifacts = artifacts

    def publish(self, job_id: str, payload: dict[str, Any]) -> None:
        for queue in list(self.subscribers.get(job_id, set())):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(payload)

    def available_preview_signatures(self, job: Job, items: list[dict[str, Any]] | None = None) -> set[str]:
        items = items if items is not None else preview_items(
            job.id,
            Path(job.run_dir),
            job.split_view_num,
            job.trajectory_modes,
            job.gs_max_steps,
            job.indoor,
            job.apply_nav_traj,
        )
        return {
            f"{item['id']}|{item.get('path') or ''}|{item.get('updated_at') or ''}"
            for item in items
            if item.get("available")
        }

    def publish_preview_updates(self, job: Job, reason: str = "file_available") -> None:
        items = preview_items(
            job.id,
            Path(job.run_dir),
            job.split_view_num,
            job.trajectory_modes,
            job.gs_max_steps,
            job.indoor,
            job.apply_nav_traj,
        )
        current = self.available_preview_signatures(job, items)
        previous = self.preview_signatures.setdefault(job.id, set())
        if not current - previous:
            return
        self.preview_signatures[job.id] = current
        self.publish(
            job.id,
            {
                "type": "preview_update",
                "job": job.public(),
                "reason": reason,
                "previews": items,
            },
        )

    async def monitor_preview_files(self, job: Job, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            try:
                self.publish_preview_updates(job)
            except Exception as exc:
                logger.warning("Could not publish preview update for %s: %s", job.id, exc)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
        try:
            self.publish_preview_updates(job)
        except Exception as exc:
            logger.warning("Could not publish final preview update for %s: %s", job.id, exc)

    async def subscribe(self, job_id: str):
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=200)
        self.subscribers.setdefault(job_id, set()).add(queue)
        try:
            job = self.store.jobs[job_id]
            yield {"type": "status", "job": job.public()}
            log_path = Path(job.run_dir) / "pipeline.log"
            if log_path.exists():
                for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-300:]:
                    yield {"type": "log", "line": line}
            while True:
                yield await queue.get()
        finally:
            self.subscribers.get(job_id, set()).discard(queue)

    async def worker(self) -> None:
        while True:
            job_id, queue_token = await self.queue.get()
            try:
                job = self.store.jobs.get(job_id)
                if not job or job.state == "canceled" or self.queue_tokens.get(job_id) != queue_token:
                    continue
                await self.run_job(job)
            finally:
                self.queue.task_done()

    async def run_job(self, job: Job) -> None:
        self.active_job_id = job.id
        try:
            self.update(job, state="running", stage="prompt synthesis", progress="Generating scene prompt from the uploaded image.")
            await self.synthesize_job_prompt(job)
            self.update(job, stage="starting", progress="Preparing HY-World pipeline.")
            for stage, cwd, command in pipeline_commands(
                Path(job.run_dir),
                job.prompt,
                job.split_view_num,
                job.trajectory_modes,
                job.gs_max_steps,
                job.apply_nav_traj,
                job.world_nav_attempts,
            ):
                if job.state == "canceled":
                    self.update(job, stage="canceled", progress="Job canceled.")
                    return
                if self.stage_is_complete(job, stage):
                    await self.skip_stage(job, stage)
                    continue
                await self.run_stage(job, stage, cwd, command)

            spz_path, _ = artifact_path(Path(job.run_dir), "point_cloud_7999.spz", job.gs_max_steps)
            if not spz_path.exists():
                raise RuntimeError("Pipeline finished without gs_result/ply/point_cloud_7999.spz.")
            self.update(job, state="succeeded", stage="complete", progress="SPZ export is ready.")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if job.state == "canceled":
                self.update(job, stage="canceled", progress="Job canceled.")
            else:
                self.update(job, state="failed", stage="failed", progress=str(exc), error=str(exc))
        finally:
            self.active_process = None
            self.active_job_id = None

    def expected_trajectory_dirs(self, job: Job, include_navigation: bool = True) -> list[Path]:
        render_root = Path(job.run_dir) / "render_results"
        modes = sanitize_trajectory_modes(job.trajectory_modes)
        dirs = [
            render_root / f"view{view_i}" / f"traj{traj_i}"
            for view_i in range(max(1, min(int(job.split_view_num), 8)))
            for traj_i in range(len(modes))
        ]
        if include_navigation and job.apply_nav_traj:
            dirs.extend(
                sorted(
                    path
                    for pattern in ("target*/traj*", "wonder*/traj*", "reconstruct*/traj*")
                    for path in render_root.glob(pattern)
                    if path.is_dir()
                )
            )
        return dirs

    def stage_is_complete(self, job: Job, stage: str) -> bool:
        run_dir = Path(job.run_dir)
        traj_dirs = self.expected_trajectory_dirs(job)
        if stage == "trajectory generation":
            regular_dirs = self.expected_trajectory_dirs(job, include_navigation=False)
            regular_complete = bool(regular_dirs) and all((path / "camera.json").exists() for path in regular_dirs)
            if not job.apply_nav_traj:
                return regular_complete
            nav_dirs = [path for path in traj_dirs if path not in regular_dirs]
            nav_artifact_exists = (run_dir / "navmesh" / "exploration" / "paths.json").exists()
            return regular_complete and nav_artifact_exists and bool(nav_dirs) and all((path / "camera.json").exists() for path in nav_dirs)
        if stage == "trajectory rendering":
            return bool(traj_dirs) and all((path / "render.mp4").exists() and (path / "render_mask.mp4").exists() for path in traj_dirs)
        if stage == "caption writing":
            render_paths = [path / "render.mp4" for path in traj_dirs]
            return bool(render_paths) and all(path.exists() and path.with_name("traj_caption.json").exists() for path in render_paths)
        if stage == "video generation":
            return (run_dir / "render_results" / "generation_bank_worldstereo-memory-dmd" / "aligned_pcd.ply").exists()
        if stage == "GS data generation":
            return (run_dir / "gs_data" / "cameras.json").exists()
        if stage.endswith("GS training"):
            spz_path, _ = artifact_path(run_dir, "point_cloud_7999.spz", job.gs_max_steps)
            return spz_path.exists()
        return False

    async def skip_stage(self, job: Job, stage: str) -> None:
        message = f"Reusing existing outputs for {stage}."
        self.update(job, stage=stage, progress=message)
        log_path = Path(job.run_dir) / "pipeline.log"
        with log_path.open("a", encoding="utf-8", errors="replace") as log:
            log.write(f"\n\n# {message}\n")
        self.publish_log_line(job, message)

    async def synthesize_job_prompt(self, job: Job) -> None:
        if job.prompt_source == "llm" and job.prompt.strip():
            update_scene_type(Path(job.run_dir), job.prompt, job.indoor)
            return

        data = await asyncio.to_thread(uploaded_image_data, Path(job.run_dir))
        prompt_synthesis = await asyncio.to_thread(synthesize_prompt_details, data)
        job.prompt = prompt_synthesis.prompt
        job.prompt_source = prompt_synthesis.source
        job.prompt_error = prompt_synthesis.error
        update_scene_type(Path(job.run_dir), job.prompt, job.indoor)
        if prompt_synthesis.source == "llm":
            self.update(job, stage="prompt synthesis", progress="Generated scene prompt from the uploaded image.")
        else:
            self.update(
                job,
                stage="prompt synthesis",
                progress=f"Using fallback prompt because LLM prompt synthesis did not produce a scene description: {prompt_synthesis.error}",
            )

    async def run_stage(self, job: Job, stage: str, cwd: Path, command: list[str]) -> None:
        self.update(job, stage=stage, progress=f"Starting {stage}.")
        stop_preview_monitor = asyncio.Event()
        preview_monitor = asyncio.create_task(self.monitor_preview_files(job, stop_preview_monitor))
        log_path = Path(job.run_dir) / "pipeline.log"
        try:
            with log_path.open("a", encoding="utf-8", errors="replace") as log:
                log.write(f"\n\n$ {' '.join(command)}\n")
                log.flush()
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                env["HYWORLD_DISABLE_SAM3"] = os.environ.get("HYWORLD_DISABLE_SAM3", "1")
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=str(cwd),
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    preexec_fn=os.setsid,
                )
                self.active_process = process
                assert process.stdout is not None
                buffered_line = ""
                while True:
                    raw_chunk = await process.stdout.read(8192)
                    if not raw_chunk:
                        break
                    text = raw_chunk.decode("utf-8", errors="replace")
                    log.write(text)
                    log.flush()

                    self.publish(job.id, {"type": "log_chunk", "chunk": text})
                    buffered_line = self.publish_log_text(job, buffered_line + text)
                    self.publish_preview_updates(job, reason=stage)
                    if job.state == "canceled":
                        await self.terminate_process(process)
                        break
                if buffered_line:
                    self.publish_log_line(job, buffered_line.rstrip())
                return_code = await process.wait()
                self.active_process = None
                if job.state == "canceled":
                    raise RuntimeError("Job canceled.")
                if return_code != 0:
                    raise RuntimeError(f"{stage} exited with code {return_code}.")
        finally:
            stop_preview_monitor.set()
            try:
                await preview_monitor
            except asyncio.CancelledError:
                pass
        self.update(job, stage=stage, progress=f"Finished {stage}.")
        self.publish_preview_updates(job, reason=stage)

    def publish_log_text(self, job: Job, text: str) -> str:
        parts = text.splitlines(keepends=True)
        if not parts:
            return ""
        if not parts[-1].endswith(("\n", "\r")):
            remainder = parts.pop()
        else:
            remainder = ""
        for part in parts:
            self.publish_log_line(job, part.rstrip())
        return remainder

    def publish_log_line(self, job: Job, line: str) -> None:
        if not line:
            return
        self.log_tails.setdefault(job.id, deque(maxlen=300)).append(line)
        job.progress = line[-500:]
        job.updated_at = utc_now()
        self.publish(job.id, {"type": "log", "line": line})
        self.publish(job.id, {"type": "status", "job": job.public()})

    async def cancel(self, job: Job) -> Job:
        if job.state in {"succeeded", "failed", "canceled"}:
            return job
        self.update(job, state="canceled", stage="canceled", progress="Cancel requested.")
        if self.active_job_id == job.id and self.active_process:
            await self.terminate_process(self.active_process)
        return job

    async def terminate_process(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=10)
        except asyncio.TimeoutError:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()


def pipeline_commands(
    scene_dir: Path,
    prompt: str,
    split_view_num: int = DEFAULT_SPLIT_VIEW_NUM,
    trajectory_modes: list[str] | None = None,
    gs_max_steps: int = DEFAULT_GS_MAX_STEPS,
    apply_nav_traj: bool = DEFAULT_APPLY_NAV_TRAJ,
    world_nav_attempts: int | None = DEFAULT_WORLD_NAV_ATTEMPTS,
) -> list[tuple[str, Path, list[str]]]:
    scene = str(scene_dir)
    split_views = str(max(1, min(int(split_view_num), 8)))
    modes = ",".join(sanitize_trajectory_modes(trajectory_modes))
    gs_steps = str(sanitize_gs_max_steps(gs_max_steps))
    trajectory_command = [
        "python",
        "traj_generate.py",
        "--target_path",
        scene,
        "--split_view_num",
        split_views,
        "--trajectory_modes",
        modes,
        "--skip_exist",
    ]
    if apply_nav_traj:
        trajectory_command.append("--apply_nav_traj")
        selected_world_nav_attempts = sanitize_world_nav_attempts(world_nav_attempts)
        trajectory_command.extend(["--wonder_topk", str(selected_world_nav_attempts), "--recon_topk", str(selected_world_nav_attempts)])
    return [
        (
            "trajectory generation",
            WORLDGEN_DIR,
            trajectory_command,
        ),
        ("trajectory rendering", WORLDGEN_DIR, ["torchrun", "--nproc_per_node", "1", "traj_render.py", "--target_path", scene, "--skip_exist"]),
        ("caption writing", ROOT_DIR, ["python", "scripts/write_traj_captions.py", "--target-path", scene, "--prompt", prompt]),
        ("video generation", WORLDGEN_DIR, ["torchrun", "--nproc_per_node", "1", "video_gen.py", "--target_path", scene, "--skip_exist"]),
        ("GS data generation", WORLDGEN_DIR, ["torchrun", "--nproc_per_node", "1", "gen_gs_data.py", "--root_path", scene, "--save_normal", "--split_sky"]),
        (
            "8000-step GS training",
            WORLDGEN_DIR,
            [
                "python",
                "-m",
                "world_gs_trainer",
                "default",
                "--data_dir",
                f"{scene}/gs_data",
                "--result_dir",
                f"{scene}/gs_result",
                "--max_steps",
                gs_steps,
                "--save_steps",
                gs_steps,
                "--eval_steps",
                gs_steps,
                "--ply_steps",
                gs_steps,
                "--save_ply",
                "--convert_to_spz",
                "--disable_video",
                "--disable-viewer",
                "--use_scale_regularization",
                "--antialiased",
                "--depth_loss",
                "--normal_loss",
                "--sky_depth_from_pcd",
                "--use_mask_gaussian",
                "--mask_export_stochastic",
                "--no-mask-export-anchor-protection",
                "--use_anchor_protection",
                "--strategy.refine-start-iter",
                "150",
                "--strategy.refine-stop-iter",
                "750",
                "--strategy.refine-every",
                "100",
                "--strategy.refine-scale2d-stop-iter",
                "750",
                "--strategy.reset-every",
                "99990",
                "--strategy.grow-grad2d",
                "0.0001",
                "--strategy.prune-scale3d",
                "0.1",
            ],
        ),
    ]


store = JobStore(JOBS_LOG)
manager = JobManager(store)

app = FastAPI(title="HY-World Web Pipeline")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup() -> None:
    store.load()
    manager.start()


@app.on_event("shutdown")
async def shutdown() -> None:
    await manager.stop()


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    split_view_num: int = Form(DEFAULT_SPLIT_VIEW_NUM),
    trajectory_modes: str = Form(",".join(DEFAULT_TRAJECTORY_MODES)),
    indoor: bool = Form(False),
    gs_max_steps: int = Form(DEFAULT_GS_MAX_STEPS),
    apply_nav_traj: bool = Form(DEFAULT_APPLY_NAV_TRAJ),
    world_nav_attempts: str | None = Form(str(DEFAULT_WORLD_NAV_ATTEMPTS)),
):
    if split_view_num < 1 or split_view_num > 8:
        raise HTTPException(status_code=400, detail="split_view_num must be between 1 and 8.")
    try:
        selected_modes = sanitize_trajectory_modes(trajectory_modes)
        selected_gs_max_steps = sanitize_gs_max_steps(gs_max_steps)
        selected_world_nav_attempts = sanitize_world_nav_attempts(world_nav_attempts)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    input_hash_value, image_sha256, input_params = task_hash(
        data,
        split_view_num,
        selected_modes,
        indoor,
        selected_gs_max_steps,
        apply_nav_traj,
        selected_world_nav_attempts,
    )
    job_id = f"task_{input_hash_value[:20]}"
    run_dir = OUTPUT_ROOT / job_id

    existing = store.jobs.get(job_id)
    if existing:
        manager.refresh_artifacts(existing)
        manager.publish(existing.id, {"type": "status", "job": existing.public()})
        return existing.public()

    try:
        prepare_job_files(run_dir, file.filename, data, "", indoor=indoor)
    except FileExistsError:
        if not (run_dir / "panorama.png").exists():
            raise HTTPException(status_code=409, detail=f"Task directory already exists but is incomplete: {job_id}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job = make_job_record(
        job_id=job_id,
        run_dir=run_dir,
        split_view_num=split_view_num,
        trajectory_modes=selected_modes,
        indoor=indoor,
        gs_max_steps=selected_gs_max_steps,
        apply_nav_traj=apply_nav_traj,
        world_nav_attempts=selected_world_nav_attempts,
        input_hash_value=input_hash_value,
        input_image_sha256=image_sha256,
        input_params=input_params,
    )
    manager.refresh_artifacts(job)
    manager.add_job(job)
    return job.public()


@app.post("/api/jobs/{job_id}/start")
async def start_job(
    job_id: str,
    split_view_num: int | None = None,
    trajectory_modes: str | None = None,
    indoor: bool | None = None,
    gs_max_steps: int | None = None,
    apply_nav_traj: bool | None = None,
    world_nav_attempts: str | None = None,
):
    job = store.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    split_view_num = job.split_view_num if split_view_num is None else split_view_num
    if split_view_num < 1 or split_view_num > 8:
        raise HTTPException(status_code=400, detail="split_view_num must be between 1 and 8.")
    try:
        selected_modes = sanitize_trajectory_modes(job.trajectory_modes if trajectory_modes is None else trajectory_modes)
        selected_gs_max_steps = sanitize_gs_max_steps(job.gs_max_steps if gs_max_steps is None else gs_max_steps)
        selected_world_nav_attempts = sanitize_world_nav_attempts(job.world_nav_attempts if world_nav_attempts is None else world_nav_attempts)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    selected_indoor = job.indoor if indoor is None else indoor
    selected_apply_nav_traj = job.apply_nav_traj if apply_nav_traj is None else apply_nav_traj

    if job.input_hash:
        try:
            data = await asyncio.to_thread(uploaded_image_data, Path(job.run_dir))
            input_hash_value, image_sha256, input_params = task_hash(
                data,
                split_view_num,
                selected_modes,
                selected_indoor,
                selected_gs_max_steps,
                selected_apply_nav_traj,
                selected_world_nav_attempts,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not compute task hash: {exc}") from exc
        target_id = f"task_{input_hash_value[:20]}"
        if target_id != job.id:
            target_job = store.jobs.get(target_id)
            if target_job is None:
                target_run_dir = OUTPUT_ROOT / target_id
                try:
                    prepare_job_files(target_run_dir, None, data, job.prompt, indoor=selected_indoor)
                except FileExistsError:
                    if not (target_run_dir / "panorama.png").exists():
                        raise HTTPException(status_code=409, detail=f"Task directory already exists but is incomplete: {target_id}")
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                target_job = make_job_record(
                    job_id=target_id,
                    run_dir=target_run_dir,
                    split_view_num=split_view_num,
                    trajectory_modes=selected_modes,
                    indoor=selected_indoor,
                    gs_max_steps=selected_gs_max_steps,
                    apply_nav_traj=selected_apply_nav_traj,
                    world_nav_attempts=selected_world_nav_attempts,
                    input_hash_value=input_hash_value,
                    input_image_sha256=image_sha256,
                    input_params=input_params,
                    prompt=job.prompt,
                    prompt_source=job.prompt_source,
                    prompt_error=job.prompt_error,
                    progress="Created from the same uploaded image with updated parameters.",
                )
                manager.refresh_artifacts(target_job)
                manager.add_job(target_job)
            job = target_job

    if job.state in {"queued", "running", "succeeded"}:
        manager.refresh_artifacts(job)
        return job.public()
    if job.state not in {"ready", "failed", "canceled"}:
        raise HTTPException(status_code=409, detail=f"Job is {job.state}, not startable.")

    manager.start_job(job, split_view_num, selected_modes, selected_indoor, selected_gs_max_steps, selected_apply_nav_traj, selected_world_nav_attempts)
    return job.public()


@app.get("/api/jobs")
async def list_jobs():
    return [job.public() for job in store.recent()]


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = store.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    manager.refresh_artifacts(job)
    return job.public()


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    if job_id not in store.jobs:
        raise HTTPException(status_code=404, detail="Job not found.")

    async def stream():
        async for event in manager.subscribe(job_id):
            yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    job = store.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    await manager.cancel(job)
    return job.public()


@app.get("/api/jobs/{job_id}/artifacts/{name}")
async def get_artifact(job_id: str, name: str):
    job = store.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    try:
        path, media_type = artifact_path(Path(job.run_dir), name, job.gs_max_steps)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Artifact not found.") from exc
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact is not available yet.")
    return FileResponse(
        path,
        media_type=media_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        filename=path.name,
    )


@app.get("/api/jobs/{job_id}/previews")
async def get_previews(job_id: str):
    job = store.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return preview_items(job.id, Path(job.run_dir), job.split_view_num, job.trajectory_modes, job.gs_max_steps, job.indoor, job.apply_nav_traj)


@app.get("/api/jobs/{job_id}/preview-files/{rel_path:path}")
async def get_preview_file(job_id: str, rel_path: str):
    job = store.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    try:
        path = safe_run_file(Path(job.run_dir), rel_path)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Preview not found.") from exc
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Preview is not available yet.")
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=path.name)


@app.get("/api/jobs/{job_id}/viewer-meta")
async def get_viewer_meta(job_id: str):
    job = store.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    try:
        return viewer_meta(Path(job.run_dir))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
