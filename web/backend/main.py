import asyncio
import json
import math
import mimetypes
import os
import signal
import uuid
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
from PIL import Image, UnidentifiedImageError


ROOT_DIR = Path(os.environ.get("HYWORLD_ROOT", Path(__file__).resolve().parents[2]))
WORLDGEN_DIR = ROOT_DIR / "hyworld2" / "worldgen"
OUTPUT_ROOT = ROOT_DIR / "outputs" / "web_runs"
JOBS_LOG = OUTPUT_ROOT / "jobs.jsonl"
TARGET_SIZE = (1920, 960)
DEFAULT_SPLIT_VIEW_NUM = 4

ARTIFACTS = {
    "panorama.png": ("panorama.png", "image/png"),
    "point_cloud_7999.spz": ("gs_result/ply/point_cloud_7999.spz", "application/octet-stream"),
    "point_cloud_7999.ply": ("gs_result/ply/point_cloud_7999.ply", "application/octet-stream"),
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

TRAJECTORY_PREVIEW_LABELS = {
    0: ("right rotation", "Camera rotates right from this split view."),
    1: ("left rotation", "Camera rotates left from this split view."),
    2: ("up-right aerial", "Camera pitches upward and rotates toward the right."),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_job_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def safe_original_name(filename: str | None) -> str:
    suffix = Path(filename or "upload").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
        suffix = ".img"
    return f"original_upload{suffix}"


def prepare_job_files(run_dir: Path, filename: str | None, data: bytes, prompt: str) -> None:
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
        json.dumps({"scene_type": "outdoor", "prompt": prompt}, indent=2),
        encoding="utf-8",
    )
    (run_dir / "pipeline.log").touch()


def artifact_path(run_dir: Path, name: str) -> tuple[Path, str]:
    if name not in ARTIFACTS:
        raise KeyError(name)
    rel_path, media_type = ARTIFACTS[name]
    return run_dir / rel_path, media_type


def safe_run_file(run_dir: Path, rel_path: str) -> Path:
    root = run_dir.resolve()
    candidate = (run_dir / rel_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("Path is outside this job run.")
    return candidate


def make_preview_item(job_id: str, run_dir: Path, item_id: str, stage: str, title: str, description: str, candidates: list[str]) -> dict[str, Any]:
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
    }
    if selected and selected_rel:
        item["url"] = f"/api/jobs/{job_id}/preview-files/{quote(selected_rel, safe='/')}"
        item["updated_at"] = datetime.fromtimestamp(selected.stat().st_mtime, timezone.utc).isoformat()
    return item


def split_view_preview_specs(split_view_num: int) -> list[tuple[str, str, str, str, list[str]]]:
    specs: list[tuple[str, str, str, str, list[str]]] = []
    split_count = max(1, min(int(split_view_num), 8))
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
        for traj_i in range(3):
            label, caption = TRAJECTORY_PREVIEW_LABELS.get(traj_i, (f"traj {traj_i}", "Generated camera trajectory."))
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


def preview_items(job_id: str, run_dir: Path, split_view_num: int = DEFAULT_SPLIT_VIEW_NUM) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, (stage, title, description, candidates) in enumerate(PREVIEW_SPECS[:1]):
        items.append(make_preview_item(job_id, run_dir, f"{stage}-{index}", stage, title, description, candidates))

    for item_id, stage, title, description, candidates in split_view_preview_specs(split_view_num):
        items.append(make_preview_item(job_id, run_dir, item_id, stage, title, description, candidates))

    for index, (stage, title, description, candidates) in enumerate(PREVIEW_SPECS[1:], start=1):
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
            job = Job(**data)
            loaded_jobs[job.id] = job
        self.jobs = loaded_jobs

        reconciled_jobs: list[Job] = []
        for job in list(self.jobs.values()):
            if job.state in {"queued", "running"}:
                spz_path, _ = artifact_path(Path(job.run_dir), "point_cloud_7999.spz")
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


class JobManager:
    def __init__(self, store: JobStore):
        self.store = store
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = {}
        self.log_tails: dict[str, deque[str]] = {}
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
        self.store.save(job)
        self.publish(job.id, {"type": "status", "job": job.public()})

    def start_job(self, job: Job, split_view_num: int) -> None:
        job.split_view_num = split_view_num
        self.update(job, state="queued", stage="queued", progress=f"Waiting for the GPU pipeline. split_view_num={split_view_num}.")
        self.queue.put_nowait(job.id)

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
        for name in ("panorama.png", "point_cloud_7999.spz", "point_cloud_7999.ply", "ckpt_7999_rank0.pt", "pipeline.log"):
            path, _ = artifact_path(run_dir, name)
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
            job_id = await self.queue.get()
            try:
                job = self.store.jobs.get(job_id)
                if not job or job.state == "canceled":
                    continue
                await self.run_job(job)
            finally:
                self.queue.task_done()

    async def run_job(self, job: Job) -> None:
        self.active_job_id = job.id
        try:
            self.update(job, state="running", stage="starting", progress="Preparing HY-World pipeline.")
            for stage, cwd, command in pipeline_commands(Path(job.run_dir), job.prompt, job.split_view_num):
                if job.state == "canceled":
                    self.update(job, stage="canceled", progress="Job canceled.")
                    return
                await self.run_stage(job, stage, cwd, command)

            spz_path, _ = artifact_path(Path(job.run_dir), "point_cloud_7999.spz")
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

    async def run_stage(self, job: Job, stage: str, cwd: Path, command: list[str]) -> None:
        self.update(job, stage=stage, progress=f"Starting {stage}.")
        log_path = Path(job.run_dir) / "pipeline.log"
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

                buffered_line = self.publish_log_text(job, buffered_line + text)
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
        self.update(job, stage=stage, progress=f"Finished {stage}.")

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


def pipeline_commands(scene_dir: Path, prompt: str, split_view_num: int = DEFAULT_SPLIT_VIEW_NUM) -> list[tuple[str, Path, list[str]]]:
    scene = str(scene_dir)
    split_views = str(max(1, min(int(split_view_num), 8)))
    return [
        ("trajectory generation", WORLDGEN_DIR, ["python", "traj_generate.py", "--target_path", scene, "--split_view_num", split_views]),
        ("trajectory rendering", WORLDGEN_DIR, ["torchrun", "--nproc_per_node", "1", "traj_render.py", "--target_path", scene]),
        ("caption writing", ROOT_DIR, ["python", "scripts/write_traj_captions.py", "--target-path", scene, "--prompt", prompt]),
        ("video generation", WORLDGEN_DIR, ["torchrun", "--nproc_per_node", "1", "video_gen.py", "--target_path", scene]),
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
                "8000",
                "--save_steps",
                "8000",
                "--eval_steps",
                "8000",
                "--ply_steps",
                "8000",
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
    prompt: str = Form("uploaded panorama"),
    split_view_num: int = Form(DEFAULT_SPLIT_VIEW_NUM),
):
    if split_view_num < 1 or split_view_num > 8:
        raise HTTPException(status_code=400, detail="split_view_num must be between 1 and 8.")
    job_id = make_job_id()
    run_dir = OUTPUT_ROOT / job_id
    data = await file.read()
    try:
        prepare_job_files(run_dir, file.filename, data, prompt or "uploaded panorama")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    now = utc_now()
    job = Job(
        id=job_id,
        state="ready",
        stage="uploaded",
        progress="Panorama uploaded. Set split views and start.",
        prompt=prompt or "uploaded panorama",
        run_dir=str(run_dir),
        created_at=now,
        updated_at=now,
        split_view_num=split_view_num,
    )
    manager.refresh_artifacts(job)
    manager.add_job(job)
    return job.public()


@app.post("/api/jobs/{job_id}/start")
async def start_job(job_id: str, split_view_num: int | None = None):
    job = store.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.state != "ready":
        raise HTTPException(status_code=409, detail=f"Job is {job.state}, not ready.")
    split_view_num = job.split_view_num if split_view_num is None else split_view_num
    if split_view_num < 1 or split_view_num > 8:
        raise HTTPException(status_code=400, detail="split_view_num must be between 1 and 8.")
    manager.start_job(job, split_view_num)
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
        path, media_type = artifact_path(Path(job.run_dir), name)
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
    return preview_items(job.id, Path(job.run_dir), job.split_view_num)


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
