import React, { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { Activity, Ban, Box, CheckCircle2, Download, FileArchive, FileText, ImageUp, Loader2, Play, RotateCcw, UploadCloud, X, XCircle } from "lucide-react";
import * as THREE from "three";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";
const DEFAULT_SPLIT_VIEW_NUM = 4;
const DEFAULT_TRAJECTORY_MODES = ["forward", "left-translation", "right-translation"];
const DEFAULT_APPLY_NAV_TRAJ = false;
const DEFAULT_GS_MAX_STEPS = 8000;
const DEFAULT_WORLD_NAV_WONDER_TOPK = 3;
const DEFAULT_WORLD_NAV_RECON_TOPK = 5;
const MAX_WORLD_NAV_TOPK = 20;
const MAX_LOG_CHARS = 1_000_000;
const SETTINGS_STORAGE_KEY = "hyworld2.pipelineSettings";
const STARTABLE_JOB_STATES = new Set<JobState>(["ready", "failed", "canceled"]);
const TRAJECTORY_MODE_OPTIONS = [
  ["right-rotation", "Right rotation"],
  ["left-rotation", "Left rotation"],
  ["up-right-aerial", "Up-right aerial"],
  ["up-rotation", "Up rotation"],
  ["forward", "Forward translation"],
  ["backward", "Backward translation"],
  ["right-translation", "Right translation"],
  ["left-translation", "Left translation"],
] as const;
const MAX_TRAJECTORIES_PER_VIEW = 8;

type JobState = "ready" | "queued" | "running" | "succeeded" | "failed" | "canceled";

type Job = {
  id: string;
  state: JobState;
  stage: string;
  progress: string;
  prompt: string;
  prompt_source?: string;
  prompt_error?: string | null;
  run_dir: string;
  created_at: string;
  updated_at: string;
  error: string | null;
  split_view_num: number;
  trajectory_modes: string[];
  indoor: boolean;
  gs_max_steps: number;
  apply_nav_traj: boolean;
  world_nav_wonder_topk?: number | null;
  world_nav_recon_topk?: number | null;
  artifacts: Record<string, string>;
};

type SseEvent = {
  type: "status" | "log" | "log_chunk" | "preview_update";
  job?: Job;
  line?: string;
  chunk?: string;
  previews?: PreviewItem[];
  reason?: string;
};

type ViewerMeta = {
  camera_key?: string;
  position: [number, number, number];
  target: [number, number, number];
  up: [number, number, number];
  fov: number;
};

type PreviewItem = {
  id: string;
  stage: string;
  title: string;
  description: string;
  available: boolean;
  kind: "image" | "video";
  media_type: string | null;
  path: string | null;
  url: string | null;
  updated_at: string | null;
  group_title?: string | null;
  group_order?: number | null;
};

type PipelineSettings = {
  splitViewNum: number;
  trajectoryModes: string[];
  indoor: boolean;
  gsMaxSteps: number;
  applyNavTraj: boolean;
  worldNavWonderTopk: number;
  worldNavReconTopk: number;
};

function apiUrl(path: string) {
  return `${API_BASE}${path}`;
}

function absoluteArtifactUrl(path: string) {
  if (path.startsWith("http")) return path;
  if (API_BASE.startsWith("http")) return `${API_BASE}${path}`;
  return `${window.location.origin}${API_BASE}${path}`;
}

function withCacheBuster(url: string, value: string) {
  const parsed = new URL(url, window.location.origin);
  parsed.searchParams.set("v", value);
  return parsed.toString();
}

function clampInteger(value: number, fallback: number, min: number, max: number) {
  const rounded = Math.round(Number.isFinite(value) ? value : fallback);
  return Math.max(min, Math.min(max, rounded || fallback));
}

function normalizeRequiredInteger(value: unknown, fallback: number, min: number, max: number): number {
  if (value === "" || value === null || value === undefined) return fallback;
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  return clampInteger(numeric, fallback, min, max);
}

function defaultPipelineSettings(): PipelineSettings {
  return {
    splitViewNum: DEFAULT_SPLIT_VIEW_NUM,
    trajectoryModes: DEFAULT_TRAJECTORY_MODES,
    indoor: false,
    gsMaxSteps: DEFAULT_GS_MAX_STEPS,
    applyNavTraj: DEFAULT_APPLY_NAV_TRAJ,
    worldNavWonderTopk: DEFAULT_WORLD_NAV_WONDER_TOPK,
    worldNavReconTopk: DEFAULT_WORLD_NAV_RECON_TOPK,
  };
}

function normalizeTrajectoryModes(modes: string[] | undefined, count?: number) {
  const supported = new Set(TRAJECTORY_MODE_OPTIONS.map(([value]) => value));
  const cleaned = (modes?.filter((mode) => supported.has(mode as (typeof TRAJECTORY_MODE_OPTIONS)[number][0])) ?? []);
  const baseCount = count ?? (cleaned.length || DEFAULT_TRAJECTORY_MODES.length);
  const targetCount = clampInteger(baseCount, DEFAULT_TRAJECTORY_MODES.length, 1, MAX_TRAJECTORIES_PER_VIEW);
  const next = cleaned.slice(0, targetCount);
  for (let index = next.length; index < targetCount; index += 1) {
    next.push(TRAJECTORY_MODE_OPTIONS[index % TRAJECTORY_MODE_OPTIONS.length][0]);
  }
  return next;
}

function normalizePipelineSettings(value: Partial<PipelineSettings> = {}): PipelineSettings {
  const defaults = defaultPipelineSettings();
  return {
    splitViewNum: clampInteger(value.splitViewNum ?? defaults.splitViewNum, defaults.splitViewNum, 1, 8),
    trajectoryModes: normalizeTrajectoryModes(value.trajectoryModes ?? defaults.trajectoryModes),
    indoor: value.indoor ?? defaults.indoor,
    gsMaxSteps: clampInteger(value.gsMaxSteps ?? defaults.gsMaxSteps, defaults.gsMaxSteps, 100, 50000),
    applyNavTraj: value.applyNavTraj ?? defaults.applyNavTraj,
    worldNavWonderTopk: normalizeRequiredInteger(
      value.worldNavWonderTopk ?? (value as Partial<PipelineSettings> & { worldNavAttempts?: number }).worldNavAttempts,
      DEFAULT_WORLD_NAV_WONDER_TOPK,
      1,
      MAX_WORLD_NAV_TOPK,
    ),
    worldNavReconTopk: normalizeRequiredInteger(
      value.worldNavReconTopk ?? (value as Partial<PipelineSettings> & { worldNavAttempts?: number }).worldNavAttempts,
      DEFAULT_WORLD_NAV_RECON_TOPK,
      1,
      MAX_WORLD_NAV_TOPK,
    ),
  };
}

function loadPipelineSettings() {
  if (typeof window === "undefined") return defaultPipelineSettings();
  try {
    const raw = window.localStorage.getItem(SETTINGS_STORAGE_KEY);
    return raw ? normalizePipelineSettings(JSON.parse(raw) as Partial<PipelineSettings>) : defaultPipelineSettings();
  } catch {
    return defaultPipelineSettings();
  }
}

function appendLogText(existing: string, text: string) {
  if (!text) return existing;
  const next = `${existing}${text}`;
  return next.length > MAX_LOG_CHARS ? next.slice(next.length - MAX_LOG_CHARS) : next;
}

function rdfToRubVector(values: [number, number, number]) {
  return new THREE.Vector3(values[0], -values[1], -values[2]);
}

function vectorFromTuple(values: [number, number, number]) {
  return new THREE.Vector3(values[0], values[1], values[2]);
}

function createPlaycanvasMetaTransform(meta: ViewerMeta) {
  const up = vectorFromTuple(meta.up).normalize();
  const facing = vectorFromTuple(meta.target);
  if (meta.camera_key !== "official-preview") {
    facing.sub(vectorFromTuple(meta.position));
  }
  facing.addScaledVector(up, -facing.dot(up)).normalize();
  const right = new THREE.Vector3().crossVectors(facing, up).normalize();
  if (
    !Number.isFinite(up.lengthSq()) ||
    !Number.isFinite(facing.lengthSq()) ||
    !Number.isFinite(right.lengthSq()) ||
    up.lengthSq() === 0 ||
    facing.lengthSq() === 0 ||
    right.lengthSq() === 0
  ) {
    return null;
  }

  return (values: [number, number, number]) => {
    const vector = vectorFromTuple(values);
    return new THREE.Vector3(vector.dot(right), -vector.dot(up), vector.dot(facing));
  };
}

function normalizeFiniteVector(vector: THREE.Vector3) {
  return Number.isFinite(vector.lengthSq()) && vector.lengthSq() > 0 ? vector.normalize() : null;
}

function useJobs() {
  const [jobs, setJobs] = useState<Job[]>([]);

  async function refresh() {
    const response = await fetch(apiUrl("/api/jobs"));
    if (response.ok) {
      setJobs(await response.json());
    }
  }

  useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, 5000);
    return () => window.clearInterval(timer);
  }, []);

  return { jobs, setJobs, refresh };
}

function SparkViewer({ jobId, url }: { jobId: string | null; url: string | null }) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const [loadState, setLoadState] = useState(url ? "loading" : "idle");

  useEffect(() => {
    const host = hostRef.current;
    if (!host || !url) {
      setLoadState("idle");
      return;
    }

    setLoadState("loading");
    let disposed = false;
    const metaPromise: Promise<ViewerMeta | null> = jobId
      ? fetch(apiUrl(`/api/jobs/${jobId}/viewer-meta`))
          .then((response) => (response.ok ? response.json() : null))
          .catch(() => null)
      : Promise.resolve(null);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0f14);

    const camera = new THREE.PerspectiveCamera(62, 1, 0.01, 2000);
    camera.up.set(0, -1, 0);
    camera.position.set(0, 0.25, 3.3);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.domElement.tabIndex = 0;
    renderer.domElement.style.outline = "none";
    host.appendChild(renderer.domElement);

    const spark = new SparkRenderer({ renderer });
    scene.add(spark);

    const pressedKeys = new Set<string>();
    const moveKeys = new Set(["KeyW", "KeyA", "KeyS", "KeyD"]);
    const rotateKeys = new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"]);
    const handledKeys = new Set([...moveKeys, ...rotateKeys]);
    const lookAtPoint = new THREE.Vector3(0, 0, 0);
    const lookDistance = { value: 3.0 };
    const forward = new THREE.Vector3();
    const right = new THREE.Vector3();
    const move = new THREE.Vector3();
    const pointer = { dragging: false, id: -1, x: 0, y: 0 };

    const focusCanvas = () => renderer.domElement.focus();
    const isTypingTarget = (target: EventTarget | null) => {
      if (!(target instanceof HTMLElement)) return false;
      return target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (handledKeys.has(event.code) && !isTypingTarget(event.target)) {
        pressedKeys.add(event.code);
        event.preventDefault();
      }
    };
    const handleKeyUp = (event: KeyboardEvent) => {
      if (handledKeys.has(event.code) && !isTypingTarget(event.target)) {
        pressedKeys.delete(event.code);
        event.preventDefault();
      }
    };
    const setCameraLook = () => {
      camera.lookAt(lookAtPoint);
      camera.updateMatrixWorld();
    };
    const rotateView = (yawRadians: number, pitchRadians: number) => {
      camera.getWorldDirection(forward).normalize();

      if (yawRadians !== 0) {
        forward.applyAxisAngle(camera.up, yawRadians).normalize();
      }

      right.copy(forward).cross(camera.up).normalize();
      if (pitchRadians !== 0 && right.lengthSq() > 0) {
        const pitched = forward.clone().applyAxisAngle(right, pitchRadians).normalize();
        if (Math.abs(pitched.dot(camera.up)) < 0.96) forward.copy(pitched);
      }

      lookAtPoint.copy(camera.position).addScaledVector(forward, lookDistance.value);
      setCameraLook();
    };
    const applyKeyboardRotate = (deltaSeconds: number) => {
      const yawAmount = (pressedKeys.has("ArrowRight") ? 1 : 0) - (pressedKeys.has("ArrowLeft") ? 1 : 0);
      const pitchAmount = (pressedKeys.has("ArrowUp") ? 1 : 0) - (pressedKeys.has("ArrowDown") ? 1 : 0);
      if (yawAmount === 0 && pitchAmount === 0) return;

      const rotateSpeed = 1.9;
      rotateView(-yawAmount * rotateSpeed * deltaSeconds, pitchAmount * rotateSpeed * deltaSeconds);
    };
    const applyKeyboardMove = (deltaSeconds: number) => {
      const forwardAmount = (pressedKeys.has("KeyW") ? 1 : 0) - (pressedKeys.has("KeyS") ? 1 : 0);
      const rightAmount = (pressedKeys.has("KeyD") ? 1 : 0) - (pressedKeys.has("KeyA") ? 1 : 0);
      if (forwardAmount === 0 && rightAmount === 0) return;

      camera.getWorldDirection(forward).normalize();
      right.copy(forward).cross(camera.up).normalize();
      move.set(0, 0, 0).addScaledVector(forward, forwardAmount).addScaledVector(right, rightAmount);
      if (move.lengthSq() === 0) return;

      const speed = THREE.MathUtils.clamp(lookDistance.value * 1.6, 0.25, 5.0);
      move.normalize().multiplyScalar(speed * deltaSeconds);
      camera.position.add(move);
      lookAtPoint.add(move);
      setCameraLook();
    };
    const handlePointerDown = (event: PointerEvent) => {
      focusCanvas();
      pointer.dragging = true;
      pointer.id = event.pointerId;
      pointer.x = event.clientX;
      pointer.y = event.clientY;
      renderer.domElement.setPointerCapture(event.pointerId);
      event.preventDefault();
    };
    const handlePointerMove = (event: PointerEvent) => {
      if (!pointer.dragging || event.pointerId !== pointer.id) return;
      const dx = event.clientX - pointer.x;
      const dy = event.clientY - pointer.y;
      pointer.x = event.clientX;
      pointer.y = event.clientY;
      rotateView(-dx * 0.0035, -dy * 0.0035);
      event.preventDefault();
    };
    const handlePointerUp = (event: PointerEvent) => {
      if (event.pointerId !== pointer.id) return;
      pointer.dragging = false;
      pointer.id = -1;
      if (renderer.domElement.hasPointerCapture(event.pointerId)) {
        renderer.domElement.releasePointerCapture(event.pointerId);
      }
    };

    renderer.domElement.addEventListener("pointerdown", handlePointerDown);
    renderer.domElement.addEventListener("pointermove", handlePointerMove);
    renderer.domElement.addEventListener("pointerup", handlePointerUp);
    renderer.domElement.addEventListener("pointercancel", handlePointerUp);
    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    const splats = new SplatMesh({
      url,
      onLoad: async (mesh) => {
        const meta = await metaPromise;
        if (disposed) return;

        mesh.quaternion.identity();
        const bounds = mesh.getBoundingBox(true);
        const center = new THREE.Vector3();
        const size = new THREE.Vector3();
        bounds.getCenter(center);
        bounds.getSize(size);
        const maxDim = Math.max(size.x, size.y, size.z);
        const fitScale = Number.isFinite(maxDim) && maxDim > 0 ? 2.5 / maxDim : 1;
        mesh.scale.setScalar(fitScale);
        mesh.position.copy(center).multiplyScalar(-fitScale);

        if (meta) {
          const metaToSpzVector = createPlaycanvasMetaTransform(meta);
          const transformPoint = (values: [number, number, number]) =>
            (metaToSpzVector?.(values) ?? rdfToRubVector(values)).sub(center).multiplyScalar(fitScale);
          const transformDirection = (values: [number, number, number]) =>
            metaToSpzVector?.(values) ?? rdfToRubVector(values);
          const position = transformPoint(meta.position);
          const viewDirection =
            meta.camera_key === "official-preview"
              ? normalizeFiniteVector(transformDirection(meta.target))
              : normalizeFiniteVector(transformPoint(meta.target).sub(position));
          const up = (metaToSpzVector?.(meta.up) ?? rdfToRubVector(meta.up)).normalize();
          const target = viewDirection ? position.clone().add(viewDirection) : new THREE.Vector3();
          if (
            Number.isFinite(position.lengthSq()) &&
            Number.isFinite(target.lengthSq()) &&
            Number.isFinite(up.lengthSq()) &&
            up.lengthSq() > 0 &&
            viewDirection
          ) {
            camera.fov = THREE.MathUtils.clamp(meta.fov || 62, 35, 95);
            camera.up.copy(up);
            camera.position.copy(position);
            lookAtPoint.copy(target);
            lookDistance.value = 1;
            setCameraLook();
          } else {
            lookAtPoint.set(0, 0, 0);
            lookDistance.value = Math.max(camera.position.distanceTo(lookAtPoint), 0.25);
            setCameraLook();
          }
        } else {
          camera.up.set(0, -1, 0);
          camera.position.set(0, 0.25, 3.3);
          lookAtPoint.set(0, 0, 0);
          lookDistance.value = Math.max(camera.position.distanceTo(lookAtPoint), 0.25);
          setCameraLook();
        }
        camera.updateProjectionMatrix();
        camera.updateMatrixWorld();
        setLoadState("ready");
      },
      onProgress: (event) => {
        if (event.lengthComputable && event.total > 0) {
          setLoadState(`${Math.round((event.loaded / event.total) * 100)}%`);
        }
      },
    });
    scene.add(splats);

    const resize = () => {
      const bounds = host.getBoundingClientRect();
      const width = Math.max(320, Math.floor(bounds.width));
      const height = Math.max(280, Math.floor(bounds.height));
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    };

    const observer = new ResizeObserver(resize);
    observer.observe(host);
    resize();

    const clock = new THREE.Clock();
    renderer.setAnimationLoop(() => {
      const deltaSeconds = Math.min(clock.getDelta(), 0.05);
      applyKeyboardRotate(deltaSeconds);
      applyKeyboardMove(deltaSeconds);
      renderer.render(scene, camera);
    });

    return () => {
      disposed = true;
      observer.disconnect();
      renderer.setAnimationLoop(null);
      renderer.domElement.removeEventListener("pointerdown", handlePointerDown);
      renderer.domElement.removeEventListener("pointermove", handlePointerMove);
      renderer.domElement.removeEventListener("pointerup", handlePointerUp);
      renderer.domElement.removeEventListener("pointercancel", handlePointerUp);
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
      scene.remove(splats);
      scene.remove(spark);
      renderer.dispose();
      host.removeChild(renderer.domElement);
    };
  }, [jobId, url]);

  return (
    <section className="viewer-shell" aria-label="Spark viewer">
      <div className="viewer-toolbar">
        <div className="viewer-title">
          <Box size={18} />
          <span>SPZ Viewer</span>
        </div>
        <span className="viewer-state">{loadState}</span>
      </div>
      <div className="viewer-canvas" ref={hostRef}>
        {!url && (
          <div className="empty-viewer">
            <FileArchive size={36} />
            <span>Awaiting SPZ</span>
          </div>
        )}
      </div>
    </section>
  );
}

function StateIcon({ state }: { state: JobState }) {
  if (state === "ready") return <Play size={16} />;
  if (state === "succeeded") return <CheckCircle2 size={16} />;
  if (state === "failed") return <XCircle size={16} />;
  if (state === "canceled") return <Ban size={16} />;
  if (state === "running") return <Loader2 className="spin" size={16} />;
  return <Activity size={16} />;
}

function previewViewIndex(item: PreviewItem) {
  const match = item.id.match(/^(?:split-view|view)-(\d+)(?:-|$)/);
  return match ? Number(match[1]) : null;
}

function previewCustomRows(items: PreviewItem[]) {
  return Array.from(
    items.reduce((groups, item) => {
      if (!item.group_title) return groups;
      const existing = groups.get(item.group_title) ?? { items: [] as PreviewItem[], order: item.group_order ?? Number.MAX_SAFE_INTEGER };
      existing.items.push(item);
      existing.order = Math.min(existing.order, item.group_order ?? Number.MAX_SAFE_INTEGER);
      groups.set(item.group_title, existing);
      return groups;
    }, new Map<string, { items: PreviewItem[]; order: number }>())
  ).sort(([titleA, rowA], [titleB, rowB]) => rowA.order - rowB.order || titleA.localeCompare(titleB));
}

function PreviewImageLightbox({ item, onClose }: { item: PreviewItem; onClose: () => void }) {
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  if (!item.url) return null;

  return (
    <div className="image-lightbox" role="dialog" aria-modal="true" aria-label={item.title} onClick={onClose}>
      <div className="image-lightbox-content" onClick={(event) => event.stopPropagation()}>
        <div className="image-lightbox-header">
          <div>
            <strong>{item.title}</strong>
            <span>{item.stage}</span>
          </div>
          <button className="icon-button" type="button" onClick={onClose} aria-label="Close image preview" title="Close">
            <X size={17} />
          </button>
        </div>
        <div className="image-lightbox-frame">
          <img src={apiUrl(item.url)} alt={item.title} />
        </div>
        <small>{item.path}</small>
      </div>
    </div>
  );
}

function PreviewCard({ item, onImageOpen }: { item: PreviewItem; onImageOpen: (item: PreviewItem) => void }) {
  return (
    <article className={`preview-card ${item.available ? "available" : "pending"}`}>
      <div className="preview-media">
        {item.available && item.url && item.kind === "video" && (
          <video src={apiUrl(item.url)} controls muted loop playsInline preload="metadata" />
        )}
        {item.available && item.url && item.kind === "image" && (
          <button className="preview-image-button" type="button" onClick={() => onImageOpen(item)} aria-label={`Open ${item.title}`}>
            <img src={apiUrl(item.url)} alt={item.title} loading="lazy" />
          </button>
        )}
        {!item.available && <FileArchive size={26} />}
      </div>
      <div className="preview-meta">
        <div>
          <strong>{item.title}</strong>
          <span>{item.stage}</span>
        </div>
        <p>{item.description}</p>
        <small>{item.available ? item.path : "pending"}</small>
      </div>
    </article>
  );
}

function PreviewPanel({ items }: { items: PreviewItem[] }) {
  const [expandedImage, setExpandedImage] = useState<PreviewItem | null>(null);
  const availableItems = items.filter((item) => item.available && item.url);
  const overviewItems = items.filter((item) => previewViewIndex(item) === null && !item.group_title);
  const viewGroups = Array.from(
    items.reduce((groups, item) => {
      const viewIndex = previewViewIndex(item);
      if (viewIndex === null) return groups;
      const groupItems = groups.get(viewIndex) ?? [];
      groupItems.push(item);
      groups.set(viewIndex, groupItems);
      return groups;
    }, new Map<number, PreviewItem[]>())
  ).sort(([a], [b]) => a - b);
  const customRows = previewCustomRows(items);

  return (
    <section className="preview-panel" aria-label="Pipeline previews">
      <div className="panel-heading">
        <h2>Pipeline Preview</h2>
        <span>{availableItems.length}/{items.length || 0}</span>
      </div>
      <div className="preview-rows">
        {items.length === 0 && <div className="preview-empty">Previews will appear as stages finish.</div>}
        {overviewItems.length > 0 && (
          <div className="preview-row">
            <div className="preview-row-heading">
              <h3>Overview</h3>
              <span>{overviewItems.filter((item) => item.available).length}/{overviewItems.length}</span>
            </div>
            <div className="preview-strip">
              {overviewItems.map((item) => <PreviewCard key={item.id} item={item} onImageOpen={setExpandedImage} />)}
            </div>
          </div>
        )}
        {viewGroups.map(([viewIndex, groupItems]) => (
          <div className="preview-row" key={viewIndex}>
            <div className="preview-row-heading">
              <h3 title={`View ${viewIndex}`}>View {viewIndex}</h3>
              <span>{groupItems.filter((item) => item.available).length}/{groupItems.length}</span>
            </div>
            <div className="preview-strip">
              {groupItems.map((item) => <PreviewCard key={item.id} item={item} onImageOpen={setExpandedImage} />)}
            </div>
          </div>
        ))}
        {customRows.map(([title, row]) => (
          <div className="preview-row" key={title}>
            <div className="preview-row-heading">
              <h3 title={title}>{title}</h3>
              <span>{row.items.filter((item) => item.available).length}/{row.items.length}</span>
            </div>
            <div className="preview-strip">
              {row.items.map((item) => <PreviewCard key={item.id} item={item} onImageOpen={setExpandedImage} />)}
            </div>
          </div>
        ))}
      </div>
      {expandedImage && <PreviewImageLightbox item={expandedImage} onClose={() => setExpandedImage(null)} />}
    </section>
  );
}

function App() {
  const centerColumnRef = useRef<HTMLDivElement | null>(null);
  const logsRef = useRef<HTMLPreElement | null>(null);
  const liveLogChunkSeenRef = useRef(false);
  const selectedJobStateRef = useRef<JobState | null>(null);
  const { jobs, setJobs, refresh } = useJobs();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [activeJob, setActiveJob] = useState<Job | null>(null);
  const [logs, setLogs] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [previews, setPreviews] = useState<PreviewItem[]>([]);
  const [pipelineSettings, setPipelineSettings] = useState<PipelineSettings>(() => loadPipelineSettings());
  const [previewHeight, setPreviewHeight] = useState(330);
  const { splitViewNum, trajectoryModes, indoor, gsMaxSteps, applyNavTraj, worldNavWonderTopk, worldNavReconTopk } = pipelineSettings;

  const currentJob = activeJob ?? jobs.find((job) => job.id === selectedId) ?? jobs[0] ?? null;
  const llmDescription = currentJob?.prompt?.trim() ?? "";
  const promptSource = currentJob?.prompt_source ?? "unknown";
  const promptError = currentJob?.prompt_error?.trim() ?? "";
  const hasLlmDescription = promptSource === "llm" && Boolean(llmDescription);
  const promptPending = currentJob?.stage === "prompt synthesis" && !hasLlmDescription && !promptError;
  const llmDescriptionTitle = hasLlmDescription ? "Auto-generated LLM description" : "LLM description status";
  const llmDescriptionText = uploading
    ? "Uploading image..."
    : hasLlmDescription
      ? llmDescription
      : promptPending
        ? "Generating the scene description from the uploaded image..."
        : currentJob && promptSource === "not_started"
          ? "Click Start to generate the scene description."
          : currentJob && promptSource !== "unknown"
          ? `${promptError || "LLM did not produce a scene-specific description."} The pipeline is using a generic fallback prompt.`
          : "Upload an image, then click Start to generate the scene description.";
  const splatUrl = currentJob?.state === "succeeded" && currentJob.artifacts["point_cloud_7999.spz"]
    ? withCacheBuster(absoluteArtifactUrl(currentJob.artifacts["point_cloud_7999.spz"]), currentJob.updated_at)
    : null;
  const spzDownloadUrl = currentJob?.state === "succeeded" && currentJob.artifacts["point_cloud_7999.spz"]
    ? withCacheBuster(absoluteArtifactUrl(currentJob.artifacts["point_cloud_7999.spz"]), currentJob.updated_at)
    : null;

  const artifactLinks = useMemo(() => {
    const artifacts = currentJob?.artifacts ?? {};
    return [
      ["PlayCanvas PLY", "point_cloud_7999_playcanvas.ply", Box],
      ["PLY", "point_cloud_7999.ply", Box],
      ["Checkpoint", "ckpt_7999_rank0.pt", FileArchive],
      ["Log", "pipeline.log", FileText],
    ].filter(([, key]) => artifacts[key as string]);
  }, [currentJob]);

  useEffect(() => {
    if (!selectedId && jobs.length > 0) setSelectedId(jobs[0].id);
  }, [jobs, selectedId]);

  useEffect(() => {
    if (logsRef.current) {
      logsRef.current.scrollTop = logsRef.current.scrollHeight;
    }
  }, [logs]);

  useEffect(() => {
    window.localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(normalizePipelineSettings(pipelineSettings)));
  }, [pipelineSettings]);

  function updatePipelineSettings(update: Partial<PipelineSettings>) {
    setPipelineSettings((existing) => normalizePipelineSettings({ ...existing, ...update }));
  }

  function setTrajectoryCount(count: number) {
    setPipelineSettings((existing) => ({
      ...existing,
      trajectoryModes: normalizeTrajectoryModes(existing.trajectoryModes, count),
    }));
  }

  function setTrajectoryMode(index: number, mode: string) {
    setPipelineSettings((existing) => {
      const next = normalizeTrajectoryModes(existing.trajectoryModes);
      next[index] = mode;
      return { ...existing, trajectoryModes: next };
    });
  }

  async function refreshPreviews(jobId: string) {
    const response = await fetch(apiUrl(`/api/jobs/${jobId}/previews`));
    if (response.ok) setPreviews(await response.json());
  }

  useEffect(() => {
    if (!currentJob) return;
    setSelectedId(currentJob.id);
    setActiveJob(currentJob);
    setLogs("");
    liveLogChunkSeenRef.current = false;
    selectedJobStateRef.current = currentJob.state;
    setPreviews([]);
    refreshPreviews(currentJob.id);

    const source = new EventSource(apiUrl(`/api/jobs/${currentJob.id}/events`));
    const maybeRefreshPreviews = (job: Job) => {
      if (job.progress.startsWith("Finished") || ["succeeded", "failed", "canceled"].includes(job.state)) {
        refreshPreviews(job.id);
      }
    };
    const handleStatus = (job: Job) => {
      selectedJobStateRef.current = job.state;
      setActiveJob(job);
      maybeRefreshPreviews(job);
    };
    const handlePreviewUpdate = (payload: SseEvent) => {
      if (payload.job) {
        selectedJobStateRef.current = payload.job.state;
        setActiveJob(payload.job);
        setJobs((existing) => [payload.job as Job, ...existing.filter((job) => job.id !== payload.job?.id)]);
      }
      if (payload.previews) {
        setPreviews(payload.previews);
      } else if (payload.job) {
        refreshPreviews(payload.job.id);
      } else {
        refreshPreviews(currentJob.id);
      }
    };
    const handleLogLine = (line: string) => {
      if (liveLogChunkSeenRef.current) return;
      setLogs((existing) => appendLogText(existing, `${line}\n`));
    };
    const handleLogChunk = (chunk: string) => {
      liveLogChunkSeenRef.current = true;
      setLogs((existing) => appendLogText(existing, chunk));
    };
    source.onmessage = (event) => {
      const payload = JSON.parse(event.data) as SseEvent;
      if (payload.type === "status" && payload.job) {
        handleStatus(payload.job);
      }
      if (payload.type === "log" && payload.line !== undefined) {
        handleLogLine(payload.line);
      }
      if (payload.type === "log_chunk" && payload.chunk !== undefined) {
        handleLogChunk(payload.chunk);
      }
      if (payload.type === "preview_update") {
        handlePreviewUpdate(payload);
      }
    };
    source.addEventListener("status", (event) => {
      const payload = JSON.parse((event as MessageEvent).data) as SseEvent;
      if (payload.job) {
        handleStatus(payload.job);
        setJobs((existing) => [payload.job as Job, ...existing.filter((job) => job.id !== payload.job?.id)]);
      }
    });
    source.addEventListener("log", (event) => {
      const payload = JSON.parse((event as MessageEvent).data) as SseEvent;
      if (payload.line !== undefined) handleLogLine(payload.line);
    });
    source.addEventListener("log_chunk", (event) => {
      const payload = JSON.parse((event as MessageEvent).data) as SseEvent;
      if (payload.chunk !== undefined) handleLogChunk(payload.chunk);
    });
    source.addEventListener("preview_update", (event) => {
      const payload = JSON.parse((event as MessageEvent).data) as SseEvent;
      handlePreviewUpdate(payload);
    });
    const previewTimer = window.setInterval(() => {
      if (selectedJobStateRef.current && ["queued", "running"].includes(selectedJobStateRef.current)) {
        refreshPreviews(currentJob.id);
      }
    }, 3000);
    return () => {
      source.close();
      window.clearInterval(previewTimer);
    };
  }, [currentJob?.id]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!file) return;
    setUploading(true);
    setError(null);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("split_view_num", String(clampInteger(splitViewNum, DEFAULT_SPLIT_VIEW_NUM, 1, 8)));
    formData.append("trajectory_modes", normalizeTrajectoryModes(trajectoryModes).join(","));
    formData.append("indoor", String(indoor));
    formData.append("gs_max_steps", String(clampInteger(gsMaxSteps, DEFAULT_GS_MAX_STEPS, 100, 50000)));
    formData.append("apply_nav_traj", String(applyNavTraj));
    formData.append("world_nav_wonder_topk", String(worldNavWonderTopk));
    formData.append("world_nav_recon_topk", String(worldNavReconTopk));
    const response = await fetch(apiUrl("/api/jobs"), { method: "POST", body: formData });
    setUploading(false);
    if (!response.ok) {
      const body = await response.json().catch(() => ({ detail: response.statusText }));
      setError(body.detail ?? "Upload failed.");
      return;
    }
    const job = (await response.json()) as Job;
    setJobs((existing) => [job, ...existing]);
    setSelectedId(job.id);
    setActiveJob(job);
    setLogs("");
    liveLogChunkSeenRef.current = false;
    setPreviews([]);
    refreshPreviews(job.id);
  }

  async function startJob() {
    if (!currentJob) return;
    setError(null);
    const safeSplitViewNum = clampInteger(splitViewNum, DEFAULT_SPLIT_VIEW_NUM, 1, 8);
    const safeTrajectoryModes = normalizeTrajectoryModes(trajectoryModes);
    const safeGsMaxSteps = clampInteger(gsMaxSteps, DEFAULT_GS_MAX_STEPS, 100, 50000);
    const safeWorldNavWonderTopk = normalizeRequiredInteger(worldNavWonderTopk, DEFAULT_WORLD_NAV_WONDER_TOPK, 1, MAX_WORLD_NAV_TOPK);
    const safeWorldNavReconTopk = normalizeRequiredInteger(worldNavReconTopk, DEFAULT_WORLD_NAV_RECON_TOPK, 1, MAX_WORLD_NAV_TOPK);
    updatePipelineSettings({
      splitViewNum: safeSplitViewNum,
      trajectoryModes: safeTrajectoryModes,
      gsMaxSteps: safeGsMaxSteps,
      applyNavTraj,
      worldNavWonderTopk: safeWorldNavWonderTopk,
      worldNavReconTopk: safeWorldNavReconTopk,
    });
    const params = new URLSearchParams({
      split_view_num: String(safeSplitViewNum),
      trajectory_modes: safeTrajectoryModes.join(","),
      indoor: String(indoor),
      gs_max_steps: String(safeGsMaxSteps),
      apply_nav_traj: String(applyNavTraj),
      world_nav_wonder_topk: String(safeWorldNavWonderTopk),
      world_nav_recon_topk: String(safeWorldNavReconTopk),
    });
    const response = await fetch(apiUrl(`/api/jobs/${currentJob.id}/start?${params.toString()}`), { method: "POST" });
    if (!response.ok) {
      const body = await response.json().catch(() => ({ detail: response.statusText }));
      setError(body.detail ?? "Start failed.");
      return;
    }
    const job = (await response.json()) as Job;
    setActiveJob(job);
    setJobs((existing) => [job, ...existing.filter((item) => item.id !== job.id)]);
  }

  async function cancelJob() {
    if (!currentJob) return;
    const response = await fetch(apiUrl(`/api/jobs/${currentJob.id}/cancel`), { method: "POST" });
    if (response.ok) setActiveJob(await response.json());
  }

  function startPreviewResize(event: React.PointerEvent<HTMLDivElement>) {
    const column = centerColumnRef.current;
    if (!column) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    event.preventDefault();
    const bounds = column.getBoundingClientRect();
    const handlePointerMove = (moveEvent: PointerEvent) => {
      const nextHeight = bounds.bottom - moveEvent.clientY;
      const maxHeight = Math.max(220, bounds.height - 260);
      setPreviewHeight(Math.max(180, Math.min(maxHeight, nextHeight)));
    };
    const handlePointerUp = () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
      window.removeEventListener("pointercancel", handlePointerUp);
    };
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);
    window.addEventListener("pointercancel", handlePointerUp);
  }

  return (
    <main className="app">
      <section className="upload-panel">
        <div className="brand-line">
          <span className="brand-mark">HY</span>
          <div>
            <h1>HY-World 3DGS</h1>
            <p>Panorama to Spark SPZ</p>
          </div>
        </div>

        <form onSubmit={submit} className="upload-form">
          <label className="drop-zone">
            <input type="file" accept="image/*" onChange={(event) => setFile(event.target.files?.[0] ?? null)} />
            <ImageUp size={28} />
            <span>{file ? file.name : "Select panorama"}</span>
          </label>
          <button className="primary-button" type="submit" disabled={!file || uploading}>
            {uploading ? <Loader2 className="spin" size={17} /> : <UploadCloud size={17} />}
            <span>Upload</span>
          </button>
          <div className="llm-description-box" aria-live="polite">
            <span>{llmDescriptionTitle}</span>
            <p>{llmDescriptionText}</p>
          </div>
          <label className="field compact-field">
            <span>Scene splits</span>
            <input
              type="number"
              min={1}
              max={8}
              step={1}
              value={splitViewNum}
              onChange={(event) => updatePipelineSettings({ splitViewNum: Number(event.target.value) })}
            />
          </label>
          <div className="field trajectory-field">
            <span>Trajectories per view</span>
            <input
              type="number"
              min={1}
              max={MAX_TRAJECTORIES_PER_VIEW}
              step={1}
              value={trajectoryModes.length}
              onChange={(event) => setTrajectoryCount(Number(event.target.value))}
            />
            <div className="trajectory-mode-grid">
              {trajectoryModes.map((mode, index) => (
                <label key={index}>
                  <span>traj{index}</span>
                  <select value={mode} onChange={(event) => setTrajectoryMode(index, event.target.value)}>
                    {TRAJECTORY_MODE_OPTIONS.map(([value, label]) => (
                      <option key={value} value={value}>{label}</option>
                    ))}
                  </select>
                </label>
              ))}
            </div>
          </div>
          <label className="checkbox-field">
            <input type="checkbox" checked={indoor} onChange={(event) => updatePipelineSettings({ indoor: event.target.checked })} />
            <span>Indoor</span>
          </label>
          <label className="checkbox-field">
            <input
              type="checkbox"
              checked={applyNavTraj}
              onChange={(event) => updatePipelineSettings({ applyNavTraj: event.target.checked })}
            />
            <span>NavMesh navigation</span>
          </label>
          <label className="field compact-field">
            <span>Wonder top-k</span>
            <input
              type="number"
              min={1}
              max={MAX_WORLD_NAV_TOPK}
              step={1}
              value={worldNavWonderTopk}
              disabled={!applyNavTraj}
              onChange={(event) => updatePipelineSettings({
                worldNavWonderTopk: normalizeRequiredInteger(event.target.value, DEFAULT_WORLD_NAV_WONDER_TOPK, 1, MAX_WORLD_NAV_TOPK),
              })}
            />
          </label>
          <label className="field compact-field">
            <span>Recon top-k</span>
            <input
              type="number"
              min={1}
              max={MAX_WORLD_NAV_TOPK}
              step={1}
              value={worldNavReconTopk}
              disabled={!applyNavTraj}
              onChange={(event) => updatePipelineSettings({
                worldNavReconTopk: normalizeRequiredInteger(event.target.value, DEFAULT_WORLD_NAV_RECON_TOPK, 1, MAX_WORLD_NAV_TOPK),
              })}
            />
          </label>
          <label className="field compact-field">
            <span>GS steps</span>
            <input
              type="number"
              min={100}
              max={50000}
              step={100}
              value={gsMaxSteps}
              onChange={(event) => updatePipelineSettings({ gsMaxSteps: Number(event.target.value) })}
            />
          </label>
          {error && <div className="error-line">{error}</div>}
        </form>

        <div className="queue-panel">
          <div className="panel-heading">
            <h2>Queue</h2>
            <button className="icon-button" onClick={refresh} aria-label="Refresh jobs" title="Refresh jobs">
              <RotateCcw size={16} />
            </button>
          </div>
          <div className="job-list">
            {jobs.length === 0 && <div className="empty-row">No jobs</div>}
            {jobs.map((job) => (
              <button key={job.id} className={`job-row ${job.id === currentJob?.id ? "selected" : ""}`} onClick={() => setSelectedId(job.id)}>
                <StateIcon state={job.state} />
                <span>{job.id}</span>
                <strong>{job.state}</strong>
              </button>
            ))}
          </div>
        </div>
      </section>

      <section className="work-panel">
        <div className="status-strip">
          <div>
            <span className={`state-pill ${currentJob?.state ?? "idle"}`}>
              {currentJob ? <StateIcon state={currentJob.state} /> : <Play size={16} />}
              {currentJob?.state ?? "idle"}
            </span>
            <h2>{currentJob?.stage ?? "Ready"}</h2>
            <p>{currentJob?.progress ?? "Upload a panorama to start."}</p>
          </div>
          <div className="actions">
            {currentJob && STARTABLE_JOB_STATES.has(currentJob.state) && (
              <button className="primary-button compact-button" onClick={startJob}>
                <Play size={16} />
                <span>Start</span>
              </button>
            )}
            {currentJob && ["ready", "queued", "running"].includes(currentJob.state) && (
              <button className="secondary-button" onClick={cancelJob}>
                <Ban size={16} />
                <span>Cancel</span>
              </button>
            )}
            {spzDownloadUrl && (
              <a className="primary-button compact-button spz-download-button" href={spzDownloadUrl} download>
                <Download size={16} />
                <span>Download SPZ</span>
              </a>
            )}
            {artifactLinks.map(([label, key, Icon]) => (
              <a className="download-button" key={key as string} href={apiUrl(currentJob!.artifacts[key as string])} download>
                {React.createElement(Icon as typeof Download, { size: 16 })}
                <span>{label as string}</span>
              </a>
            ))}
          </div>
        </div>

        <div className="workspace-grid">
          <div
            className="center-column"
            ref={centerColumnRef}
            style={{ "--preview-height": `${previewHeight}px` } as React.CSSProperties}
          >
            <SparkViewer jobId={currentJob?.id ?? null} url={splatUrl} />
            <div className="preview-resizer" role="separator" aria-orientation="horizontal" title="Resize preview panel" onPointerDown={startPreviewResize}>
              <span />
            </div>
            <PreviewPanel items={previews} />
          </div>
          <section className="logs-panel" aria-label="Pipeline logs">
            <div className="panel-heading">
              <h2>Logs</h2>
              <span>latest</span>
            </div>
            <pre ref={logsRef}>{logs || "Logs will stream here."}</pre>
          </section>
        </div>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
