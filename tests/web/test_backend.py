import json
from pathlib import Path

import pytest
from PIL import Image

from web.backend.main import (
    AUTO_PROMPT_FALLBACK,
    Job,
    JobStore,
    TARGET_SIZE,
    artifact_path,
    is_generic_generated_prompt,
    is_placeholder_prompt,
    normalize_generated_prompt,
    pipeline_commands,
    prepare_job_files,
    preview_items,
    synthesize_prompt_details,
    synthesize_prompt_from_image,
    task_hash,
    utc_now,
)


def make_image_bytes(size=(32, 16), color=(90, 120, 150)):
    path = Path("/tmp/hyworld_test_upload.png")
    Image.new("RGB", size, color).save(path)
    return path.read_bytes()


def test_prepare_job_files_resizes_without_padding(tmp_path):
    prepare_job_files(tmp_path / "job", "pano.png", make_image_bytes(size=(64, 32)), "uploaded panorama")

    panorama = Image.open(tmp_path / "job" / "panorama.png")
    meta = json.loads((tmp_path / "job" / "meta_info.json").read_text(encoding="utf-8"))

    assert panorama.size == TARGET_SIZE
    assert meta == {"scene_type": "outdoor", "prompt": "uploaded panorama"}


def test_prepare_job_files_rejects_non_images(tmp_path):
    with pytest.raises(ValueError, match="readable image"):
        prepare_job_files(tmp_path / "job", "not-image.txt", b"plain text", "prompt")


def test_generated_prompt_normalization_compacts_and_limits_text():
    prompt = normalize_generated_prompt("  A palace courtyard\n with warm light.  ")

    assert prompt == "A palace courtyard with warm light."
    assert len(normalize_generated_prompt("x" * 800)) == 700
    assert normalize_generated_prompt("   ") == AUTO_PROMPT_FALLBACK


def test_synthesize_prompt_can_be_disabled(monkeypatch):
    monkeypatch.setenv("HYWORLD_AUTO_PROMPT", "0")

    assert synthesize_prompt_from_image(make_image_bytes()) == AUTO_PROMPT_FALLBACK
    result = synthesize_prompt_details(make_image_bytes())
    assert result.prompt == AUTO_PROMPT_FALLBACK
    assert result.source == "disabled"
    assert "disabled" in (result.error or "")


def test_generic_prompt_detector_rejects_template_fallback():
    assert is_generic_generated_prompt(AUTO_PROMPT_FALLBACK)
    assert not is_generic_generated_prompt("A narrow stone alley with red lanterns, wet pavement, wooden doors, and warm shop lights under a dusky blue sky.")
    assert is_placeholder_prompt("an ancient Chinese Palace")


def test_job_store_reloads_latest_state(tmp_path):
    store_path = tmp_path / "jobs.jsonl"
    store = JobStore(store_path)
    now = utc_now()
    job = Job("job1", "queued", "queued", "waiting", "prompt", str(tmp_path / "job1"), now, now)
    store.save(job)
    job.state = "succeeded"
    job.stage = "complete"
    store.save(job)

    reloaded = JobStore(store_path)
    reloaded.load()

    assert reloaded.jobs["job1"].state == "succeeded"
    assert reloaded.jobs["job1"].stage == "complete"


def test_artifact_path_known_and_unknown(tmp_path):
    path, media_type = artifact_path(tmp_path, "point_cloud_7999.spz")
    assert path == tmp_path / "gs_result/ply/point_cloud_7999.spz"
    assert media_type == "application/octet-stream"
    playcanvas_path, _ = artifact_path(tmp_path, "point_cloud_7999_playcanvas.ply")
    assert playcanvas_path == tmp_path / "gs_result/ply/point_cloud_7999_playcanvas.ply"
    with pytest.raises(KeyError):
        artifact_path(tmp_path, "missing.zip")


def test_apply_nav_traj_changes_hash_and_command(tmp_path):
    data = make_image_bytes()
    digest_regular, _, params_regular = task_hash(data, 4, ["forward"], False, 8000, False)
    digest_nav, _, params_nav = task_hash(data, 4, ["forward"], False, 8000, True)
    digest_nav_topk, _, params_nav_topk = task_hash(data, 4, ["forward"], False, 8000, True, 2, 6)

    assert digest_regular != digest_nav
    assert digest_nav != digest_nav_topk
    assert params_regular["apply_nav_traj"] is False
    assert params_nav["apply_nav_traj"] is True
    assert "world_nav_attempts" not in params_regular
    assert "world_nav_wonder_topk" not in params_regular
    assert params_nav["world_nav_wonder_topk"] == 3
    assert params_nav["world_nav_recon_topk"] == 5
    assert params_nav_topk["world_nav_wonder_topk"] == 2
    assert params_nav_topk["world_nav_recon_topk"] == 6

    regular_command = pipeline_commands(tmp_path, "prompt", 4, ["forward"], 8000, False)[0][2]
    nav_command = pipeline_commands(tmp_path, "prompt", 4, ["forward"], 8000, True)[0][2]
    nav_topk_command = pipeline_commands(tmp_path, "prompt", 4, ["forward"], 8000, True, 2, 6)[0][2]
    gs_training_command = pipeline_commands(tmp_path, "prompt", 4, ["forward"], 3000, False)[-1][2]

    assert "--apply_nav_traj" not in regular_command
    assert "--apply_nav_traj" in nav_command
    assert nav_command[-4:] == ["--wonder_topk", "3", "--recon_topk", "5"]
    assert nav_topk_command[-4:] == ["--wonder_topk", "2", "--recon_topk", "6"]
    assert "--save_ply" in gs_training_command
    assert "--convert_to_spz" in gs_training_command
    assert gs_training_command[gs_training_command.index("--ply_steps") + 1] == "3000"


def test_preview_items_include_worldnav_rows_when_enabled(tmp_path):
    run_dir = tmp_path / "job"
    nav_traj_dir = run_dir / "render_results" / "target_green_chair_0" / "traj0"
    nav_traj_dir.mkdir(parents=True)
    (nav_traj_dir.parent / "start_frame.png").write_bytes(b"image")
    (nav_traj_dir / "traj_vis.png").write_bytes(b"image")
    (nav_traj_dir / "render.mp4").write_bytes(b"video")

    without_nav = preview_items("job1", run_dir, 1, ["forward"], apply_nav_traj=False)
    with_nav = preview_items("job1", run_dir, 1, ["forward"], apply_nav_traj=True)

    assert not any(item.get("group_title") for item in without_nav)
    nav_items = [item for item in with_nav if item.get("group_title") == "WorldNav target green chair 0"]
    assert [item["title"] for item in nav_items[:3]] == [
        "WorldNav target green chair 0 start",
        "WorldNav target green chair 0 path",
        "WorldNav target green chair 0 render",
    ]
    assert all(item["group_order"] == 0 for item in nav_items)
