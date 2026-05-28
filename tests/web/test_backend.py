import json
from pathlib import Path

import pytest
from PIL import Image

from web.backend.main import Job, JobStore, TARGET_SIZE, artifact_path, prepare_job_files, utc_now


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
    with pytest.raises(KeyError):
        artifact_path(tmp_path, "missing.zip")
