"""VSA reference preparation is bounded CPU work on temporary pictures, never a model render."""
import shutil
import subprocess

import pytest
from PIL import Image

from test_windows_ui import client, server  # noqa: F401


@pytest.fixture
def pictures(tmp_path, monkeypatch):
    home = tmp_path / "studio"
    (home / "uploads").mkdir(parents=True)
    monkeypatch.setattr(server, "HOME", home)

    def make(size, name="original.png"):
        path = home / "uploads" / name
        Image.new("RGB", size, (64, 128, 192)).save(path)
        return path

    return home, make


@pytest.mark.parametrize("size,ready", [((512, 288), True), ((128, 128), True), ((864, 480), False), ((288, 512), False)])
def test_inspection_is_read_only_and_does_not_run_tools(client, pictures, monkeypatch, size, ready):
    home, make = pictures
    original = make(size)
    before = original.read_bytes()
    monkeypatch.setattr(server.subprocess, "run", lambda *args, **kwargs: pytest.fail("Inspection must not launch a tool"))
    response = client.post("/api/video-vsa/reference", json={"path": "uploads/original.png"})
    assert response.status_code == 200
    assert response.json() == {"path": "uploads/original.png", "width": size[0], "height": size[1],
                               "ready": ready, "resized": False}
    assert original.read_bytes() == before and list((home / "uploads").iterdir()) == [original]


@pytest.mark.parametrize("size,expected", [((1920, 1080), (512, 288)), ((600, 1200), (144, 288)),
                                          ((600, 600), (288, 288)), ((128, 128), (128, 128))])
def test_explicit_resize_makes_a_new_aspect_preserving_copy(client, pictures, size, expected):
    if not shutil.which("ffmpeg"):
        pytest.skip("CPU copy integration needs ffmpeg")
    home, make = pictures
    original = make(size)
    before = original.read_bytes()
    response = client.post("/api/video-vsa/reference", json={"path": "uploads/original.png", "resize": True})
    assert response.status_code == 200, response.text
    result = response.json()
    target = home / result["path"]
    assert result["ready"] and result["resized"] and result["source"] == "uploads/original.png"
    assert target != original and target.parent == home / "uploads"
    with Image.open(target) as copy:
        assert copy.size == expected
        copy.verify()
    assert original.read_bytes() == before


@pytest.mark.parametrize("body", [
    {"path": "../outside.png"}, {"path": ".private/picture.png"}, {"path": "uploads/missing.png"},
    {"path": "uploads/original.png", "resize": "yes"}, {"path": "uploads/original.png", "width": 1024},
    {"path": 42}, {},
])
def test_reference_preparation_refuses_invalid_paths_and_unbounded_options(client, pictures, body):
    _, make = pictures
    make((32, 32))
    response = client.post("/api/video-vsa/reference", json=body)
    assert response.status_code == 400


def test_unreadable_picture_fails_explicitly(client, pictures):
    home, _ = pictures
    (home / "uploads" / "bad.png").write_bytes(b"not an image")
    response = client.post("/api/video-vsa/reference", json={"path": "uploads/bad.png"})
    assert response.status_code == 400 and "could not be read" in response.json()["error"]


def test_missing_ffmpeg_blocks_copy_but_not_inspection(client, pictures, monkeypatch):
    _, make = pictures
    make((864, 480))
    monkeypatch.setattr(server, "media_tools", lambda: {"ffmpeg": False})
    assert client.post("/api/video-vsa/reference", json={"path": "uploads/original.png"}).status_code == 200
    response = client.post("/api/video-vsa/reference", json={"path": "uploads/original.png", "resize": True})
    assert response.status_code == 400 and "needs ffmpeg" in response.json()["error"]


@pytest.mark.parametrize("failure", ["timeout", "tool", "invalid-copy"])
def test_failed_copy_is_removed_and_original_preserved(client, pictures, monkeypatch, failure):
    home, make = pictures
    original = make((864, 480))
    before = original.read_bytes()
    monkeypatch.setattr(server, "media_tools", lambda: {"ffmpeg": True})

    def run(command, **kwargs):
        assert kwargs["timeout"] == 30 and kwargs["check"]
        target = home / "uploads" / command[-1].split("\\")[-1].split("/")[-1]
        target.write_bytes(b"partial image")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if failure == "tool":
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(server.subprocess, "run", run)
    response = client.post("/api/video-vsa/reference", json={"path": "uploads/original.png", "resize": True})
    assert response.status_code == 400 and "original is unchanged" in response.json()["error"]
    assert original.read_bytes() == before and list((home / "uploads").iterdir()) == [original]


def test_recipe_preparation_is_windows_only(client, pictures, monkeypatch):
    monkeypatch.setattr(server, "WINDOWS", False)
    monkeypatch.setattr(server, "WIN32", False)
    response = client.post("/api/video-vsa/reference", json={"path": "uploads/original.png"})
    assert response.status_code == 501
