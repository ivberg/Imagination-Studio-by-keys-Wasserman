"""Bounded sequence tests: fake model parts and isolated CPU media, never real model renders."""
import json
import os
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from test_windows_ui import client, server  # noqa: F401
import windows_checks
import windows_video_vsa as vsa


def params(**changes):
    return {"image": "reference.png", "prompt": "The person in <Picture 1> smiles.",
            "seed": 7, "frames": 73, "sequence_seconds": 60, **changes}


@pytest.mark.parametrize("seconds", [0, 61, True, 1.5, "60", None])
def test_sequence_duration_is_bounded_and_typed(seconds):
    with pytest.raises(ValueError, match="whole number"):
        vsa.parameters(params(sequence_seconds=seconds))


def test_sequence_planning_keeps_model_valid_parts_and_exact_target():
    assert vsa.sequence_count(73, 60) == 20
    assert vsa.sequence_count(22, 60) == 66
    assert vsa.sequence_count(362, 1) == 1
    assert vsa.parameters(params(sequence_seconds=1))["sequence_seconds"] == 1


def test_sequence_uses_existing_native_job_and_bounded_deadline(tmp_path, monkeypatch):
    monkeypatch.setattr(vsa, "require", vsa.parameters)
    command, env, outputs = vsa.build(tmp_path, params(), "test")
    assert command[command.index("--sequence-seconds") + 1] == "60"
    assert command[command.index("--frames") + 1] == "73"
    assert not env and [Path(path).name for path in outputs] == ["video.mp4", "sequence.json"]
    job = {"kind": vsa.KIND, "params": params()}
    assert windows_checks.gpu_check(job) and windows_checks.native_image_check(job)
    assert windows_checks.timeout_seconds(job) == vsa.SEQUENCE_TIMEOUT_S + 180
    assert windows_checks.timeout_seconds({"kind": vsa.KIND, "params": {}}) == vsa.TIMEOUT_S + 180


def test_job_api_preserves_sequence_settings(client, monkeypatch):
    monkeypatch.setattr(server, "enqueue", lambda kind, title, supplied: {"kind": kind, "params": vsa.parameters(supplied)})
    response = client.post("/api/jobs", json={"kind": vsa.KIND, "params": params()})
    assert response.status_code == 200 and response.json()["params"]["sequence_seconds"] == 60


def fake_parts(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(vsa, "require", vsa.parameters)
    monkeypatch.setattr(vsa.library, "studio_home", lambda: tmp_path)
    monkeypatch.setattr(vsa.windows_runtime, "release_native", lambda: calls.append(("release",)))
    def admission(frames, observations, timeout_s):
        assert 1 <= timeout_s <= vsa.ADMISSION_TIMEOUT_S
        observations.update(state="admitted", samples=2)
        calls.append(("admission", frames))
    monkeypatch.setattr(vsa, "wait_for_memory", admission)
    monkeypatch.setattr(vsa.features, "save_pass", lambda *args: pytest.fail("A sequence is not qualification evidence"))
    def reference(clip, image, timeout_s):
        assert clip.is_file() and timeout_s <= 120
        image.write_bytes(b"synthetic reference")
        calls.append(("reference", clip, image))
    monkeypatch.setattr(vsa, "next_reference", reference)
    def part(output, image, prompt, seed, observations, frames=22, timeout_s=1800):
        calls.append(("part", image, prompt, seed, frames, timeout_s))
        output.mkdir(parents=True)
        (output / "video.mp4").write_bytes(b"synthetic part")
        observations.update(rendered=True, case=f"fixture {frames} frames")
    monkeypatch.setattr(vsa, "run_case", part)
    def assemble(output, clips, frames, seconds, observations, timeout_s):
        calls.append(("assemble", list(clips), frames, seconds))
        assert all(clip.is_file() for clip in clips) and timeout_s <= vsa.ASSEMBLY_TIMEOUT_S
        (output / "video.mp4").write_bytes(b"synthetic assembly")
    monkeypatch.setattr(vsa, "assemble_sequence", assemble)
    return calls, part


def test_sequence_is_serial_cold_and_keeps_prompt_with_wrapped_seeds(tmp_path, monkeypatch):
    calls, _ = fake_parts(tmp_path, monkeypatch)
    result = vsa.run_sequence(tmp_path / "sequence", params(frames=22, sequence_seconds=2, seed=2 ** 32 - 1))
    parts = [call for call in calls if call[0] == "part"]
    assert [call[0] for call in calls] == ["release", "admission", "part", "release", "reference", "admission", "part",
                                          "release", "reference", "admission", "part", "release", "assemble"]
    assert [call[3] for call in parts] == [2 ** 32 - 1, 0, 1]
    assert all(call[2] == params()["prompt"] and call[4] == 22 for call in parts)
    assert result["state"] == "done" and result["target_frames"] == 48 and len(result["parts"]) == 3
    saved = json.loads((tmp_path / "sequence" / "sequence.json").read_text())
    assert saved["state"] == "done" and "active_part" not in saved
    assert not (tmp_path / "sequence" / "sequence.json.tmp").exists()
    assert not list((tmp_path / "sequence").glob("reference-*.png"))


def test_sequence_stops_on_guard_failure_and_preserves_completed_parts(tmp_path, monkeypatch):
    calls, part = fake_parts(tmp_path, monkeypatch)
    def fail_second(output, image, prompt, seed, observations, **kwargs):
        if seed == 8:
            observations["refused_sample"] = {"dedicated_gpu_free_mb": 1900}
            raise vsa.native.SafetyRefusal("memory floor")
        return part(output, image, prompt, seed, observations, **kwargs)
    monkeypatch.setattr(vsa, "run_case", fail_second)
    output = tmp_path / "sequence"
    with pytest.raises(vsa.native.SafetyRefusal, match="memory floor"):
        vsa.run_sequence(output, params(frames=22, sequence_seconds=2))
    saved = json.loads((output / "sequence.json").read_text())
    assert saved["state"] == "failed" and saved["active_part"] == 2 and len(saved["parts"]) == 1
    assert saved["failed_part_observations"]["refused_sample"]["dedicated_gpu_free_mb"] == 1900
    assert (output / "parts" / "001" / "video.mp4").exists() and not (output / "video.mp4").exists()
    assert not any(call[0] == "assemble" for call in calls)


def test_sequence_waits_after_reference_extraction_and_never_renders_if_recovery_fails(tmp_path, monkeypatch):
    calls, _ = fake_parts(tmp_path, monkeypatch)
    def wait(frames, observations, timeout_s):
        calls.append(("admission", frames))
        observations.update(samples=2, state="admitted" if len([c for c in calls if c[0] == "part"]) == 0 else "failed")
        if observations["state"] == "failed":
            raise vsa.native.SafetyRefusal("memory did not recover")
    monkeypatch.setattr(vsa, "wait_for_memory", wait)
    output = tmp_path / "sequence"
    with pytest.raises(vsa.native.SafetyRefusal, match="did not recover"):
        vsa.run_sequence(output, params(frames=158, sequence_seconds=8))
    assert [call[0] for call in calls] == ["release", "admission", "part", "release", "reference", "admission"]
    saved = json.loads((output / "sequence.json").read_text())
    assert saved["failed_part_observations"]["admission"]["state"] == "failed"
    assert saved["active_part_observations"]["admission"]["samples"] == 2
    assert len(saved["parts"]) == 1 and not (output / "video.mp4").exists()
    assert not list(output.glob("reference-*.png"))


@pytest.fixture
def recovery(monkeypatch):
    import windows_memory
    from test_windows_video_vsa import snapshot
    now = [100.0]
    reads = []
    samples = []
    fits = []
    def read(*args):
        reads.append(now[0])
        sample = samples.pop(0) if samples else snapshot()
        return replace(sample, at=now[0]) if sample.at == 1.0 else sample
    class Signals:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(windows_memory, "WindowsSignals", Signals)
    monkeypatch.setattr(windows_memory, "physical_memory_mb", lambda: 65536)
    monkeypatch.setattr(windows_memory, "WddmMemory", lambda _: None)
    monkeypatch.setattr(windows_memory, "dedicated_snapshot", read)
    monkeypatch.setattr(vsa.native, "device_uuid", lambda: "synthetic-device")
    def fit(sample, frames):
        reads[-1] = (reads[-1], frames)
        return fits.pop(0) if fits else {"fits": True, "reason": ""}
    monkeypatch.setattr(vsa, "memory_fit", fit)
    monkeypatch.setattr(vsa, "time", SimpleNamespace(monotonic=lambda: now[0],
                                                   sleep=lambda seconds: now.__setitem__(0, now[0] + seconds)))
    return SimpleNamespace(now=now, reads=reads, fits=fits, samples=samples, sample=snapshot)


def test_recovery_requires_two_consecutive_fits_after_delayed_memory_release(recovery):
    recovery.fits.extend([{"fits": False, "reason": "host memory short"},
                          {"fits": True, "reason": ""},
                          {"fits": False, "reason": "commit headroom short"}])
    observations = {}
    vsa.wait_for_memory(158, observations, 10)
    assert observations["state"] == "admitted" and observations["samples"] == 5
    assert observations["consecutive_fits"] == 2 and observations["elapsed_s"] == 4
    assert observations["initial"]["at"] == 100 and observations["final"]["at"] == 104
    assert recovery.reads == [(100 + i, 158) for i in range(5)]


def test_persistent_contention_times_out_with_exact_reason_and_no_policy_relaxation(recovery):
    recovery.fits.extend([{"fits": False, "reason": "needs 12844 MiB host growth plus 2048 reserve"}] * 5)
    observations = {}
    with pytest.raises(vsa.native.SafetyRefusal, match="needs 12844 MiB host growth plus 2048 reserve"):
        vsa.wait_for_memory(158, observations, 3)
    assert observations["state"] == "failed" and observations["samples"] == 3
    assert observations["elapsed_s"] == 3 and observations["consecutive_fits"] == 0


def test_recovery_can_admit_when_only_the_historical_gpu_prediction_is_short(recovery):
    recovery.fits.extend([{"fits": True, "reason": "", "gpu_prediction": {
        "advisory": True, "fits": False, "growth_mb": 25925, "available_mb": 27626, "floor_mb": 2048}}] * 2)
    observations = {}
    vsa.wait_for_memory(158, observations, 10)
    assert observations["state"] == "admitted" and observations["samples"] == 2
    assert not observations["admission"]["gpu_prediction"]["fits"]


@pytest.mark.parametrize("mode", ["stale", "future", "host", "commit", "gpu", "missing"])
def test_recovery_fails_closed_immediately_on_invalid_telemetry_or_live_floor(recovery, monkeypatch, mode):
    sample = recovery.sample()
    if mode == "stale":
        sample = replace(sample, at=97)
    elif mode == "future":
        sample = replace(sample, at=101)
    elif mode == "host":
        sample = replace(sample, host_low_memory=True)
    elif mode == "commit":
        sample = replace(sample, host_commit_used_mb=sample.host_commit_limit_mb - 1)
    elif mode == "gpu":
        sample = replace(sample, gpu_dedicated_free_mb=1000)
    else:
        import windows_memory
        def missing(*args):
            raise RuntimeError("WDDM memory counter is missing")
        monkeypatch.setattr(windows_memory, "dedicated_snapshot", missing)
    recovery.samples.append(sample)
    observations = {}
    with pytest.raises((vsa.native.SafetyRefusal, RuntimeError)):
        vsa.wait_for_memory(158, observations, 10)
    assert observations["state"] == "failed" and observations["elapsed_s"] == 0


def test_recovery_never_uses_a_fit_computed_after_its_deadline(recovery, monkeypatch):
    def slow(*args):
        recovery.now[0] += 5
        return {"fits": True, "reason": ""}
    monkeypatch.setattr(vsa, "memory_fit", slow)
    observations = {}
    with pytest.raises(vsa.native.SafetyRefusal, match="No stable fitting samples"):
        vsa.wait_for_memory(158, observations, 3)
    assert observations["state"] == "failed"


def test_recovery_refuses_a_sample_that_became_stale_while_checking_evidence(recovery, monkeypatch):
    def slow(*args):
        recovery.now[0] += 3
        return {"fits": True, "reason": ""}
    monkeypatch.setattr(vsa, "memory_fit", slow)
    with pytest.raises(vsa.native.SafetyRefusal, match="became stale"):
        vsa.wait_for_memory(158, {}, 10)


@pytest.mark.parametrize("seconds", [0, 61, True, 1.5])
def test_recovery_timeout_is_bounded(recovery, seconds):
    with pytest.raises(ValueError, match="deadline"):
        vsa.wait_for_memory(158, {}, seconds)


def test_interrupted_sequence_keeps_partial_manifest_without_publishing_a_final(tmp_path, monkeypatch):
    fake_parts(tmp_path, monkeypatch)
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(vsa, "run_case", interrupted)
    output = tmp_path / "sequence"
    with pytest.raises(KeyboardInterrupt):
        vsa.run_sequence(output, params())
    saved = json.loads((output / "sequence.json").read_text())
    assert saved["state"] == "partial" and saved["active_part"] == 1
    assert not (output / "video.mp4").exists()


def test_sequence_deadline_stops_before_starting_a_part(tmp_path, monkeypatch):
    calls, _ = fake_parts(tmp_path, monkeypatch)
    now = iter((0,))
    monkeypatch.setattr(vsa.time, "monotonic", lambda: next(now, vsa.SEQUENCE_TIMEOUT_S + 1))
    with pytest.raises(vsa.native.SafetyRefusal, match="two-hour deadline"):
        vsa.run_sequence(tmp_path / "sequence", params())
    assert not any(call[0] in ("part", "assemble") for call in calls)


def test_sequence_refuses_existing_output_without_overwriting(tmp_path, monkeypatch):
    fake_parts(tmp_path, monkeypatch)
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "video.mp4"
    marker.write_bytes(b"keep")
    with pytest.raises(RuntimeError, match="never overwritten"):
        vsa.run_sequence(output, params())
    assert marker.read_bytes() == b"keep"


def test_reference_extraction_is_bounded_and_checks_the_result(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(vsa.library, "tool", lambda name: name)
    monkeypatch.setattr(vsa.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    monkeypatch.setattr(vsa, "require_image", lambda path: calls.append(path))
    image = tmp_path / "reference.png"
    vsa.next_reference(tmp_path / "part.mp4", image, 17)
    command, options = calls[0]
    assert "-sseof" in command and "-update" in command and "force_original_aspect_ratio=decrease" in command[command.index("-vf") + 1]
    assert options["timeout"] == 17 and options["check"] and calls[1] == image


def test_reference_extraction_reports_ffmpeg_error(tmp_path, monkeypatch):
    monkeypatch.setattr(vsa.library, "tool", lambda name: name)
    def refuse(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, stderr="invalid media")
    monkeypatch.setattr(vsa.subprocess, "run", refuse)
    with pytest.raises(RuntimeError, match="reference extraction failed: invalid media"):
        vsa.next_reference(tmp_path / "part.mp4", tmp_path / "reference.png")


def test_assembly_failure_does_not_publish_unvalidated_video(tmp_path, monkeypatch):
    clip = tmp_path / "part.mp4"
    clip.write_bytes(b"fixture")
    monkeypatch.setattr(vsa.library, "tool", lambda name: name)
    def refuse(command, output, validate, observations, timeout_s, **kwargs):
        assert kwargs["dedicated_floor_mb"] == 2048 and kwargs["pressure_policy"] is vsa.pressure
        (output / "assembling.mp4").write_bytes(b"unvalidated")
        raise vsa.native.SafetyRefusal("assembly guard")
    monkeypatch.setattr(vsa.native, "observe_native", refuse)
    with pytest.raises(vsa.native.SafetyRefusal, match="assembly guard"):
        vsa.assemble_sequence(tmp_path, [clip], 73, 1, {})
    assert not (tmp_path / "video.mp4").exists() and not (tmp_path / "assembling.mp4").exists()


def test_cpu_assembly_produces_exact_frames_and_sound(tmp_path, monkeypatch):
    from PIL import Image
    import windows_media
    path = windows_media.search_path(os.environ.get("PATH", ""))
    ffmpeg, ffprobe = (shutil.which(name, path=path) for name in ("ffmpeg", "ffprobe"))
    if not ffmpeg or not ffprobe:
        pytest.skip("CPU media assembly test needs ffmpeg and ffprobe")
    monkeypatch.setenv("PATH", path)
    monkeypatch.setattr(vsa.library, "tool", lambda name: {"ffmpeg": ffmpeg, "ffprobe": ffprobe}[name])
    clips = []
    for index in range(2):
        clip = tmp_path / f"part{index}.mp4"
        subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=864x480:rate=24",
                        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", str(22 / 24),
                        "-frames:v", "22", "-c:v", "libx264", "-preset", "ultrafast", "-threads", "2",
                        "-c:a", "aac", str(clip)], capture_output=True, check=True, timeout=30)
        clips.append(clip)
    reference, expected = tmp_path / "reference.png", tmp_path / "last-frame.png"
    vsa.next_reference(clips[0], reference)
    subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-threads", "2", "-i", str(clips[0]), "-vf",
                    "select=eq(n\\,21),scale=512:288:force_original_aspect_ratio=decrease",
                    "-frames:v", "1", str(expected)], capture_output=True, check=True, timeout=30)
    with Image.open(reference) as actual, Image.open(expected) as last:
        assert actual.width <= 512 and actual.height <= 288 and actual.tobytes() == last.tobytes()
    def observe(command, output, validate, observations, timeout_s, **kwargs):
        subprocess.run(command, capture_output=True, check=True, timeout=30)
        validate()
        observations["rendered"] = True
    monkeypatch.setattr(vsa.native, "observe_native", observe)
    observations = {}
    vsa.assemble_sequence(tmp_path, clips, 22, 1, observations)
    vsa.check_media(tmp_path / "video.mp4", 24)
    assert observations["rendered"] and not (tmp_path / "assembling.mp4").exists()


def test_sequence_ui_selects_checked_parts_and_submits_a_bounded_job():
    from test_shell_page import run
    page = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")
    controls = page[page.index("function vsaLengths()"):page.index("function viewCreate()")]
    out = run("globalThis.VSA = {image:'reference.png',prompt:'Use <Picture 1>.',seed:7,frames:22,mode:'sequence',sequenceSeconds:60};\n" + controls + """
      S.windows_vsa = {lengths:[{frames:22,seconds:22/24,qualified:true,available:true,installed:true},
                               {frames:73,seconds:73/24,qualified:true,available:true,installed:true},
                               {frames:124,seconds:124/24,qualified:false,available:false,installed:true}]};
      globalThis.select = (obj,key,options) => el("select", {}, Object.values(options));
      globalThis.choice = () => el("div"); globalThis.bind = (obj,key,node) => node;
      globalThis.typed = node => node; globalThis.jobCard = () => el("div");
      vsaModeChanged(); out.frames = VSA.frames;
      const root = viewVsa(), start = button(root, "Render 60-second sequence");
      out.enabled = !start.attrs.disabled; out.summary = text(root);
      answers["/api/jobs"] = {id:"test"}; await start.attrs.onclick();
      out.params = calls.find(([path]) => path === "/api/jobs")[1].params;
      VSA.sequenceSeconds = 61; out.overLimit = button(viewVsa(), "Render 61-second sequence").attrs.disabled;
      VSA.mode = "single"; out.single = text(viewVsa());
      jobs.push({id:"test",status:"done",outputs:["renders/test/video.mp4","renders/test/sequence.json"]});
      out.finished = text(viewVsa());
    """)
    assert out["frames"] == 73 and out["enabled"] and out["overLimit"]
    assert out["params"]["sequence_seconds"] == 60 and out["params"]["frames"] == 73
    assert "20 cold-start shots" in out["summary"] and "not a guaranteed first-frame anchor" in out["summary"]
    assert "Render 73 frames" in out["single"]
    assert "Download video" in out["finished"] and "Sequence report" in out["finished"]
