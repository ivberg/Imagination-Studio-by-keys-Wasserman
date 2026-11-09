"""Experimental routing, recipe and admission tests; no CUDA contexts or real model work."""
from dataclasses import replace
from pathlib import Path

import pytest

from test_windows_ui import client, server  # noqa: F401
import windows_checks
import windows_runtime
import windows_video_vsa as vsa
from windows_memory import Snapshot


def snapshot():
    return Snapshot(65536, 32460, 15000, 31232, 29000, 1.0, 28000, 68000, 1000, False, 31232, 29000)


def receipt():
    return {"ok": True, "checked_at": "2026-01-01T00:00:00+00:00", "results": [{
        "rendered": True, "measurements": {"samples": 10, "refused_sample": None,
            "initial": {"physical_total_mb": 65536, "host_total_mb": 32460, "host_free_mb": 15000,
                        "gpu_dedicated_total_mb": 31232, "gpu_dedicated_free_mb": 29000,
                        "host_commit_used_mb": 28000},
            "minima": {"host_available_mb": 10000, "dedicated_gpu_free_mb": 5000},
            "max_host_commit_used_mb": 45000}}]}


@pytest.mark.parametrize("params", [
    {}, {"image": "x.png", "prompt": ""}, {"image": "x.png", "prompt": "shot", "frames": 23},
    {"image": "x.png", "prompt": "shot", "seed": True}, {"image": "x.png", "prompt": "shot", "seed": -1},
    {"image": "x.png", "prompt": "shot", "seed": 2 ** 32},
    {"image": "x.png", "prompt": "shot", "ref_video": "x.mp4"},
])
def test_recipe_rejects_unsupported_or_invalid_settings(params):
    with pytest.raises(ValueError):
        vsa.parameters(params)


def test_prompt_is_not_rewritten_and_only_declared_settings_are_returned():
    params = {"image": "uploads/ref.png", "prompt": "Use <Picture 1>. No cuts.", "seed": 42}
    assert vsa.parameters(params) == params


def test_synthetic_or_unrendered_results_never_form_a_measured_profile():
    saved = receipt()
    saved["results"][0]["rendered"] = False
    with pytest.raises(ValueError, match="supervised"):
        vsa.measured_profile(saved)
    saved["results"][0]["rendered"] = True
    saved["results"][0]["measurements"]["refused_sample"] = {"host_low_memory": True}
    with pytest.raises(ValueError, match="supervised"):
        vsa.measured_profile(saved)


def test_pool_growth_is_independent_and_includes_error_margins():
    profile = vsa.measured_profile(receipt())
    assert profile.host_need_mb == 5000 + 1024
    assert profile.gpu_need_mb == 24000 + 1024
    assert profile.host_commit_need_mb == 17000 + 1024


def test_measured_commit_admission_does_not_borrow_physical_or_gpu_capacity(monkeypatch):
    monkeypatch.setattr(vsa, "status", lambda: {"qualified": True})
    monkeypatch.setattr(vsa.features, "saved", lambda *args: receipt())
    value = replace(snapshot(), host_commit_used_mb=49000)
    assert not vsa.memory_fit(value)["fits"]
    assert vsa.memory_fit(snapshot())["fits"]


def test_pageable_host_and_dynamic_gpu_growth_are_advisory_when_commit_fits(monkeypatch):
    monkeypatch.setattr(vsa, "status", lambda: {"qualified": True})
    monkeypatch.setattr(vsa.features, "saved", lambda *args: receipt())
    found = vsa.memory_fit(replace(snapshot(), host_free_mb=7000, gpu_dedicated_free_mb=26000))
    assert found["fits"] and found["measured"]
    assert found["host_prediction"] == {"advisory": True, "growth_mb": 6024,
                                       "available_mb": 7000, "floor_mb": 2048, "fits": False}
    assert not found["gpu_prediction"]["fits"]
    assert found["need_mb"] == 18024 and found["floor_mb"] == 4096


@pytest.mark.parametrize("free,configured,required", [(2047, "", 2048), (2048, "", 2048),
                                                     (4095, "4096", 4096), (4096, "4096", 4096)])
def test_pageable_host_prediction_never_relaxes_live_or_configured_physical_floor(monkeypatch, free, configured, required):
    monkeypatch.setattr(vsa, "status", lambda: {"qualified": True})
    monkeypatch.setattr(vsa.features, "saved", lambda *args: receipt())
    monkeypatch.setenv("SPARK_MEMORY_FLOOR_MB", configured)
    sample = replace(snapshot(), host_free_mb=free)
    if free < required:
        with pytest.raises(vsa.native.SafetyRefusal):
            vsa.memory_fit(sample)
    else:
        assert vsa.memory_fit(sample)["fits"]


def test_dynamic_gpu_residency_prediction_is_advisory_not_a_fixed_allocation(monkeypatch):
    monkeypatch.setattr(vsa, "status", lambda: {"qualified": True})
    monkeypatch.setattr(vsa.features, "saved", lambda *args: receipt())
    found = vsa.memory_fit(replace(snapshot(), gpu_dedicated_free_mb=26000))
    assert found["fits"] and found["measured"]
    assert found["gpu_prediction"] == {"advisory": True, "growth_mb": 25024,
                                       "available_mb": 26000, "floor_mb": 2048, "fits": False}
    assert vsa.COMFY_FLAGS == ("--use-ck-attention", "--vram-headroom", "3")


@pytest.mark.parametrize("free", [0, 1000, 2047, 2048])
def test_dynamic_gpu_advisory_never_relaxes_the_live_dedicated_floor(monkeypatch, free):
    monkeypatch.setattr(vsa, "status", lambda: {"qualified": True})
    monkeypatch.setattr(vsa.features, "saved", lambda *args: receipt())
    sample = replace(snapshot(), gpu_dedicated_free_mb=free)
    if free < 2048:
        with pytest.raises(vsa.native.SafetyRefusal):
            vsa.memory_fit(sample)
    else:
        assert vsa.memory_fit(sample)["fits"]


def test_unknown_hardware_split_does_not_reuse_a_receipt(monkeypatch):
    monkeypatch.setattr(vsa, "status", lambda: {"qualified": True})
    monkeypatch.setattr(vsa.features, "saved", lambda *args: receipt())
    with pytest.raises(vsa.native.SafetyRefusal, match="partition"):
        vsa.memory_fit(replace(snapshot(), host_total_mb=32000))


def test_qualification_bypasses_only_missing_evidence_not_live_floors(monkeypatch):
    import windows_memory
    class Signals:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(vsa, "prerequisites", lambda: "")
    monkeypatch.setattr(vsa.native, "device_uuid", lambda: "synthetic-device")
    monkeypatch.setattr(windows_memory, "WindowsSignals", Signals)
    monkeypatch.setattr(windows_memory, "WddmMemory", lambda *args: None)
    monkeypatch.setattr(windows_memory, "physical_memory_mb", lambda: 65536)
    monkeypatch.setattr(windows_memory, "dedicated_snapshot", lambda *args: replace(snapshot(), host_low_memory=True))
    with pytest.raises(vsa.native.SafetyRefusal, match="low physical"):
        vsa.preflight(qualification=True)


def test_graph_wires_int8_adapter_gate_and_sigma_into_the_sampler_model():
    graph = {key: {"inputs": {}} for key in ("1", "3", "4", "5", "9")}
    model = vsa.configure_workflow(graph)
    assert graph["1"]["inputs"]["unet_name"] == vsa.TRANSFORMER
    assert graph["3"]["inputs"]["clip_name"] == vsa.TEXT
    assert graph["4"]["inputs"]["vae_name"] == vsa.VAE
    assert graph["2"]["inputs"]["model"] == ["1", 0]
    assert graph["100"]["inputs"] == {"model": ["2", 0], "gate_file": vsa.GATE, "sparsity": .75}
    assert graph["101"]["inputs"] == {"model": ["100", 0], "shift_video": 12.0, "shift_audio": 3.0}
    assert graph["9"]["inputs"]["sampler_name"] == "euler"
    assert model == ["101", 0]


def test_unverified_basic_route_is_refused_when_qualification_is_enabled(client, monkeypatch):
    monkeypatch.setattr(vsa, "status", lambda: {"installed": True, "qualified": False, "why": "VSA not measured"})
    monkeypatch.setattr(windows_runtime, "mode", lambda: {"enabled": True})
    response = client.post("/api/jobs", json={"kind": vsa.KIND, "params": {"image": "uploads/x.png", "prompt": "shot"}})
    assert response.status_code == 501 and "not measured" in response.json()["error"]


def test_worker_placement_is_not_accepted(client):
    response = client.post("/api/jobs", json={"kind": vsa.KIND, "sparks": 2, "params": {}})
    assert response.status_code == 400 and "this machine" in response.json()["error"]


def test_normal_h3_gate_is_not_replaced_by_vsa_evidence(client, monkeypatch):
    import windows_cuda
    monkeypatch.setattr(windows_cuda, "h3_why_not", lambda: "H3 requires 90000 MB")
    monkeypatch.setattr(vsa, "status", lambda: {"available": True, "qualified": True})
    response = client.post("/api/jobs", json={"kind": "video", "params": {"prompt": "shot"}})
    assert response.status_code == 501 and "90000" in response.json()["error"]


def test_vsa_uses_native_supervision_not_draft_allocator_caps(monkeypatch):
    monkeypatch.setattr(server, "WINDOWS", True)
    expected = {"fits": True, "need_mb": 25000, "available_mb": 29000, "floor_mb": 2048}
    monkeypatch.setattr(vsa, "memory_fit", lambda frames=22: expected)
    job = {"kind": vsa.KIND, "params": {}}
    assert server.job_fit(job) == expected
    assert windows_runtime.bounded(vsa.KIND)
    assert windows_checks.native_image_check(job)
    assert windows_checks.timeout_seconds(job) > vsa.TIMEOUT_S


def test_failed_check_keeps_previous_evidence_and_saves_failure(tmp_path, monkeypatch):
    writes = []
    monkeypatch.setattr(vsa, "prerequisites", lambda: "Missing test model")
    monkeypatch.setattr(vsa.features, "save_attempt", lambda kind, value: writes.append(value))
    monkeypatch.setattr(vsa.features, "save_pass", lambda *args: pytest.fail("A failed check cannot replace a pass"))
    with pytest.raises(RuntimeError, match="Missing test model"):
        vsa.qualify(tmp_path / "check", "test")
    assert writes[-1]["ok"] is False and writes[-1]["error"] == "Missing test model"


def test_reference_paths_and_size_are_checked(tmp_path, monkeypatch):
    from PIL import Image
    monkeypatch.setattr(vsa, "route_status", lambda: {"available": True})
    monkeypatch.setattr(vsa.library, "studio_home", lambda: tmp_path)
    with pytest.raises(ValueError, match="inside"):
        vsa.require({"image": "../outside.png", "prompt": "shot"})
    path = tmp_path / "large.png"
    Image.new("RGB", (513, 288)).save(path)
    with pytest.raises(ValueError, match="512x288"):
        vsa.require({"image": "large.png", "prompt": "shot"})


@pytest.mark.parametrize("frames", vsa.CALIBRATION_FRAMES)
def test_supervised_command_parses_with_the_real_renderer_and_fixed_reference_policy(tmp_path, monkeypatch, frames):
    import sys
    from PIL import Image
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "spark"))
    import spark_generate as generation
    image = tmp_path / "reference.png"
    Image.new("RGB", (512, 288)).save(image)
    preflights = []
    monkeypatch.setattr(vsa, "preflight", lambda qualification=False, frames=22: preflights.append((qualification, frames)))
    monkeypatch.setattr(generation, "mac_release", lambda: None)
    captured = []
    monkeypatch.setattr(generation, "video", captured.append)
    def observe(command, output, validate, result, timeout_s, **kwargs):
        assert kwargs["dedicated_floor_mb"] == 2048
        monkeypatch.setattr(sys, "argv", command[3:])
        generation.main()
        return result
    monkeypatch.setattr(vsa.native, "observe_native", observe)
    monkeypatch.setattr(vsa, "fingerprint", lambda frames=22: {})
    monkeypatch.setattr(vsa, "save_render", lambda *args: None)
    vsa.run_case(tmp_path / "render", image, "Use <Picture 1> as the subject.", 7, {},
                 qualification=frames != 22, frames=frames)
    args = captured[0]
    assert (args.width, args.height, args.frames, args.steps) == (864, 480, frames, 4)
    assert args.ref_size == "match" and args.engine == "comfy" and args.weights == "int8"
    assert args.ref_plain and args.ref_mode == "vsa" and args.text_encoder == "regular"
    assert args.ref == [str(image)] and not args.ref_video and not args.ref_audio and not args.lora
    monkeypatch.setenv("STUDIO_REF2VA_VSA", "1")
    monkeypatch.setenv("STUDIO_VSA_CHECK", "1" if frames != 22 else "0")
    monkeypatch.setenv("STUDIO_VSA_FRAMES", str(frames))
    monkeypatch.setattr(vsa.sys, "platform", "win32")
    vsa.prepare_renderer(args)
    if frames != 22:
        monkeypatch.setenv("STUDIO_VSA_CHECK", "0")
        vsa.prepare_renderer(args)
        assert preflights[-1] == (False, frames)
        monkeypatch.setenv("STUDIO_VSA_CHECK", "1")
    monkeypatch.setattr(generation, "up", lambda: False)
    monkeypatch.setattr(generation, "side", lambda name: tmp_path / name)
    monkeypatch.setattr(generation, "ensure_server", lambda *args: None)
    monkeypatch.setattr(generation, "installed", lambda *args: True)
    monkeypatch.setattr(generation, "upload", lambda *args: "reference.png")
    monkeypatch.setattr(generation, "forget_uploads", lambda: None)
    monkeypatch.setenv("AUDIO_EQ", "on")
    graph = {}
    def capture_graph(workflow, sampler):
        graph.update(workflow)
        raise RuntimeError("captured complete workflow")
    monkeypatch.setattr(generation, "render", capture_graph)
    with pytest.raises(RuntimeError, match="captured complete"):
        generation.reference(args)
    assert graph["10"]["inputs"]["model"] == ["101", 0]
    assert graph["11"]["inputs"]["model"] == ["101", 0]
    assert graph["100"]["class_type"] == "Ref2VAVSAGatePatch"
    assert graph["101"]["class_type"] == "MiniMaxH3SigmaShift"
    assert graph["17"]["class_type"] == "AudioAdjustVolume"
    assert graph["18"]["class_type"] == "SaveAudio"
    assert graph["6"]["inputs"]["ref_image_size"] == "match"
    assert vsa.COMFY_FLAGS == ("--use-ck-attention", "--vram-headroom", "3")


@pytest.mark.parametrize("experimental", [False, True])
def test_http_rejections_explain_validation_without_changing_other_routes(monkeypatch, experimental):
    import io
    import urllib.error
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "spark"))
    import spark_generate as generation
    failure = urllib.error.HTTPError("http://127.0.0.1/prompt", 400, "Bad Request", {},
                                    io.BytesIO(b'{"error":"invalid MODEL link"}'))
    def refuse(*args, **kwargs):
        raise failure
    monkeypatch.setattr(generation.urllib.request, "urlopen", refuse)
    monkeypatch.setenv("STUDIO_REF2VA_VSA", "1" if experimental else "0")
    with pytest.raises(RuntimeError if experimental else urllib.error.HTTPError) as found:
        generation.call("/prompt", {})
    if experimental:
        assert "HTTP 400" in str(found.value) and "invalid MODEL link" in str(found.value)
    else:
        assert found.value is failure


@pytest.mark.parametrize("expected_frames", vsa.CALIBRATION_FRAMES)
@pytest.mark.parametrize("extra_frames", [0, 1, None])
def test_output_checks_decoded_frames_not_only_rounded_duration(monkeypatch, expected_frames, extra_frames):
    import json
    from types import SimpleNamespace
    import graphics
    frames = str(expected_frames + extra_frames) if extra_frames is not None else "N/A"
    monkeypatch.setattr(graphics, "probe", lambda target: {
        "width": 864, "height": 480, "seconds": expected_frames / 24, "fps": 24, "audio": True})
    monkeypatch.setattr(vsa.library, "tool", lambda name: name)
    def probe(command, **kwargs):
        return SimpleNamespace(stdout=json.dumps({"streams": [{"nb_read_frames": frames}]})
                               if "-count_frames" in command else b"\x00\xff")
    monkeypatch.setattr(vsa.subprocess, "run", probe)
    if extra_frames == 0:
        vsa.check_output(Path("video.mp4"), expected_frames)
    else:
        with pytest.raises(RuntimeError, match="decoded video frames"):
            vsa.check_output(Path("video.mp4"), expected_frames)


@pytest.mark.parametrize("frames", [True, "39", 23, 40, 363, None])
def test_calibration_rejects_unsupported_frame_counts(frames):
    from fastapi import HTTPException
    with pytest.raises(HTTPException, match="from 22 to 362"):
        windows_checks.parameters({"feature": vsa.KIND, "frames": frames})


@pytest.mark.parametrize("frames", vsa.CALIBRATION_FRAMES)
def test_calibration_build_preserves_frame_count_and_native_supervision(tmp_path, monkeypatch, frames):
    monkeypatch.setattr(vsa, "prerequisites", lambda: "")
    params = windows_checks.parameters({"feature": vsa.KIND, "frames": frames})
    command, env, outputs = windows_checks.build(tmp_path, params, "test")
    assert command[command.index("--frames") + 1] == str(frames)
    assert "--qualify" in command and len(outputs) == 2 and not env
    assert windows_checks.native_image_check({"kind": "qualification", "params": params})
    assert vsa.evidence_kind(frames) == (vsa.KIND if frames == 22 else f"{vsa.KIND}-{frames}")
    assert vsa.parameters({"image": "x.png", "prompt": "shot", "frames": frames}).get("frames", 22) == frames


def test_production_requires_only_baseline_evidence_not_selected_length(tmp_path, monkeypatch):
    calls = []
    def status(frames=22):
        calls.append(frames)
        return {"installed": True, "qualified": frames == 22, "why": f"{frames} frames unmeasured"}
    monkeypatch.setattr(vsa, "status", status)
    monkeypatch.setattr(windows_runtime, "mode", lambda: {"enabled": True})
    monkeypatch.setattr(vsa.library, "studio_home", lambda: tmp_path)
    monkeypatch.setattr(vsa, "require_image", lambda path: None)
    assert vsa.require({"image": "x.png", "prompt": "shot", "frames": 39})["frames"] == 39
    fit = vsa.memory_fit(snapshot(), 39)
    assert fit["fits"] and not fit["measured"] and fit["need_mb"] is None
    assert calls == [22, 22, 39]


def test_selected_length_uses_its_own_measured_admission_profile(monkeypatch):
    read = []
    monkeypatch.setattr(vsa, "status", lambda frames=22: {"qualified": True})
    def saved(kind, suffix):
        read.append(kind)
        return receipt()
    monkeypatch.setattr(vsa.features, "saved", saved)
    assert vsa.memory_fit(snapshot(), 39)["fits"]
    assert read == ["video-vsa-39"]


def test_production_command_carries_selected_verified_frame_count(tmp_path, monkeypatch):
    monkeypatch.setattr(vsa, "require", lambda params: params)
    command, _, _ = vsa.build(tmp_path, {"image": "x.png", "prompt": "shot", "seed": 7, "frames": 39}, "test")
    assert command[command.index("--frames") + 1] == "39" and "--qualify" not in command


@pytest.mark.parametrize("frames", [39, 56, 73])
def test_calibration_requires_preceding_pass_and_keeps_base_attempt_separate(tmp_path, monkeypatch, frames):
    writes = []
    monkeypatch.setattr(vsa, "prerequisites", lambda: "")
    monkeypatch.setattr(vsa.features, "saved", lambda *args: {})
    monkeypatch.setattr(vsa.features, "save_attempt", lambda kind, value: writes.append((kind, value)))
    monkeypatch.setattr(vsa.features, "save_pass", lambda *args: pytest.fail("Missing predecessor cannot pass"))
    preceding = 22
    with pytest.raises(RuntimeError, match=f"Pass the {preceding}-frame"):
        vsa.qualify(tmp_path / "output", "test", frames)
    assert all(kind == vsa.evidence_kind(frames) for kind, _ in writes)
    assert writes[-1][1]["check_only"] and not writes[-1][1]["ok"]


def test_longer_calibration_saves_separate_evidence_and_reuses_verified_checksums(tmp_path, monkeypatch):
    writes, runs, verifications = [], [], []
    previous = receipt() | {"fingerprint": {"models": {}, "recipe": {"frames": 22}}, "model_sha256": {}}
    monkeypatch.setattr(vsa, "prerequisites", lambda: "")
    monkeypatch.setattr(vsa.features, "saved", lambda kind, suffix: previous)
    monkeypatch.setattr(vsa.features, "save_attempt", lambda kind, value: writes.append(("attempt", kind, value)))
    monkeypatch.setattr(vsa.features, "save_pass", lambda kind, value: writes.append(("pass", kind, value)))
    monkeypatch.setattr(vsa, "fingerprint", lambda frames=22: {"models": {}, "recipe": {"frames": frames}})
    monkeypatch.setattr(vsa, "compatibility", lambda value: value)
    monkeypatch.setattr(vsa, "verify_model_updates", lambda *args: verifications.append(args))
    monkeypatch.setattr(vsa, "model_files", lambda: pytest.fail("Unchanged models must not be rehashed"))
    def run_case(output, image, prompt, seed, observations, qualification=False, frames=22, timeout_s=1800):
        runs.append((qualification, frames, image.is_file()))
        observations.update(previous["results"][0])
    monkeypatch.setattr(vsa, "run_case", run_case)
    result = vsa.qualify(tmp_path / "output", "test", 39)
    assert runs == [(True, 39, True)] and len(verifications) == 1
    assert all(kind == "video-vsa-39" for _, kind, _ in writes)
    assert result["fingerprint"]["recipe"]["frames"] == 39 and result["check_only"]
    assert not list(tmp_path.glob("*reference.png"))


def test_failed_longer_job_outcome_uses_its_own_attempt(monkeypatch):
    writes = []
    monkeypatch.setattr(windows_checks.features, "saved", lambda *args: {})
    monkeypatch.setattr(windows_checks.features, "save_attempt", lambda kind, value: writes.append((kind, value)))
    windows_checks.save_outcome({"id": "test", "params": {"feature": vsa.KIND, "frames": 39},
                                "status": "failed", "finished": 1, "error": "guard stopped"})
    assert writes[0][0] == "video-vsa-39" and writes[0][1]["error"] == "guard stopped"


def test_page_exposes_unmeasured_model_lengths_with_explicit_check_action():
    from test_shell_page import run
    page = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")
    chooser = page[page.index("function vsaLengths()"):page.index("function viewVsa()")]
    out = run("globalThis.VSA = {frames: 22};\n" + chooser + """
      S.windows_vsa = {available: true, qualified: true, installed: true,
        lengths: Array.from({length:21}, (_, i) => { const frames = 22 + 17 * i;
          return {frames, seconds:frames / 24, available:true, qualified:frames <= 39,
                  installed:true, scope:"Fixed canvas", why:"Unmeasured"}; })};
      globalThis.select = (obj, key, options) => el("select", {}, Object.values(options));
      globalThis.jobCard = () => el("div");
      out.count = vsaLengths().length; out.default = vsaLength().qualified;
      VSA.frames = "124"; out.unmeasured = !vsaLength().qualified;
      const panel = vsaCheckPanel(); out.panel = text(panel);
      await button(panel, "Check the 124-frame").attrs.onclick();
      out.body = calls.find(([path]) => path === "/api/feature-checks/check")[1];
      VSA.frames = 39; out.checked = vsaLength().qualified;
      VSA.frames = 362; out.maximum = vsaLength().frames;
      VSA.frames = 363; out.invalid = vsaLength().available;
    """)
    assert out["count"] == 21 and out["default"] and out["checked"]
    assert out["unmeasured"] and not out["invalid"] and out["maximum"] == 362
    assert "unmeasured lengths are not blocked" in out["panel"]
    assert out["body"] == {"feature": "video-vsa", "frames": 124}


def test_page_refreshes_completed_checks_and_retries_reported_failures():
    from test_shell_page import run
    page = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")
    refresh = page[page.index("async function refreshVsaCheck()"):page.index("async function poll()")]
    out = run("globalThis.VSA = {checkState: ''};\n" + refresh + """
      S.windows_vsa = {available: false}; S.capabilities = {video: {available: false}};
      jobs = [{id: "one", kind: "qualification", params: {feature: "video-vsa"}, status: "running"}];
      out.running = await refreshVsaCheck(); out.runningCalls = calls.length;
      answers["/api/state"] = {windows_vsa: {available: true, checked_at: "original"},
                              capabilities: {video: {available: true}}};
      jobs[0].status = "done"; out.done = await refreshVsaCheck();
      out.available = S.windows_vsa.available; out.standard = S.capabilities.video.available;
      out.repeat = await refreshVsaCheck(); out.doneCalls = calls.length;
      jobs[0] = {...jobs[0], id: "two", status: "failed"};
      out.failed = await refreshVsaCheck(); out.retained = S.windows_vsa.checked_at;
      jobs[0] = {...jobs[0], id: "three", status: "cancelled"};
      answers["/api/state"] = new Error("State refresh failed");
      out.error = await refreshVsaCheck(); out.errors = toasts;
      answers["/api/state"] = {windows_vsa: {available: true, checked_at: "original"}};
      out.retry = await refreshVsaCheck();
    """)
    assert not out["running"] and out["runningCalls"] == 0
    assert out["done"] and out["available"] and not out["standard"]
    assert not out["repeat"] and out["doneCalls"] == 1
    assert out["failed"] and out["retained"] == "original"
    assert not out["error"] and out["errors"] == ["State refresh failed"] and out["retry"]
