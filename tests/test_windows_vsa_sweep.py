"""Hardware-local sweep semantics with synthetic receipts; never runs a GPU model."""
from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest

from test_windows_ui import client, server  # noqa: F401
from test_windows_video_vsa import receipt
import windows_checks
import windows_video_vsa as vsa


@pytest.fixture
def sweep(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDIO_WINDOWS_FEATURE_DIR", str(tmp_path / "evidence"))
    monkeypatch.delenv("STUDIO_QUALIFICATION_STRICT", raising=False)
    monkeypatch.setattr(vsa, "prerequisites", lambda: "")
    monkeypatch.setattr(vsa.library, "studio_home", lambda: tmp_path)
    monkeypatch.setattr(vsa.windows_runtime, "model_root", lambda: tmp_path / "models")
    identity = {"contract": 1, "kind": vsa.KIND, "models": {}, "node": "synthetic-node",
                "model_root": "synthetic-model-root", "abi": {"contract": 1}, "floor": "",
                "hardware": {"physical_total_mb": 65536, "host_total_mb": 32460,
                             "gpu_dedicated_total_mb": 31232, "device": "synthetic-device"}}
    monkeypatch.setattr(vsa, "fingerprint", lambda frames=22: {
        **deepcopy(identity), "recipe": {"frames": frames}})
    runs, releases = [], []
    monkeypatch.setattr(vsa.windows_runtime, "release_native", lambda: releases.append(True))
    monkeypatch.setattr(vsa, "verify_model_updates", lambda *args: None)

    def save(frames, job_id="old", checked_at="2026-01-01T00:00:00+00:00"):
        value = receipt() | {"feature": vsa.KIND, "job_id": job_id, "checked_at": checked_at,
                             "fingerprint": vsa.fingerprint(frames),
                             "model_sha256": {name: "synthetic-digest" for name in identity["models"]}}
        vsa.features.save_pass(vsa.evidence_kind(frames), value)
        return value

    def qualify(output, job_id, frames=22, timeout_s=1800):
        assert os.environ["STUDIO_VSA_CHECK"] == "1"
        assert os.environ["STUDIO_VSA_FRAMES"] == str(frames)
        runs.append((frames, timeout_s))
        value = save(frames, job_id, "2026-02-01T00:00:00+00:00")
        vsa.features.save_attempt(vsa.evidence_kind(frames), value)
        return value

    real_qualify = vsa.qualify
    monkeypatch.setattr(vsa, "qualify", qualify)
    return SimpleNamespace(home=tmp_path, output=tmp_path / "renders" / "calibration" / "sweep",
                           report=tmp_path / "exports" / "sweep-qualification.json", save=save,
                           qualify=qualify, real_qualify=real_qualify, runs=runs, releases=releases, identity=identity)


def summary():
    return vsa.sweep_summary([vsa.status(frames) for frames in vsa.CALIBRATION_FRAMES])


def test_sweep_reuses_matching_passes_with_original_dates_and_checks_every_other_count(sweep):
    original = {frames: sweep.save(frames) for frames in (22, 39, 56, 73)}
    result = vsa.qualify_lengths(sweep.output, "sweep", sweep.report)
    assert [row[0] for row in sweep.runs] == list(vsa.CALIBRATION_FRAMES[4:])
    assert len(sweep.releases) == 17
    assert all(1 <= timeout <= vsa.TIMEOUT_S for _, timeout in sweep.runs)
    assert [row["frames"] for row in result["points"]] == list(vsa.CALIBRATION_FRAMES)
    assert [row["reused"] for row in result["points"]] == [True] * 4 + [False] * 17
    for frames, value in original.items():
        assert vsa.features.saved(vsa.evidence_kind(frames), "") == value
        assert next(row for row in result["points"] if row["frames"] == frames)["checked_at"] == value["checked_at"]
    assert result["state"] == "done" and "active_frames" not in result
    assert json.loads(sweep.report.read_text()) == result
    assert not list(sweep.report.parent.glob("*.tmp"))
    found = summary()
    assert found["checked_count"] == 21 and found["longest_passed"]["frames"] == 362
    assert found["longest_passed"]["seconds"] == 362 / 24
    assert not vsa.features.saved(vsa.SWEEP_KIND, "")


def test_already_checked_counts_do_not_render_or_create_new_evidence_dates(sweep):
    for frames in vsa.CALIBRATION_FRAMES:
        sweep.save(frames)
    result = vsa.qualify_lengths(sweep.output, "sweep", sweep.report)
    assert not sweep.runs and not sweep.releases
    assert all(row["reused"] and row["source_job_id"] == "old" for row in result["points"])
    assert summary()["longest_passed"]["checked_at"] == "2026-01-01T00:00:00+00:00"


@pytest.mark.parametrize("reason", [
    "dedicated GPU memory is below the 2048 MiB floor",
    "host memory is below its native floor",
    "GPU telemetry is stale",
    "The model backend rejected the graph",
])
def test_first_stop_preserves_prior_higher_pass_and_does_not_infer_coverage(sweep, monkeypatch, reason):
    old = {frames: sweep.save(frames) for frames in (22, 73)}
    calls = []
    def fail(output, job_id, frames=22, timeout_s=1800):
        calls.append(frames)
        raise vsa.native.SafetyRefusal(reason)
    monkeypatch.setattr(vsa, "qualify", fail)
    with pytest.raises(vsa.native.SafetyRefusal, match=reason):
        vsa.qualify_lengths(sweep.output, "sweep", sweep.report)
    assert calls == [39]
    for frames, value in old.items():
        assert vsa.features.saved(vsa.evidence_kind(frames), "") == value
    found = summary()
    assert found["longest_passed"]["frames"] == 73 and found["checked_count"] == 2
    assert found["sweep"]["stop"] == {"frames": 39, "seconds": 39 / 24, "reason": reason}
    assert found["sweep"]["state"] == "failed"
    persisted = json.loads(sweep.report.read_text())
    assert persisted["stop"]["reason"] == reason
    assert persisted["longest_passed"]["frames"] == 73 and persisted["checked_count"] == 2
    assert not vsa.status(39)["qualified"]


def test_a_new_machine_cannot_reuse_old_passes_even_with_general_gating_off(sweep, monkeypatch):
    old = sweep.save(22)
    vsa.qualify_lengths(sweep.output, "sweep", sweep.report)
    sweep.identity["hardware"] = {**sweep.identity["hardware"], "physical_total_mb": 131072,
                                 "host_total_mb": 131072, "gpu_dedicated_total_mb": 98304}
    monkeypatch.setattr(vsa.windows_runtime, "mode", lambda: {"enabled": False})
    found = summary()
    assert not vsa.status(22)["qualified"]
    assert found["checked_count"] == 0 and found["longest_passed"] is None
    assert found["sweep"]["state"] == "stale"
    assert vsa.features.saved(vsa.KIND, "") == old


def test_deleted_receipt_is_not_counted_as_current_coverage_after_a_completed_sweep(sweep, monkeypatch):
    vsa.qualify_lengths(sweep.output, "sweep", sweep.report)
    saved = vsa.features.saved
    monkeypatch.setattr(vsa.features, "saved", lambda kind, suffix: {} if kind == "video-vsa-39" and not suffix else saved(kind, suffix))
    found = summary()
    assert found["checked_count"] == 20 and found["sweep"]["state"] == "done"
    assert found["longest_passed"]["frames"] == 362


def test_model_update_verification_failure_never_reuses_the_old_sweep_limit(sweep, monkeypatch):
    sweep.identity["models"] = {"synthetic-model": {"size": 4, "mtime_ns": 42}}
    sweep.save(22)
    vsa.qualify_lengths(sweep.output, "sweep", sweep.report)
    sweep.identity["models"]["synthetic-model"]["mtime_ns"] = 43
    def reject(*args):
        raise RuntimeError("Model checksum mismatch")
    monkeypatch.setattr(vsa, "verify_model_updates", reject)
    found = summary()
    assert found["longest_passed"] is None and found["sweep"]["state"] == "verification_error"
    assert "Model checksum mismatch" in found["sweep"]["why"]


def test_real_qualification_uses_baseline_mtimes_and_the_sweeps_remaining_deadline(sweep, monkeypatch):
    checks, runs = [], []
    sweep.identity["models"] = {"synthetic-model": {"size": 4, "mtime_ns": 42}}
    sweep.save(22)
    sweep.identity["models"]["synthetic-model"]["mtime_ns"] = 43
    monkeypatch.setattr(vsa, "verify_model_updates", lambda *args: checks.append(args))
    def run(output, image, prompt, seed, observations, **kwargs):
        runs.append(kwargs)
        observations.update(receipt()["results"][0])
    monkeypatch.setattr(vsa, "run_case", run)
    result = sweep.real_qualify(sweep.output, "sweep", 39, timeout_s=7)
    assert checks[0][2] == {"synthetic-model": 42}
    assert runs == [{"qualification": True, "frames": 39, "timeout_s": 7}]
    assert result["ok"] and result["fingerprint"]["models"]["synthetic-model"]["mtime_ns"] == 43


def test_budget_is_rechecked_after_backend_release_and_limits_the_next_part(sweep, monkeypatch):
    clock = iter((100, 101, 7290, 7301))
    monkeypatch.setattr(vsa, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    with pytest.raises(RuntimeError, match="not a measured memory limit"):
        vsa.qualify_lengths(sweep.output, "sweep", sweep.report)
    assert sweep.runs == [(22, 10)]
    assert summary()["sweep"]["stop"]["frames"] == 39


def test_no_part_starts_if_the_budget_expires_while_draining_the_backend(sweep, monkeypatch):
    clock = iter((0, .5, 7201))
    monkeypatch.setattr(vsa, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    with pytest.raises(RuntimeError, match="two-hour"):
        vsa.qualify_lengths(sweep.output, "sweep", sweep.report)
    assert not sweep.runs and len(sweep.releases) == 1


def test_parent_cancellation_persists_partial_report_without_touching_individual_passes(sweep, monkeypatch):
    old = sweep.save(22)
    monkeypatch.setenv("STUDIO_VSA_CHECK", "original")
    monkeypatch.setenv("STUDIO_VSA_FRAMES", "original")
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(vsa, "qualify", interrupt)
    with pytest.raises(KeyboardInterrupt):
        vsa.qualify_lengths(sweep.output, "sweep", sweep.report)
    assert os.environ["STUDIO_VSA_CHECK"] == os.environ["STUDIO_VSA_FRAMES"] == "original"
    windows_checks.save_outcome({"id": "sweep", "params": {"feature": vsa.KIND, "sweep": True},
                                "status": "cancelled", "finished": 1, "error": None})
    result = json.loads(sweep.report.read_text())
    assert result["state"] == "cancelled" and result["stop"]["frames"] == 39
    assert result["finished_at"] == "1970-01-01T00:00:01+00:00"
    assert len(result["points"]) == 1 and result["points"][0]["reused"]
    assert vsa.features.saved(vsa.KIND, "") == old
    assert not vsa.features.saved(vsa.KIND, "-attempt")


def test_changed_contract_during_sweep_stops_before_the_next_render(sweep, monkeypatch):
    def change(*args, **kwargs):
        value = sweep.qualify(*args, **kwargs)
        sweep.identity["abi"]["contract"] += 1
        return value
    monkeypatch.setattr(vsa, "qualify", change)
    with pytest.raises(RuntimeError, match="changed during"):
        vsa.qualify_lengths(sweep.output, "sweep", sweep.report)
    assert [row[0] for row in sweep.runs] == [22]


def test_prerequisite_failure_has_no_pass_or_misleading_memory_limit(sweep, monkeypatch):
    monkeypatch.setattr(vsa, "prerequisites", lambda: "Missing experimental weights")
    with pytest.raises(RuntimeError, match="Missing experimental weights"):
        vsa.qualify_lengths(sweep.output, "sweep", sweep.report)
    found = summary()
    assert not sweep.runs and found["longest_passed"] is None
    assert found["sweep"]["state"] == "verification_error"
    assert "Missing experimental weights" in found["sweep"]["why"]


@pytest.mark.parametrize("extra", [{"sweep": False}, {"sweep": 1}, {"sweep": "true"},
                                  {"sweep": True, "frames": 39}, {"sweep": True, "sequence_seconds": 60}])
def test_sweep_api_rejects_ambiguous_or_untyped_modes(extra):
    from fastapi import HTTPException
    with pytest.raises(HTTPException, match="automatic VSA sweep"):
        windows_checks.parameters({"feature": vsa.KIND, **extra})


def test_sweep_uses_queue_native_supervision_report_and_bounded_deadline(sweep):
    params = windows_checks.parameters({"feature": vsa.KIND, "sweep": True})
    command, env, outputs = windows_checks.build(sweep.home, params, "sweep")
    job = {"kind": "qualification", "params": params}
    assert "--sweep" in command and "--qualify" not in command and "--frames" not in command
    assert outputs == [str(sweep.report)] and not env
    assert windows_checks.gpu_check(job) and windows_checks.native_image_check(job)
    assert windows_checks.timeout_seconds(job) == vsa.SWEEP_TIMEOUT_S + 180
    assert vsa.SEQUENCE_MAX_S == 60


def test_api_keeps_one_vsa_check_at_a_time(client, sweep, monkeypatch):
    monkeypatch.setattr(server, "jobs", {})
    monkeypatch.setattr(server, "enqueue", lambda kind, title, supplied: {"kind": kind, "params": supplied})
    response = client.post("/api/feature-checks/check", json={"feature": vsa.KIND, "sweep": True})
    assert response.status_code == 200 and response.json()["params"] == {"feature": vsa.KIND, "sweep": True}
    server.jobs["busy"] = {"kind": "qualification", "params": {"feature": vsa.KIND, "frames": 39}, "status": "running"}
    assert client.post("/api/feature-checks/check", json={"feature": vsa.KIND, "sweep": True}).status_code == 409


def test_state_exposes_dated_length_summary_without_enabling_ordinary_h3(client, sweep):
    sweep.save(22)
    response = client.get("/api/state")
    assert response.status_code == 200
    state = response.json()
    assert state["windows_vsa"]["qualification_summary"]["longest_passed"] == {
        "frames": 22, "seconds": 22 / 24, "checked_at": "2026-01-01T00:00:00+00:00"}
    assert not state["capabilities"]["video"]["available"]


@pytest.mark.parametrize("extra", [[], ["--qualify"], ["--frames", "39"], ["--sequence-seconds", "1"]])
def test_supervisor_dispatches_only_an_explicit_compatible_sweep(sweep, monkeypatch, extra):
    calls = []
    monkeypatch.setattr(vsa.windows_cuda, "activate", lambda: None)
    monkeypatch.setattr(vsa, "qualify_lengths", lambda *args: calls.append(args))
    monkeypatch.setattr(sys, "path", sys.path.copy())
    for key in ("STUDIO_REF2VA_VSA", "SPARK_COMFY_ARGS", "SPARK_REF_CFG"):
        monkeypatch.setenv(key, os.environ.get(key, ""))
    monkeypatch.setattr(sys, "argv", ["windows-vsa-check.py", "--sweep", "--output", str(sweep.output),
                                    "--report", str(sweep.report), "--job-id", "sweep", *extra])
    script = Path(__file__).resolve().parents[1] / "scripts" / "windows-vsa-check.py"
    if extra:
        with pytest.raises(SystemExit) as error:
            runpy.run_path(str(script), run_name="__main__")
        assert error.value.code == 2 and not calls and not sweep.releases
    else:
        runpy.run_path(str(script), run_name="__main__")
        assert calls == [(sweep.output, "sweep", sweep.report)]


def test_page_sweep_action_summary_and_gaps_are_explicit():
    from test_shell_page import run
    page = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")
    chooser = page[page.index("function vsaLengths()"):page.index("function viewVsa()")]
    out = run("globalThis.VSA = {frames: 22};\n" + chooser + """
      globalThis.select = () => el("select"); globalThis.jobCard = () => el("div", {}, "job card");
      S.windows_vsa = {installed:true, frames:22, seconds:22/24,
        qualification_summary: {longest_passed:{frames:73,seconds:73/24,checked_at:"original date"},
          checked_count:2,total_count:21,sweep:{state:"failed",started_at:"sweep date",
          stop:{frames:39,seconds:39/24,reason:"Host pressure"}}}};
      const panel = vsaCheckPanel(); out.panel = text(panel);
      await button(panel, "Qualify clip lengths automatically").attrs.onclick();
      out.body = calls.find(([path]) => path === "/api/feature-checks/check")[1];
      S.windows_vsa.qualification_summary.sweep = {state:"done",finished_at:"finished date"};
      out.gaps = text(vsaCheckPanel());
      jobs = [{kind:"qualification",params:{feature:"video-vsa",sweep:true},status:"running"}];
      out.disabled = button(vsaCheckPanel(), "Qualify clip lengths automatically").attrs.disabled;
    """)
    assert out["body"] == {"feature": "video-vsa", "sweep": True} and out["disabled"]
    assert "3.042 s (73 frames)" in out["panel"] and "original date" in out["panel"]
    assert "1.625 s (39 frames): Host pressure" in out["panel"] and "sweep date" in out["panel"]
    assert "2 of 21" in out["panel"] and "not an absolute" in out["panel"]
    assert "All 21" not in out["gaps"] and "Current matching evidence" in out["gaps"]


def test_page_refreshes_sweep_stage_progress_without_replacing_inputs_or_h3_state():
    from test_shell_page import run
    page = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")
    refresh = page[page.index("async function refreshVsaCheck()"):page.index("async function poll()")]
    out = run("globalThis.VSA = {checkState:'',frames:124,image:'chosen',prompt:'written'};\n" + refresh + """
      S.windows_vsa = {}; S.capabilities = {video:{available:false}};
      jobs = [{id:"sweep",kind:"qualification",params:{feature:"video-vsa",sweep:true},status:"running",progress:{at:5,of:21}}];
      answers["/api/state"] = {windows_vsa:{qualification_summary:{checked_count:4}}};
      out.first = await refreshVsaCheck(); out.repeat = await refreshVsaCheck();
      jobs[0].progress.at = 6;
      answers["/api/state"].windows_vsa.qualification_summary.checked_count = 5;
      out.next = await refreshVsaCheck(); out.count = S.windows_vsa.qualification_summary.checked_count;
      out.inputs = [VSA.frames,VSA.image,VSA.prompt]; out.h3 = S.capabilities.video.available;
    """)
    assert out["first"] and not out["repeat"] and out["next"] and out["count"] == 5
    assert out["inputs"] == [124, "chosen", "written"] and not out["h3"]
