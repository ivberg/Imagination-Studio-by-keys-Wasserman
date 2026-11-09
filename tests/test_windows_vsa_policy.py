"""Baseline-only qualification policy; simulated receipts and memory, no model runs."""
from dataclasses import replace
from pathlib import Path

import pytest

from test_windows_ui import client, server  # noqa: F401
from test_windows_video_vsa import receipt, snapshot
from test_windows_vsa_sweep import sweep  # noqa: F401
import windows_video_vsa as vsa


@pytest.mark.parametrize("enabled", [False, True])
def test_missing_baseline_blocks_only_when_qualification_is_enabled(sweep, monkeypatch, enabled):
    monkeypatch.setattr(vsa.windows_runtime, "mode", lambda: {"enabled": enabled})
    found = vsa.route_status()
    assert found["available"] is not enabled and found["qualification_enabled"] is enabled
    if enabled:
        with pytest.raises(vsa.native.SafetyRefusal, match="22-frame"):
            vsa.memory_fit(snapshot(), 124)
    else:
        fit = vsa.memory_fit(snapshot(), 124)
        assert fit["fits"] and not fit["measured"] and fit["need_mb"] is None


def test_disabled_mode_never_uses_evidence_as_a_route_gate(sweep, monkeypatch):
    monkeypatch.setattr(vsa.windows_runtime, "mode", lambda: {"enabled": False})
    monkeypatch.setattr(vsa, "status", lambda *args: pytest.fail("The disabled route must not ask for evidence"))
    monkeypatch.setattr(vsa, "require_image", lambda path: None)
    assert vsa.require({"image": "picture.png", "prompt": "The scene.", "frames": 362})["frames"] == 362
    monkeypatch.setattr(vsa, "prerequisites", lambda: "Missing native media tools")
    assert not vsa.route_status()["available"]
    with pytest.raises(vsa.windows_runtime.RouteUnavailable, match="Missing native media tools"):
        vsa.require({"image": "picture.png", "prompt": "The scene.", "frames": 362})


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("pool", ["host", "commit", "gpu"])
def test_unmeasured_lengths_and_disabled_qualification_never_bypass_live_floors(sweep, monkeypatch, enabled, pool):
    sweep.save(22)
    monkeypatch.setattr(vsa.windows_runtime, "mode", lambda: {"enabled": enabled})
    sample = snapshot()
    if pool == "host":
        sample = replace(sample, host_low_memory=True)
    elif pool == "commit":
        sample = replace(sample, host_commit_used_mb=sample.host_commit_limit_mb - 1)
    else:
        sample = replace(sample, gpu_dedicated_free_mb=1000)
    with pytest.raises(vsa.native.SafetyRefusal):
        vsa.memory_fit(sample, 362)


@pytest.mark.parametrize("enabled", [False, True])
def test_state_allows_every_valid_length_after_basic_check_or_when_mode_is_off(client, sweep, monkeypatch, enabled):
    monkeypatch.setattr(vsa.windows_runtime, "mode", lambda: {"enabled": enabled})
    if enabled:
        sweep.save(22)
    response = client.get("/api/state")
    assert response.status_code == 200
    found = response.json()["windows_vsa"]
    assert found["qualification_enabled"] is enabled
    assert len(found["lengths"]) == 21 and all(row["available"] for row in found["lengths"])
    assert not next(row for row in found["lengths"] if row["frames"] == 362)["qualified"]


def test_successful_production_render_updates_highest_without_fabricating_qualification(sweep):
    sweep.save(22)
    vsa.save_render(124, vsa.fingerprint(124), receipt()["results"][0])
    found = vsa.sweep_summary([vsa.status(frames) for frames in vsa.CALIBRATION_FRAMES])
    assert found["longest_passed"]["frames"] == 124 and found["checked_count"] == 1
    assert not vsa.status(124)["qualified"]
    assert not vsa.features.saved(vsa.evidence_kind(124), "")
    assert not vsa.features.saved(vsa.evidence_kind(124), "-attempt")


@pytest.mark.parametrize("invalid", ["unrendered", "refused", "changed"])
def test_incomplete_or_changed_render_never_updates_success_history(sweep, invalid):
    observation = receipt()["results"][0]
    before = vsa.fingerprint(124)
    if invalid == "unrendered":
        observation["rendered"] = False
    elif invalid == "refused":
        observation["measurements"]["refused_sample"] = {"host_low_memory": True}
    else:
        sweep.identity["abi"]["contract"] += 1
    with pytest.raises((RuntimeError, ValueError)):
        vsa.save_render(124, before, observation)
    assert not vsa.features.saved("video-vsa-124-render", "-attempt")


def test_old_hardware_render_does_not_inflate_current_highest(sweep):
    vsa.save_render(124, vsa.fingerprint(124), receipt()["results"][0])
    sweep.identity["hardware"]["physical_total_mb"] = 131072
    found = vsa.sweep_summary([vsa.status(frames) for frames in vsa.CALIBRATION_FRAMES])
    assert found["longest_passed"] is None
    assert found["render_diagnostics"][0]["frames"] == 124


@pytest.mark.parametrize("enabled", [False, True])
def test_page_hides_qualification_controls_when_off_but_always_keeps_lengths_selectable(enabled):
    from test_shell_page import run
    page = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")
    controls = page[page.index("function vsaLengths()"):page.index("function viewCreate()")]
    out = run("globalThis.VSA = {image:'reference.png',prompt:'A scene.',seed:7,frames:124,mode:'single',sequenceSeconds:60};\n" +
              controls + """
      const enabled = """ + str(enabled).lower() + """;
      S.windows_vsa = {qualification_enabled:enabled,available:true,
        lengths:Array.from({length:21},(_,i) => {const frames=22+17*i;
          return {frames,seconds:frames/24,installed:true,available:true,qualified:frames===22,scope:"Fixed recipe"};}),
        qualification_summary:{longest_passed:{frames:73,seconds:73/24,checked_at:"old date"},checked_count:1,total_count:21}};
      globalThis.select = (obj,key,options) => el("select", {}, Object.values(options));
      globalThis.choice = () => el("div"); globalThis.bind = (obj,key,node) => node;
      globalThis.typed = node => node; globalThis.jobCard = () => el("div");
      const root = viewVsa(); out.text = text(root);
      out.enabled = !button(root,"Render 124 frames").attrs.disabled;
      out.options = find(root,n=>n.tag==="select").kids.length;
      out.basic = Boolean(button(root,"Check the basic 22-frame"));
      out.automatic = Boolean(button(root,"Qualify clip lengths automatically"));
      VSA.frames = 362; out.maximum = !button(viewVsa(),"Render 362 frames").attrs.disabled;
    """)
    assert out["enabled"] and out["maximum"] and out["options"] == 21
    assert out["basic"] is enabled and out["automatic"] is enabled
    if enabled:
        assert "Highest successful single clip: 3.042" in out["text"] and "not a length cap" in out["text"]
    else:
        assert "Feature check" not in out["text"] and "Highest successful" not in out["text"]
        assert "locally checked" not in out["text"] and "unmeasured" not in out["text"]
