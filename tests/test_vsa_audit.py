"""VSA audit pass-through tests without Torch, ComfyUI, subprocesses or model work."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("studio_vsa_audit", ROOT / "spark" / "studio_vsa_audit" / "__init__.py")
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def test_token_capture_preserves_exact_inputs_and_return_object(tmp_path):
    audit = audit_module.Audit(tmp_path / "audit.json")
    tokens = {"qwen": [[(42, 1.0), ({"type": "image"}, 1.0), (151669, 1.0)]]}
    calls = []

    def tokenizer(text, **kwargs):
        calls.append((text, kwargs))
        return tokens

    refs = [{"type": "image"}]
    assert audit.tokenize(tokenizer)("Original words.", minimax_ref_items=refs) is tokens
    assert calls == [("Original words.", {"minimax_ref_items": refs})]
    saved = json.loads(audit.path.read_text())
    assert saved["tokenization"][0]["groups"]["qwen"] == {"ids": [42, 151669], "nontext_entries": 1}
    assert saved["tokenization"][0]["reference_types"] == ["image"]


def test_forward_capture_records_observed_times_without_changing_execution(tmp_path, monkeypatch):
    def shifted(sigma, source, target):
        base = sigma / (source + sigma * (1 - source))
        return target * base / (1 + (target - 1) * base)

    monkeypatch.setitem(sys.modules, "comfy.ldm.minimax.model", SimpleNamespace(time_shift_sigma=shifted))
    audit = audit_module.Audit(tmp_path / "audit.json")
    timestep = SimpleNamespace(flatten=lambda: [SimpleNamespace(float=lambda: 12000 / 13)])
    latents = [SimpleNamespace(shape=(1, 24, 8, 30, 54)), SimpleNamespace(shape=(1, 32, 2, 128))]
    context = SimpleNamespace(shape=(1, 400, 5120), dtype="bf16")
    options = {"minimax_h3_sigma_shift_video": 12.0, "minimax_h3_sigma_shift_audio": 3.0}
    payload = {"refs": [{"kind": "image"}]}
    returned = object()
    calls = []

    def execute(*args, **kwargs):
        calls.append((args, kwargs))
        return returned

    assert audit.forward(execute, latents, timestep, context, options, minimax_payload=payload) is returned
    assert calls == [((latents, timestep, context, options), {"minimax_payload": payload})]
    row = json.loads(audit.path.read_text())["forwards"][0]
    assert row["sigma_video"] == pytest.approx(12 / 13)
    assert row["sigma_audio"] == pytest.approx(0.75)
    assert row["time_audio"] == pytest.approx(0.25)
    assert row["reference_kinds"] == ["image"]
    assert row["context_shape"] == [1, 400, 5120]


def test_audit_wraps_only_clones_and_keeps_model_sampling(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace(get_output_directory=lambda: str(tmp_path)))
    monkeypatch.setitem(sys.modules, "comfy.patcher_extension", SimpleNamespace(
        WrappersMP=SimpleNamespace(DIFFUSION_MODEL="diffusion")))

    class Model:
        def clone(self):
            return Model()

        def get_model_object(self, name):
            assert name == "model_sampling"
            return SimpleNamespace(shift=12.0, audio_shift=3.0, audio_scale=4.0)

        def add_wrapper_with_key(self, kind, key, wrapper):
            self.wrapper = (kind, key, wrapper)

    class Clip:
        def clone(self):
            return Clip()

        def tokenize(self, text, **kwargs):
            return {"qwen": [[(42, 1.0)]]}

    model, clip = Model(), Clip()
    patched, traced = audit_module.StudioVSAAudit().capture(model, clip, "safe_audit")
    assert patched is not model and traced is not clip and not hasattr(model, "wrapper")
    assert patched.wrapper[:2] == ("diffusion", "studio_vsa_audit")
    assert traced.tokenize("A scene.") == clip.tokenize("A scene.")
    report = json.loads((tmp_path / "studio" / "safe_audit.json").read_text())
    assert report["sampling"] == {"shift": 12.0, "audio_shift": 3.0, "audio_scale": 4.0}
    with pytest.raises(ValueError, match="audit name"):
        audit_module.StudioVSAAudit().capture(model, clip, "../outside")
