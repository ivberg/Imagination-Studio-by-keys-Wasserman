"""Explicit Ref2VA VSA recipe; local render evidence never enables ordinary H3."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from feature_checks import matches_qualification
from lengths import counts, seconds_of
import windows_cuda
import windows_features as features
import windows_image_checks as native
import windows_library_checks as library
import windows_runtime
from windows_qualification import gpu_compatibility, hardware_profile, verify_model_updates

KIND = "video-vsa"
TITLE = "MiniMax H3 Ref2VA VSA"
CONTRACT = 1
WIDTH, HEIGHT, FRAMES = 864, 480, 22
CALIBRATION_FRAMES = tuple(counts())
TIMEOUT_S = 1800
SWEEP_KIND = "video-vsa-sweep"
SWEEP_TIMEOUT_S = 7200
SEQUENCE_MAX_S = 60
SEQUENCE_TIMEOUT_S = 7200
ASSEMBLY_TIMEOUT_S = 180
ADMISSION_TIMEOUT_S = 60
MARGIN_MB = 1024
COMFY_FLAGS = ("--use-ck-attention", "--vram-headroom", "3")
NODE_COMMIT = "323a0a5f7c636cc70f95464145732f81f04a72e9"
NODE_BLOBS = {"__init__.py": "39a8c6bf593f484470ded4f9172655157d6845db",
              "nodes.py": "2229cc1abbc4b619eef4cb1dbaa060797536cffe",
              "LICENSE": "7670aa58213f6bcb31c549d7b109c9d7302a98d6"}
TEXT = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
TRANSFORMER = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
TURBO = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
GATE = "fasth3_vsa_gate.safetensors"
SCOPE = "One reference picture up to 512x288; 864x480, 22 frames / 24 fps (0.917 s), four passes, VSA sparsity 0.75."
PROMPT = ("The baker in <Picture 1> stands at the counter and smiles gently. A locked, still camera. "
          "Mouth stays closed.\noverall_soundscape: Soft bakery room ambience.")


def node_root() -> Path:
    return windows_cuda.root() / "ComfyUI" / "custom_nodes" / "ComfyUI-Ref2VA-VSA"


def model_files() -> dict[str, Path]:
    root = windows_runtime.model_root()
    names = (f"FastH3-Comfy/text_encoders/{TEXT}", f"FastH3-Comfy/vae/{VAE}",
             f"FastH3-Comfy/vae/{AUDIO_VAE}", f"MiniMax-H3-Comfy/diffusion_models/{TRANSFORMER}",
             f"MiniMax-H3-Comfy/loras/{TURBO}", f"MiniMax-H3-Comfy/loras/{GATE}")
    return {name: root.joinpath(*name.split("/")) for name in names}


def prerequisites() -> str:
    if why := windows_cuda.why_not():
        return why
    if not (windows_cuda.root() / "ComfyUI" / "extra_model_paths.yaml").is_file():
        return "Run native CUDA setup to configure ComfyUI's model folders."
    import windows_media
    if any(not shutil.which(tool, path=windows_media.search_path(os.environ.get("PATH", "")))
           for tool in ("ffmpeg", "ffprobe")):
        return "Install native FFmpeg/ffprobe with scripts\\setup-windows-ffmpeg.ps1."
    missing = [path.name for path in model_files().values() if not path.is_file() or not path.stat().st_size]
    if missing:
        return "Missing VSA weights: " + ", ".join(missing)
    for name, expected in NODE_BLOBS.items():
        path = node_root() / name
        if not path.is_file():
            return "Install the pinned VSA node with scripts\\setup-windows-vsa.ps1."
        content = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if blob != expected:
            return "The VSA node differs from its pinned source; rerun its setup or restore that node."
    return ""


def parameters(params: dict) -> dict:
    if not isinstance(params, dict) or set(params) - {"image", "prompt", "seed", "frames", "sequence_seconds"}:
        raise ValueError("VSA accepts only image, prompt, seed, frames and optional sequence_seconds.")
    if not isinstance(params.get("image"), str) or not params["image"].strip():
        raise ValueError("Choose one reference picture inside Studio.")
    prompt = params.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 20000:
        raise ValueError("Write a nonempty prompt of at most 20000 characters; the picture is <Picture 1>.")
    seed = params.get("seed", 7)
    if type(seed) is not int or not 0 <= seed <= 2 ** 32 - 1:
        raise ValueError("Seed must be an integer from 0 to 4294967295.")
    frames = calibration_frames(params.get("frames", FRAMES))
    sequence = {"sequence_seconds": sequence_seconds(params["sequence_seconds"])} if "sequence_seconds" in params else {}
    return {"image": params["image"], "prompt": prompt, "seed": seed} | ({"frames": frames} if frames != FRAMES else {}) | sequence


def require_image(path: Path) -> None:
    from PIL import Image
    if not path.is_file():
        raise ValueError("The reference picture does not exist.")
    with Image.open(path) as image:
        image.load()
        if image.width > 512 or image.height > 288:
            raise ValueError("VSA reference: at most 512x288. Resize a copy explicitly before queuing.")


def calibration_frames(frames: int) -> int:
    if type(frames) is not int or frames not in CALIBRATION_FRAMES:
        raise ValueError("VSA accepts only model-valid 17n + 5 frame counts, from 22 to 362, at 864x480.")
    return frames


def evidence_kind(frames: int) -> str:
    return KIND if calibration_frames(frames) == FRAMES else f"{KIND}-{frames}"


def sequence_seconds(seconds: int) -> int:
    if type(seconds) is not int or not 1 <= seconds <= SEQUENCE_MAX_S:
        raise ValueError("A VSA sequence is a whole number of seconds from 1 to 60.")
    return seconds


def sequence_count(frames: int, seconds: int) -> int:
    return math.ceil(sequence_seconds(seconds) * 24 / calibration_frames(frames))


def fingerprint(frames: int = FRAMES) -> dict:
    frames = calibration_frames(frames)
    files = {name: {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
             for name, path in model_files().items()}
    return {"contract": CONTRACT, "kind": KIND,
            "recipe": {"width": WIDTH, "height": HEIGHT, "frames": frames, "steps": 4, "sparsity": .75,
                       "reference_width": 512, "reference_height": 288, "cold": True,
                       "comfy_flags": list(COMFY_FLAGS)},
            "models": files, "node": NODE_COMMIT,
            "model_root": hashlib.sha256(str(windows_runtime.model_root().resolve()).encode()).hexdigest(),
            "hardware": {**hardware_profile(), "device": hashlib.sha256(native.device_uuid().encode()).hexdigest()},
            "abi": gpu_compatibility(Path(windows_cuda.python()).parent.parent),
            "sources": {name: hashlib.sha256(features.PACK.joinpath(*name.split("/")).read_bytes()).hexdigest()
                        for name in ("app/windows_video_vsa.py", "spark/spark_generate.py", "spark/comfy_win.py",
                                     "app/windows_image_checks.py", "app/windows_memory.py")},
            "floor": os.environ.get("SPARK_MEMORY_FLOOR_MB", "")}


def compatibility(value: dict) -> dict:
    return {key: value[key] for key in ("contract", "kind", "recipe", "node", "hardware", "abi", "model_root", "floor")} | {
        "model_sizes": {name: row["size"] for name, row in value["models"].items()}}


def measured_profile(receipt: dict):
    from windows_memory import Profile
    result = receipt["results"][0]
    metrics = result["measurements"]
    if result.get("rendered") is not True or metrics["samples"] < 2 or metrics.get("refused_sample"):
        raise ValueError("The VSA receipt has no successful, supervised render measurements.")
    initial, minima = metrics["initial"], metrics["minima"]
    return Profile(KIND, receipt["checked_at"], initial["physical_total_mb"], initial["host_total_mb"],
                   max(0, initial["host_free_mb"] - minima["host_available_mb"]) + MARGIN_MB,
                   max(0, initial["gpu_dedicated_free_mb"] - minima["dedicated_gpu_free_mb"]) + MARGIN_MB,
                   max(0, metrics["max_host_commit_used_mb"] - initial["host_commit_used_mb"]) + MARGIN_MB,
                   initial["gpu_dedicated_total_mb"])


def status(frames: int = FRAMES) -> dict:
    frames = calibration_frames(frames)
    base = {"installed": False, "available": False, "qualified": False, "state": "not_installed",
            "experimental": False, "frames": frames, "seconds": seconds_of(frames),
            "scope": SCOPE.replace("22 frames", f"{frames} frames").replace("0.917 s", f"{frames / 24:.3f} s"),
            "check_available": True}
    try:
        if why := prerequisites():
            return {**base, "why": why}
        base.update(installed=True, state="needs_verification")
        receipt = features.saved(evidence_kind(frames), "")
        if receipt.get("ok") is not True:
            return {**base, "why": f"Run the VSA {frames}-frame feature check first. Memory for this length is not yet measured."}
        current, recorded = fingerprint(frames), receipt["fingerprint"]
        if not matches_qualification(recorded, current, compatibility(recorded), compatibility(current)):
            return {**base, "why": "The VSA recipe, runtime contract or hardware changed; check this recipe again."}
        verify_model_updates(windows_runtime.model_root(),
                             {name: (row["size"], receipt["model_sha256"][name]) for name, row in recorded["models"].items()},
                             {name: row["mtime_ns"] for name, row in recorded["models"].items()})
        measured_profile(receipt)
        return {**base, "qualified": True, "available": True, "state": "qualified", "why": "",
                "runtime_updated": recorded != current, "hardware": recorded["hardware"],
                "checked_at": receipt["checked_at"]}
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
        return {**base, "state": "qualification_error", "why": f"VSA verification failed: {error}"}


def route_status(baseline: dict | None = None, mode: dict | None = None) -> dict:
    enabled = (windows_runtime.mode() if mode is None else mode)["enabled"]
    if baseline is None:
        if enabled:
            baseline = status()
        else:
            why = prerequisites()
            baseline = {"installed": not why, "qualified": False, "why": why}
    available = baseline["installed"] and (not enabled or baseline["qualified"])
    return {**baseline, "qualification_enabled": enabled, "available": available,
            "why": "" if available else baseline["why"]}


def require(params: dict) -> dict:
    params = parameters(params)
    found = route_status()
    if not found["available"]:
        raise windows_runtime.RouteUnavailable(found["why"])
    home = library.studio_home().resolve()
    path = (home / params["image"]).resolve()
    if not path.is_relative_to(home):
        raise ValueError("The reference picture must be inside this Studio.")
    require_image(path)
    return params


def pressure(snapshot, previous=None) -> str:
    why = native.native_pressure(snapshot, previous)
    if not why and snapshot.gpu_dedicated_free_mb < 2048:
        return "Dedicated GPU memory is below the VSA 2048 MiB reserve."
    return why


def preflight(qualification: bool = False, frames: int = FRAMES, *, wait_for_admission: bool = False) -> None:
    frames = calibration_frames(frames)
    from windows_memory import WindowsSignals, WddmMemory, dedicated_snapshot, physical_memory_mb
    if why := prerequisites():
        raise RuntimeError(why)
    with WindowsSignals() as signals:
        snapshot = dedicated_snapshot(physical_memory_mb(), signals, WddmMemory(native.device_uuid()))
    if why := pressure(snapshot):
        raise native.SafetyRefusal(why)
    if qualification:
        return
    if wait_for_admission:
        # Renderer imports can consume the small margin left by the supervisor's earlier admission.
        wait_for_memory(frames, {})
        return
    found = memory_fit(snapshot, frames)
    if not found["fits"]:
        raise native.SafetyRefusal(found["reason"])


def memory_fit(snapshot=None, frames: int = FRAMES) -> dict:
    frames = calibration_frames(frames)
    if snapshot is None:
        from windows_memory import WindowsSignals, WddmMemory, dedicated_snapshot, physical_memory_mb
        with WindowsSignals() as signals:
            snapshot = dedicated_snapshot(physical_memory_mb(), signals, WddmMemory(native.device_uuid()))
    if why := pressure(snapshot):
        raise native.SafetyRefusal(why)
    baseline = status() if windows_runtime.mode()["enabled"] else None
    if baseline is not None and not baseline["qualified"]:
        raise native.SafetyRefusal(baseline["why"])
    found = baseline if baseline is not None and frames == FRAMES else status(frames)
    if not found["qualified"]:
        return {"fits": True, "measured": False, "need_mb": None, "available_mb": None, "floor_mb": None,
                "reason": "This clip length has no matching measured peak; live floors passed and runtime guards remain active."}
    profile = measured_profile(features.saved(evidence_kind(frames), ""))
    if (profile.physical_total_mb, profile.host_total_mb, profile.gpu_dedicated_total_mb) != (
            snapshot.physical_total_mb, snapshot.host_total_mb, snapshot.gpu_dedicated_total_mb):
        raise native.SafetyRefusal("The VSA measured profile does not match the current memory partition.")
    import comfy_win
    from windows_memory import host_floor_mb
    available = snapshot.host_commit_limit_mb - snapshot.host_commit_used_mb
    need, floor = profile.host_commit_need_mb, comfy_win.COMMIT_FLOOR_MB
    fits = available - need >= floor
    host_floor = max(2048, host_floor_mb(snapshot, os.environ.get("SPARK_MEMORY_FLOOR_MB", "")))
    host_prediction = {"advisory": True, "growth_mb": profile.host_need_mb,
                       "available_mb": snapshot.host_free_mb, "floor_mb": host_floor,
                       "fits": snapshot.host_free_mb - profile.host_need_mb >= host_floor}
    gpu_prediction = {"advisory": True, "growth_mb": profile.gpu_need_mb,
                      "available_mb": snapshot.gpu_dedicated_free_mb, "floor_mb": 2048,
                      "fits": snapshot.gpu_dedicated_free_mb - profile.gpu_need_mb >= 2048}
    return {"fits": fits, "measured": True, "need_mb": need, "available_mb": available, "floor_mb": floor,
            "host_prediction": host_prediction, "gpu_prediction": gpu_prediction,
            "reason": "" if fits else f"VSA needs {need} MiB measured commit growth plus its {floor} MiB reserve; {available} MiB is free."}


def wait_for_memory(frames: int, observations: dict, timeout_s: int = ADMISSION_TIMEOUT_S) -> None:
    """Wait for post-drain recovery, never for a weaker admission policy."""
    from windows_memory import WindowsSignals, WddmMemory, dedicated_snapshot, physical_memory_mb
    frames = calibration_frames(frames)
    if type(timeout_s) is not int or not 1 <= timeout_s <= ADMISSION_TIMEOUT_S:
        raise ValueError("VSA memory recovery must have a deadline from 1 to 60 seconds.")
    began = time.monotonic()
    deadline = began + timeout_s
    observations.update(state="waiting", samples=0, elapsed_s=0, consecutive_fits=0)
    previous = None
    try:
        physical = physical_memory_mb()
        with WindowsSignals() as signals:
            memory = WddmMemory(native.device_uuid())
            while True:
                if time.monotonic() >= deadline:
                    raise native.SafetyRefusal(
                        f"VSA memory did not recover within {timeout_s} seconds after backend drain. "
                        + (observations.get("reason") or "No stable fitting samples were obtained.")
                        + " Close unused host/GPU applications or choose a shorter shot, then retry; Studio stopped no unrelated processes.")
                sample = dedicated_snapshot(physical, signals, memory)
                now = time.monotonic()
                if not 0 <= now - sample.at <= 2:
                    raise native.SafetyRefusal("The VSA recovery memory sample is stale or has a future timestamp.")
                observations["samples"] += 1
                observations.setdefault("initial", asdict(sample))
                observations["final"] = asdict(sample)
                if why := pressure(sample, previous):
                    raise native.SafetyRefusal(why)
                fit = memory_fit(sample, frames)
                observations.update(admission=fit, reason=fit["reason"],
                                    consecutive_fits=observations["consecutive_fits"] + 1 if fit["fits"] else 0)
                if time.monotonic() >= deadline:
                    continue
                if time.monotonic() - sample.at > 2:
                    raise native.SafetyRefusal("The VSA recovery memory sample became stale during admission.")
                if observations["consecutive_fits"] >= 2:
                    observations["state"] = "admitted"
                    print("[tensorfold] VSA memory admission passed two fresh samples after backend drain.", flush=True)
                    return
                if observations["samples"] == 1 and not fit["fits"]:
                    print("[tensorfold] Waiting for VSA memory recovery after backend drain: " + fit["reason"], flush=True)
                previous = sample
                time.sleep(min(1, max(0, deadline - time.monotonic())))
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
        observations.update(state="failed", error=str(error))
        raise
    finally:
        observations["elapsed_s"] = round(time.monotonic() - began, 3)


def configure_workflow(workflow: dict) -> list:
    workflow["1"]["inputs"]["unet_name"] = TRANSFORMER
    workflow["3"]["inputs"]["clip_name"] = TEXT
    workflow["4"]["inputs"]["vae_name"] = VAE
    workflow["5"]["inputs"]["vae_name"] = AUDIO_VAE
    workflow["2"] = {"class_type": "LoraLoaderModelOnly",
                     "inputs": {"model": ["1", 0], "lora_name": TURBO, "strength_model": 1.0}}
    workflow["100"] = {"class_type": "Ref2VAVSAGatePatch",
                      "inputs": {"model": ["2", 0], "gate_file": GATE, "sparsity": .75}}
    workflow["101"] = {"class_type": "MiniMaxH3SigmaShift",
                      "inputs": {"model": ["100", 0], "shift_video": 12.0, "shift_audio": 3.0}}
    workflow["9"]["inputs"]["sampler_name"] = "euler"
    return ["101", 0]


def prepare_renderer(args) -> None:
    if sys.platform != "win32" or os.environ.get("STUDIO_REF2VA_VSA") != "1":
        raise RuntimeError("VSA runs only through its supervised native Windows Studio recipe.")
    qualification = os.environ.get("STUDIO_VSA_CHECK") == "1"
    frames = calibration_frames(int(os.environ.get("STUDIO_VSA_FRAMES", FRAMES))) if qualification else calibration_frames(args.frames)
    if ((args.width, args.height, args.frames, args.steps) != (WIDTH, HEIGHT, frames, 4)
            or len(args.ref or []) != 1 or args.ref_video or args.ref_audio or args.lora
            or args.crop or args.ref_video_sound or args.ref_size != "match" or args.engine != "comfy"
            or float(os.environ.get("SPARK_REF_CFG") or 1) != 1):
        raise ValueError(f"VSA requires its supervised one-picture, {frames}-frame, four-pass recipe.")
    require_image(Path(args.ref[0]))
    preflight(qualification=qualification, frames=frames, wait_for_admission=True)


def build(home: Path, params: dict, job_id: str) -> tuple[list[str], dict, list[str]]:
    params = require(params)
    target = home / "renders" / f"{job_id}-vsa" / "video.mp4"
    command = [sys.executable, "-X", "utf8", str(features.PACK / "scripts" / "windows-vsa-check.py"),
               "--output", str(target.parent), "--image", str((home / params["image"]).resolve()),
               "--prompt", params["prompt"], "--seed", str(params["seed"]),
               "--frames", str(params.get("frames", FRAMES))]
    outputs = [str(target)]
    if "sequence_seconds" in params:
        command += ["--sequence-seconds", str(params["sequence_seconds"])]
        outputs.append(str(target.parent / "sequence.json"))
    return command, {}, outputs


def check_output(target: Path, expected_frames: int = FRAMES) -> None:
    check_media(target, calibration_frames(expected_frames))


def check_media(target: Path, expected_frames: int) -> None:
    if type(expected_frames) is not int or not 1 <= expected_frames <= SEQUENCE_MAX_S * 24:
        raise ValueError("VSA output verification needs 1 to 1440 decoded frames.")
    import graphics
    found = graphics.probe(target)
    if ((found["width"], found["height"]) != (WIDTH, HEIGHT) or found["seconds"] is None
            or round(found["seconds"] * 24) != expected_frames or not math.isclose(found["fps"], 24, abs_tol=.01)
            or not found["audio"]):
        raise RuntimeError(f"VSA output does not match the checked canvas, {expected_frames} frames, 24 fps and sound.")
    frames = subprocess.run([library.tool("ffprobe"), "-v", "error", "-select_streams", "v:0", "-count_frames",
                             "-show_entries", "stream=nb_read_frames", "-of", "json", str(target)],
                            capture_output=True, text=True, check=True, timeout=30, creationflags=0x08000000)
    streams = json.loads(frames.stdout).get("streams")
    if not isinstance(streams, list) or len(streams) != 1 or streams[0].get("nb_read_frames") != str(expected_frames):
        raise RuntimeError(f"VSA output does not contain exactly {expected_frames} decoded video frames.")
    subprocess.run([library.tool("ffmpeg"), "-nostdin", "-v", "error", "-i", str(target), "-f", "null", "-"],
                   capture_output=True, check=True, timeout=30, creationflags=0x08000000)
    raw = subprocess.run([library.tool("ffmpeg"), "-nostdin", "-v", "error", "-i", str(target), "-frames:v", "1",
                          "-vf", "scale=64:36", "-pix_fmt", "gray", "-f", "rawvideo", "-"],
                         capture_output=True, check=True, timeout=30, creationflags=0x08000000).stdout
    if not raw or max(raw) - min(raw) < 8:
        raise RuntimeError("VSA produced a blank or nearly blank first frame.")


def run_case(output: Path, image: Path, prompt: str, seed: int, observations: dict, qualification=False,
             frames: int = FRAMES, timeout_s: int = TIMEOUT_S) -> dict:
    frames = calibration_frames(frames)
    if type(timeout_s) is not int or not 1 <= timeout_s <= TIMEOUT_S:
        raise ValueError("The VSA part deadline must be between 1 and 1800 seconds.")
    if output.exists():
        raise RuntimeError("VSA needs a fresh output folder.")
    output.mkdir(parents=True)
    require_image(image)
    preflight(qualification, frames)
    before = fingerprint(frames) if not qualification else None
    command = [windows_cuda.python(), "-X", "utf8", str(features.PACK / "spark" / "spark_generate.py"), "video",
               "--ref-mode", "vsa", "--ref", str(image), "--ref-size", "match", "--ref-plain",
               "--engine", "comfy", "--weights", "int8", "--text-encoder", "regular",
               "--width", str(WIDTH), "--height", str(HEIGHT), "--frames", str(frames), "--steps", "4",
               "--seed", str(seed), "--prompt", prompt, "-o", str(output / "video.mp4")]
    observations.update(case=f"Ref2VA VSA: 864x480, {frames} frames, four passes", rendered=False)
    result = native.observe_native(command, output, lambda: check_output(output / "video.mp4", frames),
                                   observations, timeout_s, pressure_policy=pressure, dedicated_floor_mb=2048)
    if before is not None:
        save_render(frames, before, observations)
    return result


def next_reference(clip: Path, image: Path, timeout_s: int = 120) -> None:
    try:
        subprocess.run([library.tool("ffmpeg"), "-nostdin", "-v", "error", "-y", "-threads", "2",
                        "-sseof", "-0.2", "-i", str(clip), "-vf", "scale=512:288:force_original_aspect_ratio=decrease",
                        "-update", "1", "-threads", "2", str(image)],
                       capture_output=True, text=True, encoding="utf-8", check=True, timeout=timeout_s, creationflags=0x08000000)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"VSA reference extraction failed: {error.stderr.strip()}") from error
    require_image(image)


def assemble_sequence(output: Path, clips: list[Path], frames: int, seconds: int, observations: dict,
                      timeout_s: int = ASSEMBLY_TIMEOUT_S) -> None:
    seconds = sequence_seconds(seconds)
    if len(clips) != sequence_count(frames, seconds):
        raise ValueError("The VSA sequence is missing one or more completed parts.")
    listing = output / "parts.txt"
    listing.write_text("".join(f"file '{clip.relative_to(output)}'\nduration {frames / 24:.12f}\n"
                               for clip in clips), encoding="utf-8")
    target = output / "video.mp4"
    if target.exists():
        raise RuntimeError("The VSA sequence needs a fresh final output.")
    pending = output / "assembling.mp4"
    command = [library.tool("ffmpeg"), "-nostdin", "-v", "error", "-threads", "2",
               "-f", "concat", "-safe", "0", "-i", str(listing),
               "-vf", f"fps=24,trim=end_frame={seconds * 24},setpts=N/(24*TB)",
               "-af", f"aresample=48000:async=1:first_pts=0,apad,atrim=duration={seconds},asetpts=N/SR/TB",
               "-frames:v", str(seconds * 24), "-t", str(seconds), "-c:v", "libx264", "-crf", "18",
               "-preset", "fast", "-pix_fmt", "yuv420p", "-threads", "2",
               "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-movflags", "+faststart", str(pending)]
    observations.update(case="VSA sequence: CPU assembly", rendered=False)
    try:
        native.observe_native(command, output, lambda: check_media(pending, seconds * 24), observations,
                              timeout_s, pressure_policy=pressure, dedicated_floor_mb=2048)
        pending.replace(target)
    finally:
        pending.unlink(missing_ok=True)


def run_sequence(output: Path, params: dict) -> dict:
    params = require(params)
    seconds = sequence_seconds(params.get("sequence_seconds"))
    frames = params.get("frames", FRAMES)
    count = sequence_count(frames, seconds)
    if output.exists():
        raise RuntimeError("A VSA sequence needs a fresh output folder; finished parts are never overwritten.")
    output.mkdir(parents=True)
    report = {"contract": 1, "state": "partial", "seconds": seconds, "target_frames": seconds * 24,
              "segment_frames": frames, "parts": [], "assembly": {}}
    deadline = time.monotonic() + SEQUENCE_TIMEOUT_S
    image = (library.studio_home().resolve() / params["image"]).resolve()
    clips = []

    def save():
        pending = output / "sequence.json.tmp"
        pending.write_text(json.dumps(report, indent=2), encoding="utf-8")
        pending.replace(output / "sequence.json")

    def remaining(limit):
        seconds_left = int(deadline - time.monotonic())
        if seconds_left < 1:
            raise native.SafetyRefusal("The VSA sequence exceeded its two-hour deadline.")
        return min(limit, seconds_left)

    save()
    observation = {}
    try:
        for index in range(count):
            report["active_part"] = index + 1
            observation = {}
            save()
            print(f"[tensorfold] stage {index + 1}/{count + 1} VSA sequence part {index + 1}/{count}", flush=True)
            # Every part gets a cold owned backend; a reference picture is not a first-frame anchor.
            windows_runtime.release_native()
            part = output / "parts" / f"{index + 1:03d}"
            seed = (params["seed"] + index) % (2 ** 32)
            reference = output / f"reference-{index + 1:03d}.png" if clips else None
            try:
                if reference is not None:
                    next_reference(clips[-1], reference, remaining(120))
                observation["admission"] = {}
                try:
                    wait_for_memory(frames, observation["admission"], remaining(ADMISSION_TIMEOUT_S))
                finally:
                    report["active_part_observations"] = observation
                    save()
                run_case(part, reference if reference is not None else image, params["prompt"], seed, observation,
                         frames=frames, timeout_s=remaining(TIMEOUT_S))
            finally:
                if reference is not None:
                    reference.unlink(missing_ok=True)
            clips.append(part / "video.mp4")
            report["parts"].append({"clip": str(clips[-1].relative_to(output)), "seed": seed,
                                    "observations": observation})
            report.pop("active_part")
            report.pop("active_part_observations")
            save()
        windows_runtime.release_native()
        print(f"[tensorfold] stage {count + 1}/{count + 1} Joining VSA sequence on the CPU", flush=True)
        assemble_sequence(output, clips, frames, seconds, report["assembly"], remaining(ASSEMBLY_TIMEOUT_S))
        report["state"] = "done"
        save()
        return report
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
        report.update(state="failed", error=str(error))
        if "active_part" in report:
            report["failed_part_observations"] = observation
        save()
        raise


def finish_aborted_sequence(home: Path, job: dict) -> None:
    """The queue owns the report after terminating the supervisor's job object."""
    if "sequence_seconds" not in job["params"] or job["status"] not in ("cancelled", "failed"):
        return
    output = home / "renders" / f"{job['id']}-vsa"
    target = output / "sequence.json"
    if not target.is_file():
        return
    report = json.loads(target.read_text(encoding="utf-8"))
    if report["state"] != "done":
        report.update(state=job["status"], error=job.get("error") or "The sequence was cancelled.")
        pending = output / "sequence.json.tmp"
        pending.write_text(json.dumps(report, indent=2), encoding="utf-8")
        pending.replace(target)
    job["outputs"] = [str(target.relative_to(home)).replace("\\", "/")]
    for part in report["parts"]:
        clip = (output / part["clip"]).resolve()
        if not clip.is_relative_to(output.resolve()):
            raise ValueError("The VSA sequence report names a clip outside its output folder.")
        if clip.is_file():
            job["outputs"].append(str(clip.relative_to(home.resolve())).replace("\\", "/"))


def save_sweep(report: dict, path: Path) -> None:
    import agents
    features.save_attempt(SWEEP_KIND, report)
    agents.write_json(path, report)


def length_summary(lengths: list[dict]) -> dict:
    passed = [row for row in lengths if row.get("qualified")]
    longest = max(passed, key=lambda row: row["frames"]) if passed else None
    return {"longest_passed": {key: longest[key] for key in ("frames", "seconds", "checked_at")} if longest else None,
            "checked_count": len(passed), "total_count": len(CALIBRATION_FRAMES)}


def save_render(frames: int, before: dict, observations: dict) -> None:
    current = fingerprint(frames)
    if before["models"] != current["models"] or not matches_qualification(
            before, current, compatibility(before), compatibility(current)):
        raise RuntimeError("The VSA recipe or model changed during rendering; its length success cannot be recorded.")
    record = {"frames": frames, "seconds": seconds_of(frames), "check_only": False,
              "checked_at": datetime.now(timezone.utc).isoformat(), "fingerprint": before, "results": [observations]}
    measured_profile(record)
    features.save_attempt(f"{KIND}-{frames}-render", record)


def render_status(frames: int) -> dict | None:
    try:
        record = features.saved(f"{KIND}-{frames}-render", "-attempt")
        if not record:
            return None
        recorded, current = record["fingerprint"], fingerprint(frames)
        if recorded["models"] != current["models"] or not matches_qualification(
                recorded, current, compatibility(recorded), compatibility(current)):
            return {"frames": frames, "successful": False, "why": "Prior render provenance does not match the current models, runtime or hardware."}
        measured_profile(record)
        return {"frames": frames, "seconds": seconds_of(frames), "successful": True, "checked_at": record["checked_at"]}
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
        return {"frames": frames, "successful": False, "why": f"Cannot verify prior VSA render: {error}"}


def sweep_summary(lengths: list[dict]) -> dict:
    result = {**length_summary(lengths), "sweep": None}
    renders = [row for frames in CALIBRATION_FRAMES if (row := render_status(frames)) is not None]
    for row in renders:
        if row["successful"] and (result["longest_passed"] is None or row["frames"] > result["longest_passed"]["frames"]):
            result["longest_passed"] = {key: row[key] for key in ("frames", "seconds", "checked_at")}
    result["render_diagnostics"] = [row for row in renders if not row["successful"]]
    report = features.saved(SWEEP_KIND, "-attempt")
    if not report:
        return result
    public = {key: report[key] for key in ("state", "job_id", "started_at", "finished_at", "active_frames",
                                          "points", "stop", "error") if key in report}
    try:
        recorded, current = report["fingerprint"], fingerprint()
        if not matches_qualification(recorded, current, compatibility(recorded), compatibility(current)):
            result["sweep"] = {"state": "stale", "why": "The previous sweep does not match this recipe, runtime contract or hardware."}
            return result
        if recorded["models"] != current["models"]:
            verify_model_updates(windows_runtime.model_root(),
                                 {name: (row["size"], report["model_sha256"][name]) for name, row in recorded["models"].items()},
                                 {name: row["mtime_ns"] for name, row in recorded["models"].items()})
        result["sweep"] = public
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
        result["sweep"] = {"state": "verification_error",
                           "why": f"Cannot verify the previous sweep: {report.get('error') or error}"}
    return result


def qualify_lengths(output: Path, job_id: str, report_path: Path) -> dict:
    """Reuse dated matching passes, then test increasing counts until the first stop."""
    report = {"feature": KIND, "job_id": job_id, "started_at": datetime.now(timezone.utc).isoformat(),
              "state": "running", "points": [], "active_frames": FRAMES}
    deadline = time.monotonic() + SWEEP_TIMEOUT_S
    environment = {key: os.environ.get(key) for key in ("STUDIO_VSA_CHECK", "STUDIO_VSA_FRAMES")}
    save_sweep(report, report_path)
    try:
        if why := prerequisites():
            raise RuntimeError(why)
        if output.exists():
            raise RuntimeError("The VSA length sweep requires a fresh output folder.")
        report["fingerprint"] = fingerprint()
        checked = {frames: status(frames) for frames in CALIBRATION_FRAMES}
        report.update(length_summary(list(checked.values())))
        for index, frames in enumerate(CALIBRATION_FRAMES, 1):
            report["active_frames"] = frames
            save_sweep(report, report_path)
            remaining = math.floor(deadline - time.monotonic())
            if remaining < 1:
                raise RuntimeError("The two-hour length qualification budget expired; this is not a measured memory limit.")
            current, recorded = fingerprint(), report["fingerprint"]
            if not matches_qualification(recorded, current, compatibility(recorded), compatibility(current)):
                raise RuntimeError("The recipe, runtime contract or hardware changed during the VSA sweep.")
            found = status(frames)
            reused = found["qualified"]
            print(f"[tensorfold] stage {index}/{len(CALIBRATION_FRAMES)} "
                  f"{'Reusing dated evidence for' if reused else 'Checking'} {frames} VSA frames", flush=True)
            if reused:
                receipt = features.saved(evidence_kind(frames), "")
            else:
                windows_runtime.release_native()
                remaining = math.floor(deadline - time.monotonic())
                if remaining < 1:
                    raise RuntimeError("The two-hour length qualification budget expired; this is not a measured memory limit.")
                os.environ.update(STUDIO_VSA_CHECK="1", STUDIO_VSA_FRAMES=str(frames))
                receipt = qualify(output / f"frames-{frames:03d}", job_id, frames,
                                  timeout_s=min(TIMEOUT_S, remaining))
            report["model_sha256"] = receipt["model_sha256"]
            report["points"].append({"frames": frames, "seconds": seconds_of(frames),
                                      "checked_at": receipt["checked_at"], "reused": reused,
                                      "source_job_id": receipt.get("job_id", "")})
            checked[frames] = {"qualified": True, "frames": frames, "seconds": seconds_of(frames),
                               "checked_at": receipt["checked_at"]}
            report.update(length_summary(list(checked.values())))
            save_sweep(report, report_path)
        report.update(state="done", finished_at=datetime.now(timezone.utc).isoformat())
        report.pop("active_frames")
        save_sweep(report, report_path)
        return report
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
        frames = report["active_frames"]
        report.update(state="failed", finished_at=datetime.now(timezone.utc).isoformat(), error=str(error),
                      stop={"frames": frames, "seconds": seconds_of(frames), "reason": str(error)})
        save_sweep(report, report_path)
        raise
    finally:
        for key, value in environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def qualify(output: Path, job_id: str, frames: int = FRAMES, timeout_s: int = TIMEOUT_S) -> dict:
    frames = calibration_frames(frames)
    kind = evidence_kind(frames)
    attempt = {"feature": KIND, "frames": frames, "check_only": True, "job_id": job_id, "ok": False,
               "checked_at": datetime.now(timezone.utc).isoformat()}
    observations = {}
    features.save_attempt(kind, attempt)
    try:
        if why := prerequisites():
            raise RuntimeError(why)
        previous = None
        if frames != FRAMES:
            preceding = FRAMES
            previous = features.saved(evidence_kind(preceding), "")
            if previous.get("ok") is not True:
                raise RuntimeError(f"Pass the {preceding}-frame VSA check before testing {frames} frames.")
            recorded, current = previous["fingerprint"], fingerprint(preceding)
            if not matches_qualification(recorded, current, compatibility(recorded), compatibility(current)):
                raise RuntimeError(f"The {preceding}-frame evidence no longer matches this runtime; check it again first.")
            verify_model_updates(windows_runtime.model_root(),
                                 {name: (row["size"], previous["model_sha256"][name]) for name, row in recorded["models"].items()},
                                 {name: row["mtime_ns"] for name, row in recorded["models"].items()})
            measured_profile(previous)
        if output.exists():
            raise RuntimeError("The VSA check requires a fresh output folder.")
        from PIL import Image
        fixture = features.PACK / "samples" / "baker_studio_1344x768_8s.png"
        if hashlib.sha256(fixture.read_bytes()).hexdigest() != library.FIXTURES[fixture.name]:
            raise RuntimeError("The pinned baker qualification picture changed.")
        output.parent.mkdir(parents=True, exist_ok=True)
        image = output.parent / f"{output.name}-reference.png"
        with Image.open(fixture) as source:
            source.resize((512, 288), Image.Resampling.LANCZOS).save(image)
        before = fingerprint(frames)
        try:
            run_case(output, image, PROMPT, 7, observations, qualification=True, frames=frames, timeout_s=timeout_s)
        finally:
            image.unlink(missing_ok=True)
        digests = dict(previous["model_sha256"]) if previous is not None else {}
        if previous is None:
            for name, path in model_files().items():
                with path.open("rb") as handle:
                    digests[name] = hashlib.file_digest(handle, "sha256").hexdigest()
        current = fingerprint(frames)
        if before["models"] != current["models"] or not matches_qualification(
                before, current, compatibility(before), compatibility(current)):
            raise RuntimeError("VSA models or execution contract changed during the check.")
        receipt = {**attempt, "ok": True, "fingerprint": before, "model_sha256": digests, "results": [observations]}
        measured_profile(receipt)
        features.save_pass(kind, receipt)
        features.save_attempt(kind, receipt)
        return receipt
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
        features.save_attempt(kind, {**attempt, "state": "failed", "error": str(error), "observations": observations})
        raise
