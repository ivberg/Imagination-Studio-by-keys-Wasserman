#!/usr/bin/env python3
"""Studio-owned experimental VSA rendering and qualification; no automatic downloads."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "spark"))
import windows_cuda
windows_cuda.activate()
import windows_media
os.environ["PATH"] = windows_media.search_path(os.environ.get("PATH", ""))
import windows_video_vsa as vsa

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
checks = parser.add_mutually_exclusive_group()
checks.add_argument("--qualify", action="store_true")
checks.add_argument("--sweep", action="store_true")
parser.add_argument("--report", type=Path)
parser.add_argument("--job-id", default="")
parser.add_argument("--image", type=Path)
parser.add_argument("--prompt")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--frames", type=int, choices=vsa.CALIBRATION_FRAMES, default=vsa.FRAMES)
parser.add_argument("--sequence-seconds", type=int)
args = parser.parse_args()
try:
    checking = args.qualify or args.sweep
    if checking and (args.report is None or args.image is not None or args.prompt is not None or args.sequence_seconds is not None):
        parser.error("Qualification needs --report, not a custom image or prompt.")
    if args.sweep and args.frames != vsa.FRAMES:
        parser.error("An automatic sweep chooses its own model-valid frame counts.")
    if not checking and (args.image is None or args.prompt is None):
        parser.error("Rendering needs --image and --prompt.")
    # A transplanted gate mutates the model's modules: never reuse it for another recipe or warm take.
    import windows_runtime
    windows_runtime.release_native()
    os.environ.update(STUDIO_REF2VA_VSA="1", SPARK_COMFY_ARGS=" ".join(vsa.COMFY_FLAGS), SPARK_REF_CFG="1")
    if args.sweep:
        result = vsa.qualify_lengths(args.output, args.job_id, args.report)
        print("[qualification] VSA length sweep completed. Reused evidence keeps its original date; ordinary H3 remains unchanged.", flush=True)
    elif args.qualify:
        os.environ["STUDIO_VSA_CHECK"] = "1"
        os.environ["STUDIO_VSA_FRAMES"] = str(args.frames)
        result = vsa.qualify(args.output, args.job_id, args.frames)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[qualification] VSA passed only its {args.frames}-frame recipe; other lengths need their own check and ordinary H3 remains unchanged.", flush=True)
    else:
        os.environ.pop("STUDIO_VSA_CHECK", None)
        os.environ.pop("STUDIO_VSA_FRAMES", None)
        params = vsa.parameters({"image": str(args.image), "prompt": args.prompt, "seed": args.seed, "frames": args.frames})
        if args.sequence_seconds is not None:
            params = vsa.parameters({**params, "sequence_seconds": args.sequence_seconds})
        vsa.require(params)
        if args.sequence_seconds is not None:
            vsa.run_sequence(args.output, params)
        else:
            observations = {}
            vsa.run_case(args.output, args.image, args.prompt, args.seed, observations, frames=args.frames)
            (args.output / "measurements.json").write_text(json.dumps(observations, indent=2), encoding="utf-8")
except (OSError, RuntimeError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
    print(str(error), file=sys.stderr)
    raise SystemExit(1) from None
