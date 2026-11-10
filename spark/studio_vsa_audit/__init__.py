"""Pass-through diagnostics for the Studio's native VSA conditioning and timesteps."""
import hashlib
import json
from pathlib import Path
import re


class Audit:
    def __init__(self, path):
        self.path = Path(path)
        self.data = {"tokenization": [], "forwards": []}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def tokenize(self, tokenizer):
        def capture(text, **kwargs):
            tokens = tokenizer(text, **kwargs)
            groups = {}
            for name, batches in tokens.items():
                ids = [entry[0] for batch in batches for entry in batch if isinstance(entry[0], int)]
                groups[name] = {
                    "ids": ids,
                    "nontext_entries": sum(not isinstance(entry[0], int) for batch in batches for entry in batch),
                }
            self.data["tokenization"].append({
                "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "reference_types": [item["type"] for item in kwargs.get("minimax_ref_items") or []],
                "groups": groups,
            })
            self.save()
            return tokens
        return capture

    def forward(self, executor, x, timestep, context, transformer_options=None, **kwargs):
        from comfy.ldm.minimax.model import time_shift_sigma

        options = transformer_options or {}
        shift_video = float(options["minimax_h3_sigma_shift_video"])
        shift_audio = float(options["minimax_h3_sigma_shift_audio"])
        sigma_video = timestep.flatten()[0].float() / 1000.0
        sigma_audio = time_shift_sigma(sigma_video, shift_video, shift_audio)
        payload = kwargs.get("minimax_payload") or {}
        self.data["forwards"].append({
            "sigma_video": float(sigma_video),
            "sigma_audio": float(sigma_audio),
            "time_video": 1.0 - float(sigma_video),
            "time_audio": 1.0 - float(sigma_audio),
            "shift_video": shift_video, "shift_audio": shift_audio,
            "context_shape": list(context.shape), "context_dtype": str(context.dtype),
            "latent_shapes": [list(part.shape) for part in x],
            "reference_kinds": [ref["kind"] for ref in payload.get("refs") or []],
        })
        self.save()
        return executor(x, timestep, context, transformer_options, **kwargs)


class StudioVSAAudit:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",), "clip": ("CLIP",),
            "audit_name": ("STRING", {"default": "vsa_audit"}),
        }}

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "capture"
    CATEGORY = "Studio/diagnostics"

    def capture(self, model, clip, audit_name):
        import folder_paths
        from comfy.patcher_extension import WrappersMP

        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", audit_name):
            raise ValueError("A VSA audit name contains only letters, digits, underscores and hyphens.")
        audit = Audit(Path(folder_paths.get_output_directory()) / "studio" / f"{audit_name}.json")
        sampling = model.get_model_object("model_sampling")
        audit.data["sampling"] = {
            "shift": float(sampling.shift), "audio_shift": float(sampling.audio_shift),
            "audio_scale": float(sampling.audio_scale),
        }
        audit.save()
        patched = model.clone()
        # A function keeps the recorder shared when ComfyUI deep-copies model options.
        def forward(*args, **kwargs):
            return audit.forward(*args, **kwargs)
        patched.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "studio_vsa_audit", forward)
        traced_clip = clip.clone()
        traced_clip.tokenize = audit.tokenize(traced_clip.tokenize)
        return patched, traced_clip


NODE_CLASS_MAPPINGS = {"StudioVSAAudit": StudioVSAAudit}
NODE_DISPLAY_NAME_MAPPINGS = {"StudioVSAAudit": "Studio VSA conditioning audit"}
