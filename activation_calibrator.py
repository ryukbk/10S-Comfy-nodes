"""
activation_calibrator.py

Captures per-layer input activation second moments E[x^2] during real sampling,
to build an importance / override mask that guides spectral compression in
outlier_dampen.py. Adds the BEHAVIORAL axis that weight-space alone can't see:
which delta directions actually activate under a given condition.

Why two nodes
─────────────
Forward hooks fire during sampling, which happens AFTER a node's function
returns (same reason model_inspector's trace hooks "print on next pass"). So
capture is split:
  • LTXActivationCalibrator  — clones the model, attaches forward-pre-hooks that
    accumulate E[x^2] per target linear into process-global state, passes MODEL
    through. Runs at graph-build time; the accumulation happens later, during the
    sampler pass.
  • LTXCalibratorFlush       — placed DOWNSTREAM of the sampler (takes its LATENT
    so it sequences after sampling). Writes the accumulated running-mean mask to
    disk. Because global state persists across queued prompts, each flush writes
    cumulative-so-far; the final prompt's flush is the complete pool.

Workflow (two contrast axes)
────────────────────────────
  Override axis (drives compression):
    - point output_path at "override_agree.safetensors", label "agree",
      queue your AGREE prompts (DMD loaded), reset=True on the first only.
    - point output_path at "override_conflict.safetensors", label "conflict",
      queue your CONFLICT prompts (DMD loaded).
  DMD axis (diagnostic — is DMD distinct from override?):
    - "dmd_on.safetensors" / "dmd_off.safetensors", same prompt, DMD toggled.

  outlier_dampen then differences the masks and weights the winsorize by
  per-direction override activation (conflict - agree), optionally sparing
  directions that light up on the DMD axis.

Each condition uses its own output_path -> its own global bucket -> its own
file, so no cross-condition contamination. Use a fresh path per condition, or
reset=True on the first prompt of a pool.

Notes
─────
  • Accumulation is on CPU (float32) to avoid VRAM growth; a few MB total.
  • v1 accumulates over the whole sample. Zone-aware bucketing (early/structure
    steps vs late/refinement steps) is a planned refinement — the override
    commits early, so a structure-zone mask will be sharper, but reading the
    current sigma inside a pre-hook needs transformer_options plumbing that's
    best added after basic capture is validated.
"""

import os
import json
import torch
from safetensors.torch import save_file, load_file

try:
    from safetensors import safe_open
except Exception:
    safe_open = None


_CALIB_FORMAT = "sum_sq+counts_v1"


def _try_load_bucket(output_path):
    """
    Resume an accumulator from disk if a compatible file exists.
    Returns a bucket dict (raw sum_sq + counts) or None if absent/incompatible.
    """
    if safe_open is None or not os.path.isfile(output_path):
        return None
    try:
        with safe_open(output_path, framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
            if meta.get("calib_format") != _CALIB_FORMAT:
                # old means-only file or foreign format — can't safely resume
                return None
            counts = json.loads(meta.get("calib_counts", "{}"))
            sum_sq = {k: f.get_tensor(k).to(torch.float32) for k in f.keys()}
        return {
            "label":   meta.get("calib_label", ""),
            "sum_sq":  sum_sq,
            "count":   {k: int(v) for k, v in counts.items()},
            "targets": set(sum_sq.keys()),
            "passes":  int(meta.get("calib_passes", "0") or 0),
            "resumed": True,
        }
    except Exception as e:
        print(f"[10S] Calibrator: could not resume from {output_path}: "
              f"{type(e).__name__}: {e}")
        return None


# ── Process-global accumulator state ─────────────────────────────────────────
# keyed by output_path -> {"label": str, "sum_sq": {layer: cpu tensor},
#                          "count": {layer: int}, "targets": set, "passes": int}
_CALIB_STATE = {}


def _resolve_diffusion_model(m):
    """Find the actual nn.Module backbone (mirrors model_inspector)."""
    for path in ("diffusion_model", "model", "transformer", "dit", "net"):
        obj = getattr(m.model, path, None)
        if obj is not None and hasattr(obj, "named_modules"):
            return obj, path
    if hasattr(m.model, "named_modules"):
        return m.model, "model"
    return None, None


def _make_pre_hook(layer_name, bucket):
    """
    forward_pre_hook: hook(module, inputs) -> None
    Accumulates sum over tokens of x^2 (per input feature) + token count, so a
    true running E[x^2] can be formed at flush time across the whole pool.
    """
    def hook(module, inputs):
        if not inputs:
            return None
        x = inputs[0]
        if not torch.is_tensor(x):
            return None
        try:
            xf = x.detach().to(torch.float32)
            feat = xf.shape[-1]
            xf = xf.reshape(-1, feat)                 # [tokens, feat]
            ss = xf.pow(2).sum(dim=0).cpu()           # [feat]
            n = xf.shape[0]
        except Exception:
            return None
        sq = bucket["sum_sq"]
        cn = bucket["count"]
        if layer_name in sq:
            # guard against feature-dim drift between calls
            if sq[layer_name].shape == ss.shape:
                sq[layer_name] += ss
                cn[layer_name] += n
        else:
            sq[layer_name] = ss
            cn[layer_name] = n
        return None
    return hook


class LTXActivationCalibrator:
    """
    Attaches forward-pre-hooks to target linear modules and accumulates
    per-feature E[x^2] into a named global bucket during the next sampling pass.
    Passes the MODEL through unchanged (cloned).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "output_path": ("STRING", {
                    "default": "calib/override_agree.safetensors",
                    "tooltip": "Bucket/file this pool accumulates into. Use a "
                               "distinct path per condition (agree/conflict/"
                               "dmd_on/dmd_off).",
                }),
                "condition_label": ("STRING", {
                    "default": "agree",
                    "tooltip": "Free-form label stored in metadata.",
                }),
            },
            "optional": {
                "target_filter": ("STRING", {
                    "default": "(attn1|ff\\.net|proj_out|to_gate_logits)",
                    "tooltip": "Regex; a module is captured if its dotted path "
                               "matches. Note 'attn1' also matches 'audio_attn1' "
                               "— use exclude_filter to drop the audio stream.",
                }),
                "exclude_filter": ("STRING", {
                    "default": "",
                    "tooltip": "Regex; modules whose path matches are skipped "
                               "even if target_filter matched. E.g. 'audio_' for "
                               "video-only, or 'to_gate_logits' to drop gates.",
                }),
                "reset": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Clear this path's accumulator before this pass. "
                               "Set True on the FIRST prompt of a pool only.",
                }),
                "max_targets": ("INT", {
                    "default": 2000, "min": 1, "max": 20000,
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "calibrate"
    CATEGORY = "10S Nodes/Calibration"
    DESCRIPTION = ("Accumulates per-layer input E[x^2] during sampling into a "
                   "named bucket. Pair with LTXCalibratorFlush downstream of the "
                   "sampler to write the mask.")

    def calibrate(self, model, output_path, condition_label,
                  target_filter="(attn1|ff\\.net|proj_out|to_gate_logits)",
                  exclude_filter="", reset=False, max_targets=2000):
        import re
        m = model.clone()
        backbone, backbone_path = _resolve_diffusion_model(m)
        if backbone is None:
            msg = "[10S] Calibrator: could not locate diffusion backbone."
            print(msg)
            return (m, msg)

        try:
            inc_re = re.compile(target_filter) if target_filter.strip() else None
        except re.error as e:
            msg = f"[10S] Calibrator: bad target_filter regex: {e}"
            print(msg)
            return (m, msg)
        try:
            exc_re = re.compile(exclude_filter) if exclude_filter.strip() else None
        except re.error as e:
            msg = f"[10S] Calibrator: bad exclude_filter regex: {e}"
            print(msg)
            return (m, msg)

        resumed_from_disk = False
        if reset:
            _CALIB_STATE[output_path] = {
                "label": condition_label, "sum_sq": {}, "count": {},
                "targets": set(), "passes": 0,
            }
        elif output_path not in _CALIB_STATE:
            loaded = _try_load_bucket(output_path)
            if loaded is not None:
                _CALIB_STATE[output_path] = loaded
                resumed_from_disk = True
            else:
                _CALIB_STATE[output_path] = {
                    "label": condition_label, "sum_sq": {}, "count": {},
                    "targets": set(), "passes": 0,
                }
        bucket = _CALIB_STATE[output_path]
        bucket["label"] = condition_label

        attached = 0
        n_audio = 0
        for name, module in backbone.named_modules():
            if not name:
                continue
            if inc_re is not None and not inc_re.search(name):
                continue
            if exc_re is not None and exc_re.search(name):
                continue
            # only leaf-ish modules with own parameters (skip containers)
            try:
                if sum(p.numel() for p in module.parameters(recurse=False)) == 0:
                    continue
            except Exception:
                continue
            try:
                module.register_forward_pre_hook(_make_pre_hook(name, bucket))
                bucket["targets"].add(name)
                attached += 1
                if "audio" in name.lower():
                    n_audio += 1
            except Exception as e:
                print(f"  ! hook failed on {name}: {type(e).__name__}: {e}")
            if attached >= max_targets:
                break

        bucket["passes"] += 1
        resume_note = ("  (RESUMED accumulator from disk)\n"
                       if resumed_from_disk else "")
        report = (f"[10S] Calibrator: bucket '{output_path}' label='{condition_label}' "
                  f"reset={reset}\n"
                  f"{resume_note}"
                  f"  backbone: m.model.{backbone_path}\n"
                  f"  hooks attached: {attached}  ({n_audio} audio-stream, "
                  f"{attached - n_audio} video-stream)  (pool passes so far: "
                  f"{bucket['passes']})\n"
                  f"  accumulation happens on the NEXT sampler pass; place "
                  f"LTXCalibratorFlush after the sampler.")
        print(report)
        return (m, report)


class LTXCalibratorFlush:
    """
    Writes the accumulated running-mean E[x^2] mask for a bucket to disk.
    Place downstream of the sampler (feed it the sampler's LATENT) so it runs
    after the pass's hooks have fired. Passes the LATENT through unchanged.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "output_path": ("STRING", {
                    "default": "calib/override_agree.safetensors",
                    "tooltip": "Must match the calibrator's output_path.",
                }),
            },
            "optional": {
                "clear_after_flush": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Drop the in-memory bucket after writing. Leave "
                               "False while accumulating a pool.",
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "report")
    FUNCTION = "flush"
    CATEGORY = "10S Nodes/Calibration"
    DESCRIPTION = ("Writes a bucket's running-mean E[x^2] mask to a safetensors "
                   "file. Run after the sampler.")

    def flush(self, latent, output_path, clear_after_flush=False):
        bucket = _CALIB_STATE.get(output_path)
        if not bucket or not bucket["sum_sq"]:
            msg = (f"[10S] Flush: nothing accumulated for '{output_path}' yet "
                   f"(did the sampler run downstream of the calibrator?).")
            print(msg)
            return (latent, msg)

        # Persist RAW sum(x^2) + per-layer counts (not the mean) so the
        # accumulator can be reloaded and continued across sessions. Consumer
        # computes mean = sum_sq / count.
        sums = {}
        counts = {}
        total_tokens = 0
        for name, ss in bucket["sum_sq"].items():
            c = int(bucket["count"].get(name, 0))
            sums[name] = ss.to(torch.float32).contiguous()
            counts[name] = c
            total_tokens += c

        meta = {
            "calib_format": _CALIB_FORMAT,
            "calib_label": str(bucket.get("label", "")),
            "calib_passes": str(bucket.get("passes", 0)),
            "calib_layers": str(len(sums)),
            "calib_total_tokens": str(total_tokens),
            "calib_counts": json.dumps(counts),
            "calib_note": "raw sum(x^2) of linear inputs; mean = sum/count",
        }
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".",
                    exist_ok=True)
        save_file(sums, output_path, metadata=meta)

        if clear_after_flush:
            _CALIB_STATE.pop(output_path, None)

        report = (f"[10S] Flush: wrote {len(sums)} layer masks -> {output_path}\n"
                  f"  label={meta['calib_label']}  passes={meta['calib_passes']}  "
                  f"tokens={total_tokens}\n"
                  f"  (raw sum(x^2)+counts; reloadable & resumable across sessions)")
        print(report)
        return (latent, report)


NODE_CLASS_MAPPINGS = {
    "LTXActivationCalibrator": LTXActivationCalibrator,
    "LTXCalibratorFlush": LTXCalibratorFlush,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXActivationCalibrator": "\U0001f4ca LTX Activation Calibrator",
    "LTXCalibratorFlush": "\U0001f4be LTX Calibrator Flush",
}
