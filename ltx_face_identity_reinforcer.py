"""
ltx_face_identity_reinforcer.py — Unified i2v Identity Reinforcer

Single node that makes the Best-Face-ID LoRA workflow work correctly with
i2v conditioning by combining reference-latent injection, RoPE source-phase
tagging, face detection, and spatial identity gating into one drop-in node.

Replaces the LikenessAnchor pattern for identity workflows. LikenessGuide
can still be used for conditioning-level identity injection alongside this
node — the two mechanisms are complementary (Guide modifies cross-attention
conditioning, Reinforcer modifies self-attention token stream).

────────────────────────────────────────────────────────────────────────────
Best-Face-ID compatibility
────────────────────────────────────────────────────────────────────────────

The Alissonerdx/LTX-Best-Face-ID LoRA was trained with:
  - Reference latent concatenated at frame-0 grid (overlap)
  - source_id = 2 (reference) vs source_id = 0 (target)
  - phase_scale = 1.0
  - phase[d] = source_id · phase_scale · θ^(−d/L)   (θ = 10000)

v2 IMPLEMENTATION

Full Best-Face-ID mechanism per creator's spec:
  1. Reference tokens sit at IDENTICAL raw T/H/W coordinates as target
     (overlap layout — the coord grid is reused, no shift or multiplication)
  2. Reference tokens have denoise_mask=0 (always clean, never noised)
  3. Per-rotary-dimension phase rotation composed on top of RoPE:
        rate(d) = theta ** (-2d / L)
        extra_angle(d) = source_id * phase_scale * rate(d)
        cos_new = cos * cos_extra - sin * sin_extra
        sin_new = cos * sin_extra + sin * cos_extra
     Applied to first ref_len positions of self-attention frequencies only.
     Cross-attention (context-to-visual) untouched.  Target tokens
     (source_id=0) get a perfect no-op — base model behavior preserved.

The phase rotation is what disambiguates reference from target when both
are clean at the same spatial coordinate near end of sampling.  With source_id=2,
phase_scale=1.0 (Best-Face-ID defaults), reference tokens rotate into a
distinct phase band that attention can separate from target regardless of
noise state.

────────────────────────────────────────────────────────────────────────────
Workflow
────────────────────────────────────────────────────────────────────────────

  [Load Model]                 → MODEL
  [Load LTX-Best-Face-ID LoRA] → MODEL (on the MODEL path)
  [Load VAE]                   → VAE
  [Load Reference Image]       → IMAGE
  [Standard i2v Latent Setup]  → LATENT

  MODEL, VAE, IMAGE, LATENT ──→ LTX Face Identity Reinforcer ──→ MODEL' ──→ Sampler

────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

from .latent_likeness_guide import _detect_face_bbox
from .ltx_reference_enable import (
    _import_comfy,
    _log,
    _patched_process_input,
    _patched_prepare_timestep,
    _apply_patchifier_wrap,
    apply_global_patches,
)


# ────────────────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────────────────

def _pad_image_to_multiple(image_bhwc: torch.Tensor, divisor: int = 32) -> torch.Tensor:
    """Pad (B, H, W, C) so H and W are multiples of divisor."""
    if image_bhwc.dim() != 4:
        raise ValueError(f"Expected IMAGE (B,H,W,C), got {tuple(image_bhwc.shape)}")
    B, H, W, C = image_bhwc.shape
    pad_h = (divisor - H % divisor) % divisor
    pad_w = (divisor - W % divisor) % divisor
    if pad_h == 0 and pad_w == 0:
        return image_bhwc
    x = image_bhwc.permute(0, 3, 1, 2).contiguous()
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x.permute(0, 2, 3, 1).contiguous()


def _resize_image_to_latent(image_bhwc: torch.Tensor,
                             target_latent: torch.Tensor,
                             vae_scale: int = 32) -> torch.Tensor:
    """
    Resize a (B,H,W,C) image so its VAE-encoded latent will match the
    target latent's spatial dimensions.  target_latent may be (B,C,F,H,W)
    or a dict with "samples" key.
    """
    if isinstance(target_latent, dict) and "samples" in target_latent:
        target_latent = target_latent["samples"]
    if target_latent.dim() == 5:
        Ht = int(target_latent.shape[3]) * vae_scale
        Wt = int(target_latent.shape[4]) * vae_scale
    elif target_latent.dim() == 4:
        Ht = int(target_latent.shape[2]) * vae_scale
        Wt = int(target_latent.shape[3]) * vae_scale
    else:
        return image_bhwc

    B, H, W, C = image_bhwc.shape
    if (H, W) == (Ht, Wt):
        return image_bhwc
    x = image_bhwc.permute(0, 3, 1, 2).contiguous()
    x = F.interpolate(x, size=(Ht, Wt), mode="bicubic", align_corners=False)
    return x.permute(0, 2, 3, 1).contiguous().clamp(0.0, 1.0)




def _align_to_reference_bbox(
    image_bhwc: torch.Tensor,
    face_bbox: Tuple[float, float, float, float],
    target_face_bbox: Tuple[float, float, float, float],
    target_H: int,
    target_W: int,
    debug: bool = False,
) -> Tuple[torch.Tensor, Tuple[float, float, float, float]]:
    """
    Crop and pad an image so its detected face bbox lands at exactly the same
    normalized position and size as target_face_bbox within a canvas of
    (target_H, target_W).  Used to align secondary references to the
    primary reference so the model sees consistent face proportions and
    positions across all identity signals.

    Args:
      image_bhwc      : (B, H, W, C) source image
      face_bbox       : (x1,y1,x2,y2) normalized detected face in source
      target_face_bbox: (x1,y1,x2,y2) normalized where the face should land
      target_H/W      : output canvas dimensions in pixels
      debug           : logging

    Returns:
      (aligned_image [B, target_H, target_W, C], target_face_bbox)
    """
    B, sH, sW, C = image_bhwc.shape
    fx1, fy1, fx2, fy2 = face_bbox
    tx1, ty1, tx2, ty2 = target_face_bbox

    # Source face pixel dimensions
    src_face_w = (fx2 - fx1) * sW
    src_face_h = (fy2 - fy1) * sH
    if src_face_w <= 0 or src_face_h <= 0:
        # Fallback: just resize
        x = image_bhwc.permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(x, size=(target_H, target_W),
                          mode="bicubic", align_corners=False)
        return x.permute(0, 2, 3, 1).contiguous().clamp(0.0, 1.0), target_face_bbox

    # Target face pixel dimensions in the output canvas
    tgt_face_w = (tx2 - tx1) * target_W
    tgt_face_h = (ty2 - ty1) * target_H

    # Scale factor: use whichever dimension is more constraining to preserve
    # face proportions (uniform scaling — no stretching)
    scale_x = tgt_face_w / src_face_w
    scale_y = tgt_face_h / src_face_h
    # Use average scale to preserve face aspect ratio without distortion
    # (if scale_x differs significantly from scale_y, we lose exact target
    # bbox match but keep face proportions correct)
    scale = (scale_x + scale_y) * 0.5

    # New source dimensions after scaling
    new_sW = int(round(sW * scale))
    new_sH = int(round(sH * scale))
    if new_sW <= 0 or new_sH <= 0:
        x = image_bhwc.permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(x, size=(target_H, target_W),
                          mode="bicubic", align_corners=False)
        return x.permute(0, 2, 3, 1).contiguous().clamp(0.0, 1.0), target_face_bbox

    # Resize source to scaled dimensions
    x = image_bhwc.permute(0, 3, 1, 2).contiguous()
    x = F.interpolate(x, size=(new_sH, new_sW),
                      mode="bicubic", align_corners=False)
    x = x.permute(0, 2, 3, 1).contiguous().clamp(0.0, 1.0)

    # After scaling, face center in scaled image (pixel coords)
    face_cx_scaled = ((fx1 + fx2) * 0.5) * new_sW
    face_cy_scaled = ((fy1 + fy2) * 0.5) * new_sH

    # Target face center in output canvas (pixel coords)
    tgt_cx = ((tx1 + tx2) * 0.5) * target_W
    tgt_cy = ((ty1 + ty2) * 0.5) * target_H

    # We need scaled image to be placed so face center matches target center.
    # Offset in output canvas where scaled image's origin (0,0) should land:
    origin_x = tgt_cx - face_cx_scaled  # positive = shift right
    origin_y = tgt_cy - face_cy_scaled

    # Create output canvas and paste
    canvas = torch.zeros(B, target_H, target_W, C,
                         dtype=image_bhwc.dtype, device=image_bhwc.device)

    # Source region in scaled image that maps into canvas
    src_x1 = max(0, int(round(-origin_x)))
    src_y1 = max(0, int(round(-origin_y)))
    src_x2 = min(new_sW, int(round(target_W - origin_x)))
    src_y2 = min(new_sH, int(round(target_H - origin_y)))

    # Destination region in canvas
    dst_x1 = max(0, int(round(origin_x)))
    dst_y1 = max(0, int(round(origin_y)))
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if src_x2 > src_x1 and src_y2 > src_y1:
        canvas[:, dst_y1:dst_y2, dst_x1:dst_x2, :] = x[:, src_y1:src_y2,
                                                       src_x1:src_x2, :]

    # Fill empty regions with edge-replicated content from canvas edges
    # Simple approach: replicate the edge rows/columns of the pasted region
    if dst_y1 > 0 and dst_y2 > dst_y1:
        canvas[:, :dst_y1, dst_x1:dst_x2, :] = canvas[:, dst_y1:dst_y1+1,
                                                     dst_x1:dst_x2, :]
    if dst_y2 < target_H and dst_y2 > dst_y1:
        canvas[:, dst_y2:, dst_x1:dst_x2, :] = canvas[:, dst_y2-1:dst_y2,
                                                     dst_x1:dst_x2, :]
    if dst_x1 > 0:
        canvas[:, :, :dst_x1, :] = canvas[:, :, dst_x1:dst_x1+1, :]
    if dst_x2 < target_W:
        canvas[:, :, dst_x2:, :] = canvas[:, :, dst_x2-1:dst_x2, :]

    if debug:
        print(f"  [Reinforcer] aligned ref2 to ref1: "
              f"src {sW}x{sH} scaled by {scale:.2f}x → {new_sW}x{new_sH}, "
              f"canvas {target_W}x{target_H}, face at "
              f"({tx1:.2f},{ty1:.2f},{tx2:.2f},{ty2:.2f})")

    return canvas, target_face_bbox

def _auto_face_crop(
    image_bhwc: torch.Tensor,
    face_bbox: Tuple[float, float, float, float],
    zoom_factor: float = 2.0,
    target_aspect: Optional[float] = None,
    debug: bool = False,
) -> Tuple[torch.Tensor, Tuple[float, float, float, float]]:
    """
    Crop an image around the detected face with generous context padding,
    then resize/pad to match the target aspect ratio.

    Result: VAE gets much more face detail to encode without stretching
    or distorting the face itself.

    Args:
      image_bhwc     : (B, H, W, C) image in [0,1]
      face_bbox      : (x1, y1, x2, y2) normalized 0-1
      zoom_factor    : 2.0 = crop extends 2x the face bbox in each direction
                       Higher = more context.  Lower = tighter face focus.
      target_aspect  : W/H ratio to match (None = keep source ratio)
      debug          : logging

    Returns:
      (cropped_image, new_face_bbox_normalized)
    """
    B, H, W, C = image_bhwc.shape
    x1, y1, x2, y2 = face_bbox
    fx1_px, fy1_px = x1 * W, y1 * H
    fx2_px, fy2_px = x2 * W, y2 * H
    face_w = fx2_px - fx1_px
    face_h = fy2_px - fy1_px
    fcx = (fx1_px + fx2_px) * 0.5
    fcy = (fy1_px + fy2_px) * 0.5

    # Desired crop size: zoom_factor * face size, matching target aspect
    if target_aspect is None:
        target_aspect = W / max(H, 1)

    # Base crop dimensions from zoom
    base_h = face_h * zoom_factor
    base_w = face_w * zoom_factor

    # Enforce target aspect ratio by expanding the smaller dimension
    if base_w / max(base_h, 1e-6) < target_aspect:
        # Too tall for aspect — widen
        base_w = base_h * target_aspect
    else:
        # Too wide for aspect — heighten
        base_h = base_w / target_aspect

    # Compute crop bounds
    cx1 = fcx - base_w * 0.5
    cy1 = fcy - base_h * 0.5
    cx2 = fcx + base_w * 0.5
    cy2 = fcy + base_h * 0.5

    # Track padding needed if crop extends outside image bounds
    pad_left   = max(0.0, -cx1)
    pad_top    = max(0.0, -cy1)
    pad_right  = max(0.0, cx2 - W)
    pad_bottom = max(0.0, cy2 - H)

    # Clamp crop to image bounds
    cx1c = int(max(0, cx1))
    cy1c = int(max(0, cy1))
    cx2c = int(min(W, cx2))
    cy2c = int(min(H, cy2))

    if cx2c <= cx1c or cy2c <= cy1c:
        if debug:
            print(f"  [Reinforcer] auto-crop bounds invalid, using full image")
        return image_bhwc

    # Crop
    cropped = image_bhwc[:, cy1c:cy2c, cx1c:cx2c, :]

    # Pad if needed (edge-replicate to avoid black borders)
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        x = cropped.permute(0, 3, 1, 2).contiguous()
        x = F.pad(
            x,
            (int(pad_left), int(pad_right), int(pad_top), int(pad_bottom)),
            mode="replicate",
        )
        cropped = x.permute(0, 2, 3, 1).contiguous()

    # Compute face bbox in the new cropped+padded image space.
    # The transform: crop uses [cy1c:cy2c, cx1c:cx2c] from original, then pads
    # (pad_left, pad_right, pad_top, pad_bottom).  A pixel at original (px, py)
    # lands at:
    #     x_new = (px - cx1c) + pad_left
    #     y_new = (py - cy1c) + pad_top
    # Reference: cx1c, cy1c are the ACTUAL slicing offsets (0 or positive),
    # not the (possibly negative) desired crop bounds cx1, cy1.
    new_H, new_W = cropped.shape[1], cropped.shape[2]
    new_fx1 = (fx1_px - cx1c) + pad_left
    new_fy1 = (fy1_px - cy1c) + pad_top
    new_fx2 = new_fx1 + face_w
    new_fy2 = new_fy1 + face_h
    new_bbox = (
        max(0.0, new_fx1 / max(new_W, 1)),
        max(0.0, new_fy1 / max(new_H, 1)),
        min(1.0, new_fx2 / max(new_W, 1)),
        min(1.0, new_fy2 / max(new_H, 1)),
    )

    if debug:
        face_frac = (face_w * face_h) / max(new_H * new_W, 1)
        print(f"  [Reinforcer] auto-crop: {W}x{H} -> {new_W}x{new_H}, "
              f"face fills {face_frac*100:.1f}% of frame "
              f"(was {(face_w*face_h)/(W*H)*100:.1f}%), "
              f"new bbox: ({new_bbox[0]:.2f},{new_bbox[1]:.2f},"
              f"{new_bbox[2]:.2f},{new_bbox[3]:.2f})")

    return cropped, new_bbox


# ────────────────────────────────────────────────────────────────────────────
# RoPE source-phase (Best-Face-ID convention)
# ────────────────────────────────────────────────────────────────────────────

def _apply_source_phase(
    ref_pixel_coords: torch.Tensor,
    source_id: float = 2.0,
    phase_scale: float = 1.0,
) -> torch.Tensor:
    """
    v1: No-op.  The proper mechanism (per-dimension phase rotation composed
    with RoPE output at attention time) is not implemented at the coordinate
    level — it requires hooking the attention module's RoPE application.
    Deferred to v2.  Reference tokens use pure overlap coords in v1.
    """
    return ref_pixel_coords


# ────────────────────────────────────────────────────────────────────────────
# Face-mask spatial gating
# ────────────────────────────────────────────────────────────────────────────

def _make_face_mask_latent(
    face_bbox: Optional[Tuple[float, float, float, float]],
    latent_shape: Tuple[int, int, int, int, int],
    gating_mode: str = "mask_soft",
    dilation: float = 0.10,
) -> Optional[torch.Tensor]:
    """
    Construct a spatial mask in latent space that concentrates identity
    influence within the face region and softly falls off outside.

    Returns None for gating_mode='off' or missing bbox — caller then
    applies identity influence uniformly across the frame.

    Args:
      face_bbox   : (x1, y1, x2, y2) normalized 0-1 in image space
      latent_shape: (B, C, F, H, W)
      gating_mode : 'mask_soft' | 'mask_hard' | 'off'
      dilation    : bbox expansion fraction to include hair/context

    Returns:
      (1, 1, 1, H, W) mask tensor, broadcastable over batch/channel/frame,
      values in [0, 1].
    """
    if gating_mode == "off" or face_bbox is None:
        return None

    B, C, F_lat, H, W = latent_shape
    x1, y1, x2, y2 = face_bbox

    # Dilate the bbox
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    hw = (x2 - x1) * (0.5 + dilation)
    hh = (y2 - y1) * (0.5 + dilation)
    x1d = max(0.0, cx - hw)
    y1d = max(0.0, cy - hh)
    x2d = min(1.0, cx + hw)
    y2d = min(1.0, cy + hh)

    mask = torch.zeros(H, W, dtype=torch.float32)

    if gating_mode == "mask_hard":
        px1 = int(x1d * W)
        py1 = int(y1d * H)
        px2 = int(x2d * W)
        py2 = int(y2d * H)
        mask[py1:py2, px1:px2] = 1.0
    else:
        # mask_soft: cosine falloff around bbox
        yy = torch.arange(H, dtype=torch.float32).view(-1, 1) / H
        xx = torch.arange(W, dtype=torch.float32).view(1, -1) / W

        # Distance from bbox center, normalized to bbox half-extents
        dx = (xx - cx).abs() / max(hw, 1e-6)
        dy = (yy - cy).abs() / max(hh, 1e-6)
        dist = torch.maximum(dx, dy)   # L∞ distance in bbox units

        # 1.0 inside bbox (dist ≤ 1), cosine falloff to 0.0 by dist=1.5
        soft = torch.where(
            dist <= 1.0,
            torch.ones_like(dist),
            torch.where(
                dist >= 1.5,
                torch.zeros_like(dist),
                0.5 * (1.0 + torch.cos((dist - 1.0) * math.pi / 0.5))
            ),
        )
        mask = soft

    return mask.view(1, 1, 1, H, W)


# ────────────────────────────────────────────────────────────────────────────
# Position mode: i2v_safe
# ────────────────────────────────────────────────────────────────────────────

def _shift_to_i2v_safe(ref_pixel_coords: torch.Tensor) -> torch.Tensor:
    """
    DEPRECATED — kept for backward compatibility only.

    Per Best-Face-ID creator: no negative additive shift is needed.
    The multiplicative phase applied via source_id * phase_scale
    already places reference tokens in a distinct RoPE space slice
    that doesn't collide with i2v frame 0.  This function is now a no-op.
    """
    return ref_pixel_coords


# ────────────────────────────────────────────────────────────────────────────
# Main node
# ────────────────────────────────────────────────────────────────────────────

class LTXFaceIdentityReinforcer:
    """
    Drop-in identity reinforcer for LTX-Best-Face-ID LoRA workflows.
    Makes Best-Face-ID work with i2v conditioning by placing reference
    tokens at a non-conflicting RoPE position while preserving the
    source_id=2 phase tag the LoRA was trained on.

    Composes:
      - VAE encoding of reference image
      - Face detection with configurable padding
      - Reference token injection with i2v_safe placement
      - RoPE source-phase tagging (source_id=2, phase_scale=1.0)
      - Optional spatial identity gating via face bbox mask
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":            ("MODEL",),
                "vae":              ("VAE",),
                "reference_image":  ("IMAGE",),
                "target_latent":    ("LATENT",),
            },
            "optional": {
                "identity_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Scales reference latent magnitude. 1.0 = Best-Face-ID default.",
                }),
                "face_padding": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 0.5, "step": 0.05,
                    "tooltip": "Face bbox expansion — captures hair/neck context.",
                }),
                "auto_face_crop": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When a face is detected, auto-crop the reference "
                               "image around the face at zoom_factor extent and "
                               "match target aspect ratio. Dramatically improves "
                               "identity transfer for wide/full-body references "
                               "by giving the VAE much more face detail to encode. "
                               "Turn off if reference is already tightly cropped.",
                }),
                "crop_zoom_factor": ("FLOAT", {
                    "default": 2.0, "min": 1.2, "max": 4.0, "step": 0.1,
                    "tooltip": "How much context around the face to include. "
                               "2.0 = crop is 2x the face bbox (shoulders + hair). "
                               "1.5 = very tight (face + hair only). "
                               "3.0 = wide (upper body). Ignored if auto_face_crop off.",
                }),
                "spatial_gating": (["mask_soft", "mask_hard", "off"], {
                    "default": "mask_soft",
                    "tooltip": "Constrain identity influence to face region. "
                               "mask_soft = cosine falloff (recommended). "
                               "mask_hard = binary. off = uniform (raw Best-Face-ID).",
                }),
                "placement_mode": (["i2v_safe", "t2v_overlap", "prefix"], {
                    "default": "i2v_safe",
                    "tooltip": "i2v_safe / t2v_overlap = pure overlap layout "
                               "(Best-Face-ID's 'what we use' default). Reference "
                               "reuses target's coord grid, disambiguated by clean/"
                               "noisy state and sequence position. "
                               "prefix = additive offset (legacy).",
                }),
                "source_id": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 8.0, "step": 1.0,
                    "tooltip": "RoPE source tag applied via v2 phase rotation. "
                               "Best-Face-ID LoRA expects 2.0. source_id=0 disables "
                               "rotation (falls back to overlap-only v1 behavior).",
                }),
                "phase_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "Phase rotation magnitude multiplier. Best-Face-ID LoRA "
                               "expects 1.0. Lower values reduce reference/target "
                               "separation strength.",
                }),
                "reference_image_2": ("IMAGE", {
                    "tooltip": "Optional secondary reference (multi-subject). Uses source_id=3.",
                }),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "reinforce"
    CATEGORY = "10S Nodes/Identity"
    DESCRIPTION = (
        "Unified LTX-Best-Face-ID identity reinforcer. Combines reference "
        "latent injection, RoPE source-phase tagging, face detection, and "
        "spatial mask gating in a single node. Makes Best-Face-ID work "
        "correctly alongside i2v frame 0 conditioning by placing reference "
        "tokens at a distinct RoPE position while preserving the source_id=2 "
        "tag the LoRA was trained against. Replaces LikenessAnchor for "
        "identity workflows."
    )

    def reinforce(
        self,
        model,
        vae,
        reference_image,
        target_latent,
        identity_strength: float = 1.0,
        face_padding: float = 0.15,
        auto_face_crop: bool = True,
        crop_zoom_factor: float = 2.0,
        spatial_gating: str = "mask_soft",
        placement_mode: str = "i2v_safe",
        source_id: float = 2.0,
        phase_scale: float = 1.0,
        reference_image_2 = None,
        debug: bool = False,
    ):
        # ── 1. Ensure patches are installed (class + instance level) ─────
        _install_all_patches(model, debug=debug)

        # ── 2. Face detection FIRST (before encoding, so we can auto-crop) ──
        # Detect on the primary reference at original resolution — detection
        # quality is best there before any resize squashing.
        face_bbox = None
        face_bbox_secondary = None

        def _detect_on(img_tensor):
            try:
                img_np = (img_tensor[0].cpu().clamp(0, 1) * 255.0
                          ).to(torch.uint8).numpy()
                return _detect_face_bbox(img_np, padding=face_padding, debug=debug)
            except Exception as e:
                if debug:
                    print(f"  [Reinforcer] face detect failed: {e}")
                return None

        face_bbox = _detect_on(reference_image)
        if reference_image_2 is not None:
            face_bbox_secondary = _detect_on(reference_image_2)

        if debug:
            if face_bbox is not None:
                print(f"  [Reinforcer] primary face bbox: {face_bbox}")
            else:
                print(f"  [Reinforcer] primary: no face found")

        # ── 3. Auto-crop references around detected faces ────────────────
        # Determine target aspect ratio from target latent for aspect matching
        if isinstance(target_latent, dict):
            _tlat = target_latent["samples"]
        else:
            _tlat = target_latent
        if _tlat.dim() == 5:
            _tH = int(_tlat.shape[3])
            _tW = int(_tlat.shape[4])
        elif _tlat.dim() == 4:
            _tH = int(_tlat.shape[2])
            _tW = int(_tlat.shape[3])
        else:
            _tH, _tW = 1, 1
        _target_aspect = _tW / max(_tH, 1)

        primary_post_crop_bbox = None    # face position in post-crop image space

        def _prepare_ref(img_tensor, bbox, label=""):
            """Auto-crop (if enabled + face found) then resize/pad for VAE.
            Returns (processed_image, post_crop_bbox_or_None)."""
            if img_tensor is None:
                return None, None
            processed = img_tensor
            post_crop_bbox = bbox
            if auto_face_crop and bbox is not None:
                processed, post_crop_bbox = _auto_face_crop(
                    processed, bbox,
                    zoom_factor=crop_zoom_factor,
                    target_aspect=_target_aspect,
                    debug=debug,
                )
            processed = _resize_image_to_latent(processed, target_latent, vae_scale=32)
            processed = _pad_image_to_multiple(processed, divisor=32)
            return processed, post_crop_bbox

        ref_prepared = []
        primary_prepared, primary_post_crop_bbox = _prepare_ref(
            reference_image, face_bbox, "primary")
        if primary_prepared is not None:
            ref_prepared.append(primary_prepared)
        if reference_image_2 is not None:
            # If we have both a primary post-crop bbox AND a detected face in
            # ref2, align ref2 so its face lands at exactly the same position
            # and scale as ref1's face.  This eliminates aspect/proportion
            # desync that causes composition drift.
            if (primary_prepared is not None
                    and primary_post_crop_bbox is not None
                    and face_bbox_secondary is not None
                    and auto_face_crop):
                _, prim_H, prim_W, _ = primary_prepared.shape
                aligned_ref2, _ = _align_to_reference_bbox(
                    reference_image_2, face_bbox_secondary,
                    primary_post_crop_bbox, prim_H, prim_W, debug=debug,
                )
                sec_prepared = _pad_image_to_multiple(aligned_ref2, divisor=32)
                ref_prepared.append(sec_prepared)
                if debug:
                    print(f"  [Reinforcer] ref2 aligned to match ref1 "
                          f"face position/scale")
            else:
                # Fallback: standard preparation (no alignment)
                sec_prepared, _ = _prepare_ref(
                    reference_image_2, face_bbox_secondary, "secondary")
                if sec_prepared is not None:
                    ref_prepared.append(sec_prepared)

        # ── 4. VAE encode all prepared references ────────────────────────
        ref_latents = []
        for idx, img in enumerate(ref_prepared):
            try:
                lat = vae.encode(img)
            except Exception as e:
                if debug:
                    print(f"  [Reinforcer] VAE encode failed on ref {idx}: {e}")
                continue
            if lat.dim() == 4:
                lat = lat.unsqueeze(2)
            ref_latents.append(lat * identity_strength)

        if not ref_latents:
            if debug:
                print("  [Reinforcer] no valid reference latents; passing through")
            return (model,)

        # Concatenate along temporal dim — each ref is one "reference frame"
        if len(ref_latents) > 1:
            primary_ref = torch.cat(ref_latents, dim=2)
            if debug:
                print(f"  [Reinforcer] concatenated {len(ref_latents)} refs "
                      f"→ latent shape {tuple(primary_ref.shape)}")
        else:
            primary_ref = ref_latents[0]

        # Spatial gating mask uses the ACTUAL post-crop face position (tracked
        # through the crop+pad transform), not a centered assumption which
        # was wrong for aspect-padded crops.
        if auto_face_crop and primary_post_crop_bbox is not None:
            face_bbox_for_mask = primary_post_crop_bbox
        else:
            face_bbox_for_mask = face_bbox

        if debug:
            if face_bbox_for_mask is not None:
                print(f"  [Reinforcer] mask bbox: "
                      f"({face_bbox_for_mask[0]:.2f},{face_bbox_for_mask[1]:.2f},"
                      f"{face_bbox_for_mask[2]:.2f},{face_bbox_for_mask[3]:.2f})")
            else:
                print(f"  [Reinforcer] gating disabled (no bbox)")

        # ── 4. Build spatial mask (if applicable) ────────────────────────
        if isinstance(target_latent, dict):
            target_shape = tuple(target_latent["samples"].shape)
        else:
            target_shape = tuple(target_latent.shape)

        if len(target_shape) == 4:
            target_shape = (target_shape[0], target_shape[1], 1,
                           target_shape[2], target_shape[3])

        face_mask = _make_face_mask_latent(
            face_bbox=face_bbox_for_mask,
            latent_shape=target_shape,
            gating_mode=spatial_gating,
            dilation=face_padding,
        )

        # ── 5. Attach reference to model via transformer_options ─────────
        m = model.clone()
        model_options = m.model_options.setdefault("transformer_options", {})

        # Primary reference — source_id=2 (Best-Face-ID)
        model_options["reference_latent"]        = primary_ref
        model_options["reference_position_mode"] = placement_mode
        model_options["reference_source_id"]     = source_id
        model_options["reference_phase_scale"]   = phase_scale

        # Spatial mask for identity gating
        if face_mask is not None:
            model_options["reference_spatial_mask"] = face_mask
            model_options["reference_mask_gating"]  = spatial_gating

        if debug:
            print(f"  [Reinforcer] attached: strength={identity_strength}, "
                  f"placement={placement_mode}, source_id={source_id}, "
                  f"phase={phase_scale}, gating={spatial_gating}")

        return (m,)


# ────────────────────────────────────────────────────────────────────────────
# Patch installation (extends ltx_reference_enable for source_id support)
# ────────────────────────────────────────────────────────────────────────────

def _install_all_patches(model, debug: bool = False):
    """
    Install all three levels of patches needed for reference injection:
      1. Class-level: LTXAVModel._process_input and _prepare_timestep
         (idempotent, applies once per process)
      2. Instance-level: patchifier.unpatchify wrap on this model instance
         so reference tokens are stripped before output reshape

    Without step 2, the transformer sees the extra reference tokens correctly
    but the output stage tries to unpatchify them as target tokens, producing
    the "8100 != 7500" einops shape mismatch.
    """
    # Step 1: class-level patches (uses the tested apply_global_patches)
    try:
        ok = apply_global_patches()
        if debug and ok:
            print("  [Reinforcer] class-level patches installed (or already were)")
    except Exception as e:
        print(f"  [Reinforcer] ⚠ class-level patch failed: {type(e).__name__}: {e}")
        return False

    # Step 2: instance-level patchifier wrap
    # Traverse model → model → diffusion_model to reach the LTXAVModel instance
    try:
        # ModelPatcher wraps a BaseModel that has `.diffusion_model`
        inner = model.model if hasattr(model, "model") else model
        av_instance = inner.diffusion_model if hasattr(inner, "diffusion_model") else inner

        if av_instance is None or not hasattr(av_instance, "patchifier"):
            if debug:
                print(f"  [Reinforcer] ⚠ could not locate LTXAVModel instance "
                      f"(got {type(av_instance).__name__})")
            return False

        wrapped_now = _apply_patchifier_wrap(av_instance)
        if debug:
            state = "wrapped now" if wrapped_now else "already wrapped"
            print(f"  [Reinforcer] patchifier {state} on {type(av_instance).__name__}")
        return True
    except Exception as e:
        print(f"  [Reinforcer] ⚠ instance patch failed: {type(e).__name__}: {e}")
        return False


# ────────────────────────────────────────────────────────────────────────────
# Registration
# ────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "LTXFaceIdentityReinforcer": LTXFaceIdentityReinforcer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXFaceIdentityReinforcer": "\U0001f9d1 LTX Face Identity Reinforcer",
}
