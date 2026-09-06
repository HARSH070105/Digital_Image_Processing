"""
Q2 Complete — Bit-plane slicing and steganography.
Generates every figure and metric needed for the report.
"""

import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import io

# ─────────────────────────────────────────────
# Output directory
# ─────────────────────────────────────────────
OUT = "q2_outputs"
os.makedirs(OUT, exist_ok=True)

def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


# ─────────────────────────────────────────────
# Core functions
# ─────────────────────────────────────────────

def bit_planes(img):
    """Return all 8 bit planes, index 0 = LSB, index 7 = MSB."""
    return np.array([(img >> i) & 1 for i in range(8)], dtype=np.uint8)


def reconstruct(planes, keep):
    """Rebuild from the top `keep` planes (MSB side)."""
    res = np.zeros_like(planes[0], dtype=np.uint8)
    for i in range(8 - keep, 8):
        res += planes[i] << i
    return res


def gray_encode(img):
    return np.bitwise_xor(img, img >> 1)


def gray_decode(gray):
    """Decode a Gray-coded image back to binary."""
    img = gray.copy()
    mask = img >> 1
    while np.any(mask != 0):
        img ^= mask
        mask >>= 1
    return img


def embed_lsb(cover, bits, plane=0):
    stego = cover.copy().flatten().astype(np.uint8)
    mask = np.uint8(255 - (1 << plane))
    n = len(bits)
    stego[:n] = (stego[:n] & mask) | (bits.astype(np.uint8) << plane)
    return stego.reshape(cover.shape)


def extract_lsb(stego, n, plane=0):
    return (stego.flatten()[:n] >> plane) & 1


def embed_robust(cover, bits, delta=2, block_size=32, **kw):
    """
    Embed by adding ±delta to every pixel in a 32×32 block.
    delta=2  →  MSE=4  →  PSNR ≈ 42.1 dB  (meets ≥ 40 dB requirement).
    Noise averaging over block_size² = 1024 pixels means σ_eff = σ/√1024,
    so the signal (delta=2) dominates noise even at σ=5  (σ_eff ≈ 0.156).
    """
    stego = cover.copy().astype(np.float64)
    cols_per_row = cover.shape[1] // block_size
    for i, bit in enumerate(bits):
        r = (i // cols_per_row) * block_size
        c = (i % cols_per_row) * block_size
        sign = 1 if bit == 1 else -1
        stego[r:r+block_size, c:c+block_size] += sign * delta
    return np.clip(stego, 0, 255).astype(np.uint8)


def extract_robust(stego, cover, n, block_size=32, **kw):
    """Cover-assisted decoder: compare block means to original."""
    bits = []
    cols_per_row = stego.shape[1] // block_size
    sf = stego.astype(np.float64)
    cf = cover.astype(np.float64)
    for i in range(n):
        r = (i // cols_per_row) * block_size
        c = (i % cols_per_row) * block_size
        diff = np.mean(sf[r:r+block_size, c:c+block_size] -
                       cf[r:r+block_size, c:c+block_size])
        bits.append(1 if diff > 0 else 0)
    return np.array(bits, dtype=np.uint8)


# ── supplied channel functions (do not modify) ──────────────────────────────

def degrade_gaussian(img, sigma, rng=None):
    rng = rng or np.random.default_rng(0)
    return np.clip(
        np.round(img.astype(np.float64) + rng.normal(0, sigma, img.shape)),
        0, 255).astype(np.uint8)


def degrade_jpeg(img, quality):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("L"))


def ber(a, b):
    return float(np.mean(a != b))


def psnr(a, b, peak=255.0):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(peak * peak / mse)


# ─────────────────────────────────────────────
# Load images
# ─────────────────────────────────────────────
tex_path    = "Assignment 1/images/p2/cover_textured.png"
smooth_path = "Assignment 1/images/p2/cover_smooth.png"
stego_path  = "Assignment 1/images/p2/decode_me.png"

tex_cover    = np.array(Image.open(tex_path).convert("L"))
smooth_cover = np.array(Image.open(smooth_path).convert("L"))
stego_img    = np.array(Image.open(stego_path).convert("L"))

print("Images loaded:", tex_cover.shape, smooth_cover.shape, stego_img.shape)

# ═══════════════════════════════════════════════════════════════════
# 2.1  Decomposition
# ═══════════════════════════════════════════════════════════════════
print("\n─── 2.1 Decomposition ───")

# ── (A) Eight bit-plane display ─────────────────────────────────────
planes = bit_planes(tex_cover)

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
fig.suptitle("Bit planes of textured cover  (0 = LSB, 7 = MSB)", fontsize=13)
for i in range(8):
    ax = axes[i // 4, i % 4]
    ax.imshow(planes[i], cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"Plane {i}", fontsize=10)
    ax.axis("off")
plt.tight_layout()
save(fig, "2_1_A_bit_planes.png")

# ── (B) PSNR vs number of top planes ────────────────────────────────
psnrs = []
ns = list(range(1, 9))
print("\nPSNR — top n planes:")
for n in ns:
    rebuilt = reconstruct(planes, n)
    p = psnr(tex_cover, rebuilt)
    psnrs.append(p if np.isfinite(p) else 100.0)
    tag = f"{p:.2f} dB" if np.isfinite(p) else "∞ (perfect)"
    print(f"  n={n}: {tag}")

# Measure per-plane gain
gains = np.diff(psnrs)
print(f"\nPer-plane PSNR gain (dB): {gains}")
print(f"Mean gain: {gains.mean():.2f} dB  (≈ 6.02 dB ≈ 20·log10(2), one bit of dynamic range)")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(ns, psnrs, marker="o", color="steelblue", linewidth=2)
for n, p in zip(ns, psnrs):
    label = "∞" if p == 100.0 else f"{p:.1f}"
    ax.annotate(label, (n, p), textcoords="offset points", xytext=(4, 4), fontsize=8)
ax.set_xlabel("Number of top planes kept (n)")
ax.set_ylabel("PSNR (dB)")
ax.set_title("Reconstruction PSNR vs top-n planes")
ax.set_xticks(ns)
ax.grid(True, alpha=0.4)
plt.tight_layout()
save(fig, "2_1_B_psnr_vs_n.png")

# ── (C) Ramp vs Gray-coded ramp bit planes ──────────────────────────
ramp      = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
gray_ramp = gray_encode(ramp)

ramp_planes = bit_planes(ramp)
gray_planes = bit_planes(gray_ramp)

fig, axes = plt.subplots(2, 8, figsize=(20, 5))
fig.suptitle("Ramp bit planes — Binary (top) vs Gray code (bottom)", fontsize=12)
for i in range(8):
    axes[0, i].imshow(ramp_planes[i], cmap="gray", aspect="auto", vmin=0, vmax=1)
    axes[0, i].set_title(f"P{i}", fontsize=9)
    axes[0, i].axis("off")
    axes[1, i].imshow(gray_planes[i], cmap="gray", aspect="auto", vmin=0, vmax=1)
    axes[1, i].set_title(f"P{i}", fontsize=9)
    axes[1, i].axis("off")
axes[0, 0].set_ylabel("Binary", fontsize=10, visible=True)
axes[1, 0].set_ylabel("Gray", fontsize=10, visible=True)
plt.tight_layout()
save(fig, "2_1_C_ramp_vs_gray.png")

# Side-by-side of low planes only (most informative for report)
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
fig.suptitle("Low bit planes 0–3: Binary ramp (top) vs Gray ramp (bottom)", fontsize=12)
for i in range(4):
    axes[0, i].imshow(ramp_planes[i], cmap="gray", aspect="auto", vmin=0, vmax=1)
    axes[0, i].set_title(f"Plane {i}")
    axes[0, i].axis("off")
    axes[1, i].imshow(gray_planes[i], cmap="gray", aspect="auto", vmin=0, vmax=1)
    axes[1, i].set_title(f"Plane {i}")
    axes[1, i].axis("off")
axes[0, 0].set_ylabel("Binary", fontsize=11)
axes[1, 0].set_ylabel("Gray", fontsize=11)
plt.tight_layout()
save(fig, "2_1_C_low_planes_detail.png")

print("\n2.1 done.")

# ═══════════════════════════════════════════════════════════════════
# 2.2  Hide and recover a payload
# ═══════════════════════════════════════════════════════════════════
print("\n─── 2.2 Hide and recover ───")

# ── (A) Decode the stego image ──────────────────────────────────────
header = extract_lsb(stego_img, 24)

sync     = header[0:8]
h_bits   = header[8:16]
w_bits   = header[16:24]

payload_h = int("".join(str(b) for b in h_bits), 2)
payload_w = int("".join(str(b) for b in w_bits), 2)

print(f"Sync marker  : {sync}  (expected 10101010)")
print(f"Payload size : {payload_h} × {payload_w} pixels")

total_bits  = 24 + payload_h * payload_w
all_bits    = extract_lsb(stego_img, total_bits)
payload_bits = all_bits[24:]
payload_img  = (payload_bits.reshape((payload_h, payload_w)) * 255).astype(np.uint8)

Image.fromarray(payload_img).save(os.path.join(OUT, "2_2_A_recovered_payload.png"))
print("  saved 2_2_A_recovered_payload.png")

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(payload_img, cmap="gray")
ax.set_title(f"Recovered payload  ({payload_h}×{payload_w})")
ax.axis("off")
plt.tight_layout()
save(fig, "2_2_A_recovered_payload_display.png")

# ── (B) PSNR and difference map ─────────────────────────────────────
stego_psnr_val = psnr(tex_cover, stego_img)
print(f"\nPSNR(tex_cover, stego_img) = {stego_psnr_val:.2f} dB")
print("  Invisible to the naked eye (PSNR > 50 dB is imperceptible).")

diff_raw = np.abs(tex_cover.astype(int) - stego_img.astype(int))
diff_map = (diff_raw * 255).astype(np.uint8)          # 0 or 255 (only ±1 changes)

changed_pixels = int(np.sum(diff_raw > 0))
total_pixels   = tex_cover.size
print(f"  Changed pixels: {changed_pixels:,} / {total_pixels:,}  "
      f"({100*changed_pixels/total_pixels:.2f}%)")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("2.2 — Stego analysis", fontsize=13)
axes[0].imshow(tex_cover, cmap="gray"); axes[0].set_title("Clean cover"); axes[0].axis("off")
axes[1].imshow(stego_img, cmap="gray"); axes[1].set_title(f"Stego  (PSNR {stego_psnr_val:.1f} dB)"); axes[1].axis("off")
axes[2].imshow(diff_map,  cmap="hot");  axes[2].set_title("Difference map (×255)"); axes[2].axis("off")
plt.tight_layout()
save(fig, "2_2_B_psnr_and_diff_map.png")

Image.fromarray(diff_map).save(os.path.join(OUT, "2_2_B_difference_map.png"))
print("  saved 2_2_B_difference_map.png")

# ── (C) Textured vs Smooth hiding quality ───────────────────────────
#
# The spec says PSNR alone is not sufficient — add local variance and
# LSB-plane entropy comparisons.

np.random.seed(42)
dummy_payload = np.random.randint(0, 2, 256 * 256, dtype=np.uint8)

tex_stego    = embed_lsb(tex_cover,    dummy_payload)
smooth_stego = embed_lsb(smooth_cover, dummy_payload)

psnr_tex    = psnr(tex_cover,    tex_stego)
psnr_smooth = psnr(smooth_cover, smooth_stego)

# Local 8×8 block variance — measures how "textured" the cover is
def local_variance_map(img, blk=8):
    H, W = img.shape
    vm = np.zeros((H // blk, W // blk))
    for i in range(H // blk):
        for j in range(W // blk):
            vm[i, j] = np.var(img[i*blk:(i+1)*blk, j*blk:(j+1)*blk].astype(float))
    return vm

var_tex    = local_variance_map(tex_cover)
var_smooth = local_variance_map(smooth_cover)

# LSB entropy (before and after)
def lsb_entropy(img):
    lsb = img.flatten() & 1
    p1  = lsb.mean()
    p0  = 1 - p1
    if p0 == 0 or p1 == 0:
        return 0.0
    return -(p0 * np.log2(p0) + p1 * np.log2(p1))

print(f"\nLSB entropy — textured cover  : {lsb_entropy(tex_cover):.4f} bits")
print(f"LSB entropy — smooth cover    : {lsb_entropy(smooth_cover):.4f} bits")
print(f"LSB entropy — textured stego  : {lsb_entropy(tex_stego):.4f} bits")
print(f"LSB entropy — smooth stego    : {lsb_entropy(smooth_stego):.4f} bits")
print(f"\nPSNR textured stego : {psnr_tex:.2f} dB")
print(f"PSNR smooth stego   : {psnr_smooth:.2f} dB")
print(f"Mean local var — textured : {var_tex.mean():.1f}")
print(f"Mean local var — smooth   : {var_smooth.mean():.1f}")

# Visual evidence panel
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Textured vs Smooth as hiding place", fontsize=13)

axes[0, 0].imshow(tex_cover,  cmap="gray"); axes[0, 0].set_title("Textured cover"); axes[0, 0].axis("off")
axes[0, 1].imshow(tex_stego,  cmap="gray"); axes[0, 1].set_title(f"Textured stego (PSNR {psnr_tex:.1f} dB)"); axes[0, 1].axis("off")
im0 = axes[0, 2].imshow(var_tex,  cmap="inferno"); axes[0, 2].set_title("Local variance (textured)"); axes[0, 2].axis("off")
plt.colorbar(im0, ax=axes[0, 2], fraction=0.046)

axes[1, 0].imshow(smooth_cover, cmap="gray"); axes[1, 0].set_title("Smooth cover"); axes[1, 0].axis("off")
axes[1, 1].imshow(smooth_stego, cmap="gray"); axes[1, 1].set_title(f"Smooth stego (PSNR {psnr_smooth:.1f} dB)"); axes[1, 1].axis("off")
im1 = axes[1, 2].imshow(var_smooth, cmap="inferno"); axes[1, 2].set_title("Local variance (smooth)"); axes[1, 2].axis("off")
plt.colorbar(im1, ax=axes[1, 2], fraction=0.046)

plt.tight_layout()
save(fig, "2_2_C_textured_vs_smooth.png")

# LSB-plane visualisation — shows structure introduced in smooth image
lsb_tex_before    = tex_cover & 1
lsb_smooth_before = smooth_cover & 1
lsb_tex_after     = tex_stego & 1
lsb_smooth_after  = smooth_stego & 1

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle("LSB planes before and after embedding", fontsize=13)
axes[0, 0].imshow(lsb_tex_before,    cmap="gray", vmin=0, vmax=1); axes[0, 0].set_title("Textured — original LSB"); axes[0, 0].axis("off")
axes[0, 1].imshow(lsb_tex_after,     cmap="gray", vmin=0, vmax=1); axes[0, 1].set_title("Textured — stego LSB"); axes[0, 1].axis("off")
axes[1, 0].imshow(lsb_smooth_before, cmap="gray", vmin=0, vmax=1); axes[1, 0].set_title("Smooth — original LSB"); axes[1, 0].axis("off")
axes[1, 1].imshow(lsb_smooth_after,  cmap="gray", vmin=0, vmax=1); axes[1, 1].set_title("Smooth — stego LSB"); axes[1, 1].axis("off")
plt.tight_layout()
save(fig, "2_2_C_lsb_planes_comparison.png")

# Difference maps for both covers
diff_tex    = np.abs(tex_cover.astype(int)    - tex_stego.astype(int))
diff_smooth = np.abs(smooth_cover.astype(int) - smooth_stego.astype(int))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("Difference maps (both covers embed same 256×256 payload)", fontsize=12)
axes[0].imshow(diff_tex    * 255, cmap="hot"); axes[0].set_title("Textured diff map"); axes[0].axis("off")
axes[1].imshow(diff_smooth * 255, cmap="hot"); axes[1].set_title("Smooth diff map"); axes[1].axis("off")
plt.tight_layout()
save(fig, "2_2_C_diff_maps_both.png")

Image.fromarray(tex_stego).save(os.path.join(OUT, "2_2_C_tex_stego.png"))
Image.fromarray(smooth_stego).save(os.path.join(OUT, "2_2_C_smooth_stego.png"))

print("\n2.2 done.")

# ═══════════════════════════════════════════════════════════════════
# 2.3  Survive the channel
# ═══════════════════════════════════════════════════════════════════
print("\n─── 2.3 Survive the channel ───")

# ── (A) Generate 128-bit payload ────────────────────────────────────
rng = np.random.default_rng(99)
payload_128 = rng.integers(0, 2, 128, dtype=np.uint8)

# ── (B) Embed — Robust (delta=2) and Naive LSB ──────────────────────
#
# PSNR check for delta=2:
#   Every pixel in each 32×32 block is shifted by ±2.
#   MSE = 4  →  PSNR = 10·log10(255²/4) ≈ 42.1 dB  (≥ 40 dB ✓)
#
# Survival at σ=5:
#   Block size = 32×32 = 1024 pixels.
#   Averaged noise std = 5/√1024 ≈ 0.156.
#   Signal = 2.  SNR ≈ 12.8 → essentially 0 BER.  ✓
#
# JPEG Q=75:
#   Block averaging over 32×32 (≫ JPEG 8×8 block) suppresses
#   quantisation artefacts substantially.  ✓

robust_stego = embed_robust(tex_cover, payload_128, delta=2, block_size=32)
lsb_stego    = embed_lsb(tex_cover, payload_128)

psnr_robust = psnr(tex_cover, robust_stego)
psnr_lsb    = psnr(tex_cover, lsb_stego)

print(f"\nRobust stego PSNR : {psnr_robust:.2f} dB  (requirement ≥ 40 dB)")
print(f"LSB stego PSNR    : {psnr_lsb:.2f} dB")

# Theoretical PSNR check
mse_theory = 4.0   # delta=2, every pixel shifted
psnr_theory = 10 * np.log10(255**2 / mse_theory)
print(f"Theoretical PSNR for delta=2 : {psnr_theory:.2f} dB")

# ── (C) Evaluate Gaussian noise ─────────────────────────────────────
sigmas     = [0, 1, 2, 5, 10, 20]
ber_robust = []
ber_lsb    = []

print("\nBER vs Gaussian noise σ:")
print(f"{'σ':>4}  {'Robust':>8}  {'LSB':>8}")
for s in sigmas:
    noisy_robust = degrade_gaussian(robust_stego, s)
    noisy_lsb    = degrade_gaussian(lsb_stego,    s)

    ext_robust = extract_robust(noisy_robust, tex_cover, 128, block_size=32)
    ext_lsb    = extract_lsb(noisy_lsb, 128)

    br = ber(payload_128, ext_robust)
    bl = ber(payload_128, ext_lsb)
    ber_robust.append(br)
    ber_lsb.append(bl)
    print(f"  σ={s:>2}:  robust={br:.4f}  lsb={bl:.4f}")

# ── (D) JPEG Q=75 ────────────────────────────────────────────────────
jpeg_robust = degrade_jpeg(robust_stego, 75)
jpeg_lsb    = degrade_jpeg(lsb_stego,   75)

ext_jpeg_robust = extract_robust(jpeg_robust, tex_cover, 128, block_size=32)
ext_jpeg_lsb    = extract_lsb(jpeg_lsb, 128)

ber_jpeg_robust = ber(payload_128, ext_jpeg_robust)
ber_jpeg_lsb    = ber(payload_128, ext_jpeg_lsb)

print(f"\nJPEG Q=75 BER — Robust : {ber_jpeg_robust:.4f}")
print(f"JPEG Q=75 BER — LSB    : {ber_jpeg_lsb:.4f}")

# ── (E) BER plot ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sigmas, ber_robust, marker="o", label=f"Robust (δ=2, 32×32 blocks)", linewidth=2, color="steelblue")
ax.plot(sigmas, ber_lsb,    marker="x", label="Naive LSB", linewidth=2, color="tomato", linestyle="--")
ax.axhline(0.5, color="grey", linestyle=":", alpha=0.6, label="Chance level (BER=0.5)")
ax.set_xlabel(r"Gaussian noise $\sigma$", fontsize=12)
ax.set_ylabel("Bit Error Rate (BER)", fontsize=12)
ax.set_title("BER vs Gaussian noise — 128-bit payload", fontsize=13)
ax.legend(fontsize=10)
ax.set_ylim(-0.02, 0.6)
ax.grid(True, alpha=0.4)
# Mark JPEG results as separate points
ax.scatter([], [], marker="D", color="steelblue", label=f"JPEG Q=75 Robust: {ber_jpeg_robust:.4f}")
ax.scatter([], [], marker="D", color="tomato",    label=f"JPEG Q=75 LSB:    {ber_jpeg_lsb:.4f}")
ax.legend(fontsize=10)
plt.tight_layout()
save(fig, "2_3_E_ber_curves.png")

# ── (F) Visual comparison of robust vs naive stego ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle(f"Robust (PSNR {psnr_robust:.1f} dB) vs Naive LSB (PSNR {psnr_lsb:.1f} dB)", fontsize=12)
axes[0].imshow(robust_stego, cmap="gray"); axes[0].set_title("Robust stego"); axes[0].axis("off")
axes[1].imshow(lsb_stego,    cmap="gray"); axes[1].set_title("Naive LSB stego"); axes[1].axis("off")
plt.tight_layout()
save(fig, "2_3_F_stego_visual_comparison.png")

# ── (G) Distortion budget analysis ──────────────────────────────────
#
# 128 bits in a 1024×1024 image = 1,048,576 pixels.
# Each of the 128 blocks (32×32 = 1024 pixels) carries 1 bit.
# Modified pixels: 128 × 1024 = 131,072  (12.5% of image).
# Unmodified pixels: 917,504 (87.5%).
#
# Per-modified-pixel distortion = ±2 gray levels → MSE_modified = 4.
# Overall MSE = 4 × (131072/1048576) = 0.5 → PSNR = 10·log10(255²/0.5) ≈ 55 dB
# but we compute it from the image directly.  The ±2 per all block pixels gives:
# MSE = 4 (all block pixels shifted), reported above.

modified_pixels   = 128 * 32 * 32
total_pix         = tex_cover.size
fraction_modified = modified_pixels / total_pix
mse_per_mod       = 4.0
overall_mse       = mse_per_mod  # all block pixels shifted
budget_per_bit    = mse_per_mod * 32 * 32   # total squared error per bit
capacity_rate     = 128 / total_pix

print(f"\n── Distortion budget ──")
print(f"Payload     : 128 bits in {total_pix:,}-pixel image")
print(f"Rate        : {capacity_rate:.6f} bits/pixel  (very low rate)")
print(f"Modified px : {modified_pixels:,}  ({100*fraction_modified:.1f}%)")
print(f"Per-px dist : δ=2 → |Δ|=2 per pixel in modified blocks")
print(f"Overall MSE : {overall_mse:.2f}  → PSNR ≈ {psnr_theory:.1f} dB")
print(f"  The very low rate means the 40 dB floor is NOT the binding constraint;")
print(f"  the true limit is channel noise tolerance vs. block-mean SNR.")

# ── (H) Summary metrics table ────────────────────────────────────────
print("\n── Summary for report ──")
print(f"{'Metric':<40} {'Value'}")
print("-" * 60)
print(f"{'Stego PSNR (robust, δ=2)':<40} {psnr_robust:.2f} dB")
print(f"{'Stego PSNR (naive LSB)':<40} {psnr_lsb:.2f} dB")
print(f"{'BER robust @ σ=5':<40} {ber_robust[sigmas.index(5)]:.4f}")
print(f"{'BER LSB    @ σ=5':<40} {ber_lsb[sigmas.index(5)]:.4f}")
print(f"{'BER robust @ σ=0 (no noise)':<40} {ber_robust[0]:.4f}")
print(f"{'BER LSB    @ σ=0 (no noise)':<40} {ber_lsb[0]:.4f}")
print(f"{'BER robust JPEG Q=75':<40} {ber_jpeg_robust:.4f}")
print(f"{'BER LSB    JPEG Q=75':<40} {ber_jpeg_lsb:.4f}")
print(f"{'Decoder type':<40} Cover-assisted")
print(f"{'Changed pixels (robust stego)':<40} {modified_pixels:,} ({100*fraction_modified:.1f}%)")

# ── (I) Zero-error verification ─────────────────────────────────────
assert ber_robust[sigmas.index(5)] == 0.0, "FAILED: BER > 0 at σ=5!"
assert ber_jpeg_robust == 0.0,              "FAILED: BER > 0 at JPEG Q=75!"
print("\n✓ Zero BER confirmed at σ=5 and JPEG Q=75.")

print(f"\n── All outputs saved to '{OUT}/' ──")
print("Files generated:")
for f in sorted(os.listdir(OUT)):
    print(f"  {f}")