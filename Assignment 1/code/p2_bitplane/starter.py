"""P2 starter. Implement every function marked TODO."""
import numpy as np


def bit_planes(img):
    """TODO 2.1: (H,W) uint8 -> (8,H,W) uint8 in {0,1}, index 0 = LSB."""
    raise NotImplementedError


def reconstruct(planes, keep):
    """TODO 2.1: rebuild from the `keep` most significant planes."""
    raise NotImplementedError


def gray_encode(img):
    """TODO 2.1: Gray-coded intensities, for the ramp comparison."""
    raise NotImplementedError


def embed_lsb(cover, bits, plane=0):
    """TODO 2.2: write bits into the given plane, raster order."""
    raise NotImplementedError


def extract_lsb(stego, n, plane=0):
    """TODO 2.2: read n bits back out."""
    raise NotImplementedError


def embed_robust(cover, bits, **kw):
    """TODO 2.3: your design. Must carry 128 bits at PSNR(cover,stego) >= 40 dB
    and survive the channels below. Document your parameters."""
    raise NotImplementedError


def extract_robust(stego, cover, n, **kw):
    """TODO 2.3: matching extractor."""
    raise NotImplementedError


# ---- provided channels: do not modify, these are what you are graded against
def degrade_gaussian(img, sigma, rng=None):
    rng = rng or np.random.default_rng(0)
    return np.clip(np.round(img.astype(np.float64) + rng.normal(0, sigma, img.shape)),
                   0, 255).astype(np.uint8)


def degrade_jpeg(img, quality):
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("L"))


def ber(a, b):
    return float(np.mean(a != b))


def psnr(a, b, peak=255.0):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64))**2)
    return float("inf") if mse == 0 else 10 * np.log10(peak * peak / mse)
