"""P1 starter. Implement every function marked TODO.

Rules: NumPy array arithmetic only. scipy/cv2/skimage may be used to CHECK
your answers, never to produce them. numpy.fft is allowed in conv2d_fft.
"""
import numpy as np


def kernel_bank(k=15):
    """Provided. Do not modify -- your rank table must match these kernels."""
    ax = np.arange(k) - (k - 1) / 2
    box = np.ones((k, k)) / (k * k)
    s = k / 6.0
    g1 = np.exp(-(ax**2) / (2 * s * s)); g1 /= g1.sum()
    gauss = np.outer(g1, g1)
    sobel = np.outer([1, 2, 1], [-1, 0, 1]).astype(float)
    xx, yy = np.meshgrid(ax, ax); r2 = xx**2 + yy**2
    log = (r2 - 2*s*s) / (s**4) * np.exp(-r2 / (2*s*s))
    motion0 = np.zeros((k, k)); motion0[k // 2, :] = 1.0 / k
    disk = (r2 <= (k/2.0)**2).astype(float); disk /= disk.sum()
    rand = np.random.default_rng(0).normal(size=(k, k)); rand /= np.abs(rand).sum()
    return {"box": box, "gaussian": gauss, "sobel3": sobel, "log": log,
            "log_dc_removed": log - log.mean(), "motion_0deg": motion0,
            "motion_45deg": np.eye(k) / k, "disk": disk, "random": rand}


def numeric_rank(K, tol=1e-10):
    """TODO 1.2: numerical rank from the singular values."""
    s = np.linalg.svd(np.asarray(K, dtype=float), compute_uv=False)
    return int(np.count_nonzero(s > tol))


def conv2d_loops(img, K):
    """TODO 1.1: four nested Python loops. Zero-padded, 'same', TRUE convolution.
    Only run this on small crops -- see the handout."""
    img, K = np.asarray(img), np.asarray(K)
    h, w = img.shape
    kh, kw = K.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, kh - ph - 1), (pw, kw - pw - 1)))
    out = np.zeros_like(img, dtype=np.result_type(img, K, float))
    for i in range(h):
        for j in range(w):
            for u in range(kh):
                for v in range(kw):
                    out[i, j] += padded[i + u, j + v] * K[kh - 1 - u, kw - 1 - v]
    return out


def conv2d_taps(img, K):
    """TODO 1.1: loop over kernel taps, vectorised over the image."""
    img, K = np.asarray(img), np.asarray(K)
    kh, kw = K.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, kh - ph - 1), (pw, kw - pw - 1)))
    out = np.zeros_like(img, dtype=np.result_type(img, K, float))
    for u in range(kh):
        for v in range(kw):
            out += K[kh - 1 - u, kw - 1 - v] * padded[u:u + img.shape[0], v:v + img.shape[1]]
    return out


def conv2d_im2col(img, K):
    """TODO 1.1: build the (H*W, kh*kw) patch matrix, then one matmul.
    numpy.lib.stride_tricks.sliding_window_view is allowed. Report the peak
    memory and be ready to explain which step actually costs it."""
    img, K = np.asarray(img), np.asarray(K)
    kh, kw = K.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, kh - ph - 1), (pw, kw - pw - 1)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))
    patches = windows.reshape(img.size, kh * kw)
    return (patches @ K[::-1, ::-1].reshape(-1)).reshape(img.shape)


def conv2d_fft(img, K):
    """TODO 1.1: multiply in the frequency domain. numpy.fft is allowed."""
    img, K = np.asarray(img), np.asarray(K)
    shape = (img.shape[0] + K.shape[0] - 1, img.shape[1] + K.shape[1] - 1)
    full = np.fft.ifft2(np.fft.fft2(img, shape) * np.fft.fft2(K, shape)).real
    ph, pw = (K.shape[0] - 1) // 2, (K.shape[1] - 1) // 2
    return full[ph:ph + img.shape[0], pw:pw + img.shape[1]]


def conv2d_separable(img, K, tol=1e-10):
    """TODO 1.3: rank-1 only. Raise if K is not rank-1."""
    u, s, vh = np.linalg.svd(np.asarray(K, dtype=float), full_matrices=False)
    if np.count_nonzero(s > tol) != 1:
        raise ValueError("kernel is not rank-1")
    a, b = u[:, 0] * np.sqrt(s[0]), vh[0] * np.sqrt(s[0])
    return conv2d_taps(conv2d_taps(img, a[:, None]), b[None, :])


def conv2d_lowrank(img, K, r):
    """TODO 1.3: sum of r separable passes from the truncated SVD."""
    u, s, vh = np.linalg.svd(np.asarray(K, dtype=float), full_matrices=False)
    out = np.zeros_like(np.asarray(img), dtype=np.result_type(img, K, float))
    for i in range(min(int(r), len(s))):
        a, b = u[:, i] * np.sqrt(s[i]), vh[i] * np.sqrt(s[i])
        out += conv2d_taps(conv2d_taps(img, a[:, None]), b[None, :])
    return out


def psnr(a, b, peak=255.0):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64))**2)
    return float("inf") if mse == 0 else 10 * np.log10(peak * peak / mse)


if __name__ == "__main__":
    from pathlib import Path
    from PIL import Image
    from scipy.signal import convolve2d          # checking only
    root = Path(__file__).resolve().parents[2]
    img = np.asarray(Image.open(root / "images/p1/base_2048.png")).astype(float)[:128, :128]
    K = kernel_bank(7)["gaussian"]
    ref = convolve2d(img, K, mode="same", boundary="fill")
    for fn in (conv2d_loops, conv2d_taps, conv2d_im2col, conv2d_fft):
        try:
            print(f"{fn.__name__:16s} max|err| = {np.abs(fn(img, K) - ref).max():.3e}")
        except NotImplementedError:
            print(f"{fn.__name__:16s} not implemented")
