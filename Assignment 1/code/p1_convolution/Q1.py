"""P1 starter. Implement every function marked TODO.

Rules: NumPy array arithmetic only. scipy/cv2/skimage may be used to CHECK
your answers, never to produce them. numpy.fft is allowed in conv2d_fft.
"""

########### I have modified the starter file to give the outputs

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

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
    singular_values = np.linalg.svd(K,compute_uv=False)
    return np.sum(singular_values > tol)


def conv2d_loops(img, K):
    img = np.asarray(img, dtype=float)
    K = np.asarray(K, dtype=float)

    H, W = img.shape
    kh, kw = K.shape

    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    out = np.zeros((H, W), dtype=float)
    K_flip = K[::-1, ::-1]

    for i in range(H):
        for j in range(W):
            value = 0.0
            for u in range(kh):
                for v in range(kw):
                    value += (padded[i + u, j + v] * K_flip[u, v])
            out[i, j] = value
            
    return out


def conv2d_taps(img, K):
    img = np.asarray(img, dtype=float)
    K = np.asarray(K, dtype=float)

    H, W = img.shape
    kh, kw = K.shape

    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(img ,((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    out = np.zeros((H, W), dtype=float)

    K_flip = K[::-1, ::-1]

    for u in range(kh):
        for v in range(kw):
            out += (K_flip[u, v] * padded[u:u + H, v:v + W])
    return out


def conv2d_im2col(img, K):
    img = np.asarray(img, dtype=float)
    K = np.asarray(K, dtype=float)

    H, W = img.shape
    kh, kw = K.shape

    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)),mode="constant")

    windows = np.lib.stride_tricks.sliding_window_view(padded,(kh, kw))
    patches = windows.reshape(H * W, kh * kw)
    K_flip = K[::-1, ::-1]
    kernel_vector = K_flip.reshape(-1)
    out = patches @ kernel_vector
    return out.reshape(H, W)


def conv2d_fft(img, K):
    img = np.asarray(img, dtype=float)
    K = np.asarray(K, dtype=float)

    H, W = img.shape
    kh, kw = K.shape

    # Size needed for FULL linear convolution
    fft_shape = (H + kh - 1,W + kw - 1)
    IMG = np.fft.fft2(img,fft_shape)
    KERNEL = np.fft.fft2(K,fft_shape)

    full = np.fft.ifft2(IMG * KERNEL).real

    # Extract SAME portion
    start_h = kh // 2
    start_w = kw // 2

    out = full[start_h:start_h + H, start_w:start_w + W]
    return out


def conv2d_separable(img, K, tol=1e-10):
    K = np.asarray(K, dtype=float)

    U, S, Vh = np.linalg.svd(K,full_matrices=False)

    rank = np.sum(S > tol)

    if rank != 1:
        raise ValueError("Kernel is not rank-1")

    # K ≈ sigma * u * v^T
    #
    # Split sqrt(sigma) between the two vectors
    vertical = (U[:, 0] * np.sqrt(S[0]))

    horizontal = (Vh[0, :] * np.sqrt(S[0]))
    # First vertical convolution
    temp = conv2d_taps(img,vertical[:, None])

    # Then horizontal convolution
    out = conv2d_taps(temp,horizontal[None, :])
    return out



def conv2d_lowrank(img, K, r):
    img = np.asarray(img, dtype=float)
    K = np.asarray(K, dtype=float)

    U, S, Vh = np.linalg.svd(K,full_matrices=False)

    r = min(r, len(S))

    out = np.zeros_like(img,dtype=float)

    for i in range(r):
        vertical = (U[:, i] * np.sqrt(S[i]))
        horizontal = (Vh[i, :] * np.sqrt(S[i]))
        temp = conv2d_taps(img,vertical[:, None])
        out += conv2d_taps(temp,horizontal[None, :])
        
    return out


def psnr(a, b, peak=255.0):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64))**2)
    return float("inf") if mse == 0 else 10 * np.log10(peak * peak / mse)

if __name__ == "__main__":
    from pathlib import Path
    from PIL import Image
    from scipy.signal import convolve2d          # checking only
    from scipy.signal import correlate2d

    root = Path(__file__).resolve().parents[2]
    
    img = np.asarray(Image.open(root / "images/p1/base_2048.png")).astype(float)[:128, :128]
    K = kernel_bank(7)["gaussian"]
    ref = convolve2d(img, K, mode="same", boundary="fill")
    for fn in (conv2d_loops, conv2d_taps, conv2d_im2col, conv2d_fft):
        try:
            print(f"{fn.__name__:16s} max|err| = {np.abs(fn(img, K) - ref).max():.3e}")
        except NotImplementedError:
            print(f"{fn.__name__:16s} not implemented")
    
    ###########################################
    ######## Assymetric kernel test ###########
    ###########################################
    K_asym = np.array([
        [1,  2,  3],
        [4,  5,  6],
        [7,  8, 10]
    ], dtype=float)
    out_conv = conv2d_taps(img, K_asym)

    ref_conv = convolve2d(
        img,
        K_asym,
        mode="same",
        boundary="fill"
    )
    
    ref_corr = correlate2d(
        img,
        K_asym,
        mode="same",
        boundary="fill"
    )
    
    print("\nAsymmetric kernel test:")
    
    print(
        "Our convolution vs scipy convolution:",
        np.abs(out_conv - ref_conv).max()
    )
    
    print(
        "Our convolution vs scipy correlation:",
        np.abs(out_conv - ref_corr).max()
    )
    
    ###################################
    ############### TIMING ############
    ###################################
    
    img128 = np.asarray(Image.open(root / "images/p1/base_2048.png")).astype(float)[:128, :128]

    K7 = kernel_bank(7)["gaussian"]

    # Warm-up
    conv2d_loops(img128, K7)

    runs = 10
    times = []

    for _ in range(runs):
        start = time.perf_counter()
        conv2d_loops(img128, K7)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = np.mean(times)

    print("\nTiming:")
    print(f"128x128, k=7 average time = {avg_time:.4f} seconds")
    
    #### Scaling
    
    scale = ((2048 * 2048 * 15 * 15)/(128 * 128 * 7 * 7))
    estimated_time = (avg_time * scale)
    print(f"Scaling factor: {scale:.2f}x")
    print(f"Estimated 2048x2048, k=15: "f"{estimated_time:.2f} sec")
    print(f"Estimated minutes: "f"{estimated_time / 60:.2f} min")
    
    
    import matplotlib.pyplot as plt

############################################################
####################### KERNAL RANKS #######################
############################################################
    k = 15
    kernels = kernel_bank(k)
    tol = 1e-10

    print("\n========== KERNEL RANKS ==========")
    print(f"Tolerance = {tol}\n")

    ranks = {}

    for name, K in kernels.items():
        S = np.linalg.svd(K,compute_uv=False)
        rank = numeric_rank(K,tol)
        ranks[name] = rank
        print(f"{name:20s} " f"rank = {rank:2d}")

    fig, axes = plt.subplots(3,3,figsize=(14, 11))
    axes = axes.flatten()
    for ax, (name, K) in zip(axes, kernels.items()):
        S = np.linalg.svd(K,compute_uv=False)
        S_normalized = S / S[0]
        rank = numeric_rank(K,tol)
        x = np.arange(1,len(S) + 1)
        ax.semilogy(x,S_normalized,marker="o",linewidth=2,markersize=5)

        ax.axhline(tol,linestyle="--",linewidth=1)
        ax.set_title(f"{name}\nRank = {rank}",fontsize=12)
        ax.set_xlabel("Singular value index")

        ax.set_ylabel(r"$\sigma_i / \sigma_1$")
        ax.set_xticks(np.arange(1, len(S) + 1))

        ax.grid(True,which="both",alpha=0.3)

    for ax in axes[len(kernels):]:
        ax.axis("off")

    fig.suptitle("Normalized Singular Value Spectra (k = 15)",fontsize=18,y=0.98)

    plt.tight_layout()

    plt.savefig("p1_2_singular_spectra_grid.png",dpi=250,bbox_inches="tight")

    plt.show()
    
    
# =====================================================
#  LOW-RANK APPROXIMATION: PSNR vs r
# =====================================================


    print("\n========== PSNR vs R ==========")

    # Load top-left 512 x 512 crop
    img512 = np.asarray(
        Image.open(
            root / "images/p1/base_2048.png"
        )
    ).astype(float)[:512, :512]


    # Kernel size
    k = 21

    kernels21 = kernel_bank(k)

    kernel_names = [
        "disk",
        "log",
        "motion_45deg",
        "random"
    ]

    results = {}


    for name in kernel_names:

        print(f"\nProcessing: {name}")

        K = kernels21[name]

        # Exact convolution reference
        exact = convolve2d(
            img512,
            K,
            mode="same",
            boundary="fill"
        )

        # Maximum possible rank
        max_rank = min(K.shape)

        psnr_values = []

        for r in range(1, max_rank + 1):

            approx = conv2d_lowrank(
                img512,
                K,
                r
            )

            value = psnr(
                exact,
                approx
            )

            psnr_values.append(value)

            print(
                f"r = {r:2d}, "
                f"PSNR = {value:.3f} dB"
            )

        results[name] = psnr_values


    # =====================================================
    # PLOT PSNR vs R
    # =====================================================

    plt.figure(figsize=(9, 6))

    for name in kernel_names:

        r_values = np.arange(
            1,
            len(results[name]) + 1
        )

        plt.plot(
            r_values,
            results[name],
            marker="o",
            linewidth=2,
            label=name
        )

    plt.xlabel("Number of SVD terms (r)")
    plt.ylabel("PSNR (dB)")

    plt.title(
        "Low-Rank Convolution Approximation: PSNR vs r\n"
        "k = 21, Image = 512 × 512"
    )

    plt.grid(True, alpha=0.3)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "p1_3_psnr_vs_r.png",
        dpi=250,
        bbox_inches="tight"
    )

    plt.show()
    
    # =====================================================
    # TEST SEPARABLE CONVOLUTION
    # =====================================================

    K_gaussian = kernels21["gaussian"]

    ref_gaussian = convolve2d(
        img512,
        K_gaussian,
        mode="same",
        boundary="fill"
    )

    our_gaussian = conv2d_separable(
        img512,
        K_gaussian
    )

    print("\n========== SEPARABLE TEST ==========")

    print(
        "Gaussian max error:",
        np.abs(
            ref_gaussian - our_gaussian
        ).max()
    )
    
    # ============================================================
    # P1.4 RUNTIME ANALYSIS
    # ============================================================

    def benchmark(fn, img, K, repeats=3):
        """
        Return average runtime in seconds.
        """
        # Warm-up
        fn(img, K)

        times = []

        for _ in range(repeats):
            start = time.perf_counter()
            fn(img, K)
            end = time.perf_counter()

            times.append(end - start)

        return np.mean(times)

    # ============================================================
    # LOAD IMAGE
    # ============================================================

    full_img = np.asarray(
        Image.open(
            root / "images/p1/base_2048.png"
        )
    ).astype(float)


    # ============================================================
    # EXPERIMENT 1:
    # Runtime vs kernel size k at 512 x 512
    # ============================================================

    print("\n======================================")
    print("P1.4(a): Runtime vs kernel size")
    print("======================================")

    img512 = full_img[:512, :512]

    k_values = [3, 7, 11, 15, 21, 31]

    times_k = []

    for k in k_values:

        K = kernel_bank(k)["random"]

        # Fewer repeats for expensive kernels
        repeats = 3 if k <= 15 else 1

        t = benchmark(
            conv2d_taps,
            img512,
            K,
            repeats=repeats
        )

        times_k.append(t)

        print(
            f"k = {k:2d}, "
            f"time = {t:.6f} sec"
        )

    # Fit log(runtime) = slope * log(k) + intercept

    slope_k, intercept_k = np.polyfit(
        np.log(k_values),
        np.log(times_k),
        1
    )

    print(
        f"\nFitted slope vs k = {slope_k:.4f}"
    )


    # Plot

    plt.figure(figsize=(8, 6))

    plt.loglog(
        k_values,
        times_k,
        "o-",
        label="Measured"
    )

    fit_k = np.exp(intercept_k) * np.array(k_values) ** slope_k

    plt.loglog(
        k_values,
        fit_k,
        "--",
        label=f"Fit: slope = {slope_k:.3f}"
    )

    plt.xlabel("Kernel size k")
    plt.ylabel("Runtime (seconds)")

    plt.title(
        "Tap-loop Runtime vs Kernel Size\n"
        "Image = 512 × 512"
    )

    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "p1_4_runtime_vs_k.png",
        dpi=250,
        bbox_inches="tight"
    )

    plt.show()


    # ============================================================
    # EXPERIMENT 2:
    # Runtime vs image size N at k = 15
    # ============================================================

    print("\n======================================")
    print("P1.4(b): Runtime vs image size")
    print("======================================")

    N_values = [128, 256, 512, 1024, 2048]

    k = 15
    K15 = kernel_bank(k)["random"]

    times_N = []

    for N in N_values:

        imgN = full_img[:N, :N]

        # Large images are extremely expensive for Python loops
        if N <= 512:
            repeats = 3
        else:
            repeats = 1

        t = benchmark(
            conv2d_taps,
            imgN,
            K15,
            repeats=repeats
        )

        times_N.append(t)

        print(
            f"N = {N:4d}, "
            f"time = {t:.6f} sec"
        )

    # Fit log(runtime) against log(N)

    slope_N, intercept_N = np.polyfit(
        np.log(N_values),
        np.log(times_N),
        1
    )

    print(
        f"\nFitted slope vs N = {slope_N:.4f}"
    )

    # Plot

    plt.figure(figsize=(8, 6))

    plt.loglog(
        N_values,
        times_N,
        "o-",
        label="Measured"
    )

    fit_N = np.exp(intercept_N) * np.array(N_values) ** slope_N

    plt.loglog(
        N_values,
        fit_N,
        "--",
        label=f"Fit: slope = {slope_N:.3f}"
    )

    plt.xlabel("Image side length N")
    plt.ylabel("Runtime (seconds)")

    plt.title(
        "Tap-loop Runtime vs Image Size\n"
        "Kernel = 15 × 15"
    )

    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "p1_4_runtime_vs_N.png",
        dpi=250,
        bbox_inches="tight"
    )

    plt.show()

    # ============================================================
    # EXPERIMENT 3:
    # im2col memory
    # ============================================================

    print("\n======================================")
    print("P1.4(c): im2col patch memory")
    print("======================================")

    dtype_bytes = np.dtype(np.float64).itemsize

    print(
        f"{'N':>8} {'k':>8} "
        f"{'patch elements':>18} "
        f"{'memory (MB)':>15} "
        f"{'memory (GiB)':>15}"
    )

    memory_values = []

    for N in N_values:

        elements = N * N * k * k

        bytes_used = elements * dtype_bytes

        MB = bytes_used / 1e6
        GiB = bytes_used / (1024 ** 3)

        memory_values.append(bytes_used)

        print(
            f"{N:8d} "
            f"{k:8d} "
            f"{elements:18,d} "
            f"{MB:15.2f} "
            f"{GiB:15.3f}"
        )

    # ===========================================================
    # EXPERIMENT 4:
    # im2col vs tap loop
    # ============================================================

    print("\n======================================")
    print("P1.4(d): im2col vs tap loop")
    print("======================================")

    im2col_times = []
    tap_times = []

    for N in [128, 256, 512, 1024, 2048]:

        imgN = full_img[:N, :N]

        t_tap = benchmark(
            conv2d_taps,
            imgN,
            K15,
            repeats=3
        )

        try:

            t_im2col = benchmark(
                conv2d_im2col,
                imgN,
                K15,
                repeats=3
            )

        except MemoryError:

            t_im2col = np.nan

        tap_times.append(t_tap)
        im2col_times.append(t_im2col)

        print(
            f"N={N:4d}: "
            f"tap={t_tap:.6f}s, "
            f"im2col={t_im2col:.6f}s"
        )

    # ============================================================
    # EXPERIMENT 5:
    # Separable speedup
    # ============================================================

    print("\n======================================")
    print("P1.4(e): Separable speedup")
    print("======================================")

    K_gaussian15 = kernel_bank(15)["gaussian"]

    for N in [512, 2048]:

        imgN = full_img[:N, :N]

        # Direct k x k convolution
        repeats = 3 if N == 512 else 1

        t_direct = benchmark(
            conv2d_taps,
            imgN,
            K_gaussian15,
            repeats=repeats
        )

        # Two 1-D convolutions
        t_sep = benchmark(
            conv2d_separable,
            imgN,
            K_gaussian15,
            repeats=repeats
        )

        speedup = t_direct / t_sep

        print(
            f"N={N:4d}: "
            f"direct={t_direct:.6f}s, "
            f"separable={t_sep:.6f}s, "
            f"speedup={speedup:.3f}x"
        )