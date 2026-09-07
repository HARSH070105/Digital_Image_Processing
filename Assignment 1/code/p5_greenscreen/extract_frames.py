"""
Extract 48 evenly-spaced frames from a green-screen clip, and (optionally)
build a wide background plate by stitching a panned background clip/photos
into one image at least 2.5x the frame width.

Usage:
    python extract_frames.py --video mechanic_explosion.mp4 --out_dir frames --n_frames 48

    # If you shot/found a separate panning background video and want a single
    # wide plate stitched from it:
    python extract_frames.py --bg_video bg_pan.mp4 --bg_out background.png --bg_width_factor 2.5
"""
import argparse
import os
import subprocess

import imageio.v3 as iio
import numpy as np


def get_video_info(path):
    meta = iio.immeta(path, plugin="FFMPEG")
    return meta


def extract_n_frames(video_path, out_dir, n_frames=48):
    os.makedirs(out_dir, exist_ok=True)
    reader = iio.imiter(video_path, plugin="FFMPEG")
    frames = [f for f in reader]
    total = len(frames)
    if total == 0:
        raise RuntimeError(f"No frames read from {video_path}")

    idx = np.linspace(0, total - 1, n_frames).round().astype(int)
    idx = np.unique(idx)
    if len(idx) < n_frames:
        # pad by allowing repeats near the end if the clip is very short
        idx = np.linspace(0, total - 1, n_frames).round().astype(int)

    for i, fi in enumerate(idx):
        out_path = os.path.join(out_dir, f"frame_{i:03d}.png")
        iio.imwrite(out_path, frames[fi])

    print(f"Source clip: {total} total frames, {video_path}")
    print(f"Wrote {len(idx)} frames to {out_dir}/ (frame_000.png .. frame_{len(idx)-1:03d}.png)")
    return len(idx)


def build_background_plate(bg_source, out_path, min_width_factor, frame_width):
    """
    bg_source: either a single wide photo (already >= min width) or a video
    of a pan across the background, which gets stitched into one wide image
    by simple horizontal concatenation of evenly sampled frames.

    NOTE: for a clean, artifact-free plate, the best source is a *separate*
    still photo (or panorama) of the empty backing/set with no subject in
    frame, shot wide enough on its own. Stitching video frames is a fallback
    and can show seams; a single wide photo is strongly preferred if you have
    one.
    """
    ext = os.path.splitext(bg_source)[1].lower()
    min_width = int(frame_width * min_width_factor)

    if ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
        img = iio.imread(bg_source)
        if img.shape[1] < min_width:
            print(f"WARNING: background photo width {img.shape[1]}px < required {min_width}px "
                  f"({min_width_factor}x frame width {frame_width}px). Source a wider plate.")
        iio.imwrite(out_path, img)
        print(f"Background plate saved to {out_path} (width={img.shape[1]}px, "
              f"need >= {min_width}px)")
        return

    # otherwise treat as a video: sample a handful of frames and concatenate
    # horizontally (crude stitch -- fine for a static pan with no parallax
    # subject, but check the seams visually afterward)
    frames = [f for f in iio.imiter(bg_source, plugin="FFMPEG")]
    total = len(frames)
    n_strips = 6
    strip_idx = np.linspace(0, total - 1, n_strips).round().astype(int)
    h, w = frames[0].shape[:2]
    strip_w = w // n_strips
    strips = []
    for k, fi in enumerate(strip_idx):
        x0 = k * strip_w
        strips.append(frames[fi][:, x0:x0 + strip_w])
    stitched = np.concatenate(strips, axis=1)
    if stitched.shape[1] < min_width:
        print(f"WARNING: stitched plate width {stitched.shape[1]}px < required {min_width}px. "
              f"Increase n_strips or source a wider clip/photo.")
    iio.imwrite(out_path, stitched)
    print(f"Stitched background plate saved to {out_path} (width={stitched.shape[1]}px, "
          f"need >= {min_width}px)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="path to the green-screen source clip")
    ap.add_argument("--out_dir", default="frames", help="output folder for extracted frames")
    ap.add_argument("--n_frames", type=int, default=48)

    ap.add_argument("--bg_source", help="path to a wide background photo, OR a panning bg video")
    ap.add_argument("--bg_out", default="background.png")
    ap.add_argument("--bg_width_factor", type=float, default=2.5,
                     help="required background width as a multiple of frame width")

    args = ap.parse_args()

    frame_width = None
    if args.video:
        n = extract_n_frames(args.video, args.out_dir, args.n_frames)
        sample = iio.imread(os.path.join(args.out_dir, "frame_000.png"))
        frame_width = sample.shape[1]
        print(f"Frame size: {sample.shape[1]}x{sample.shape[0]}")

    if args.bg_source:
        if frame_width is None:
            # if frames weren't extracted in this run, ask user or default to 1920
            frame_width = 1920
            print("No --video given this run; assuming 1920px frame width for the "
                  "2.5x background-width check. Pass --video too if you want this exact.")
        build_background_plate(args.bg_source, args.bg_out, args.bg_width_factor, frame_width)