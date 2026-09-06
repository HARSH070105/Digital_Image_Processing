from PIL import Image
import sys

input_path = sys.argv[1]
output_path = sys.argv[2]

TARGET_W = 5120
TARGET_H = 1080

img = Image.open(input_path)

w, h = img.size

# Center crop
left = max(0, (w - TARGET_W) // 2)
top = max(0, (h - TARGET_H) // 2)

right = min(w, left + TARGET_W)
bottom = min(h, top + TARGET_H)

img = img.crop((left, top, right, bottom))

# Resize if the cropped image isn't exactly the target size
img = img.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)

img.save(output_path)

print(f"Saved: {output_path}")
print(f"Size: {img.size}")