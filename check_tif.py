#!/usr/bin/env python3
"""Check GeoTIFF properties."""
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Allow large images (CTX images are huge!)
Image.MAX_IMAGE_PIXELS = None

tif_path = Path("data/test/ctx_single/MOI_000009_0186_XI_18S051W.tif")

img = Image.open(tif_path)
# Don't load entire array into memory, just get info
# arr = np.array(img)

print(f"File: {tif_path.name}")
print(f"Size: {img.size[0]} x {img.size[1]} pixels")
print(f"Mode: {img.mode}")
print(f"Format: {img.format}")

# Sample a small region to check values
sample = np.array(img.crop((0, 0, 1000, 1000)))
print(f"Sample shape: {sample.shape}")
print(f"Data type: {sample.dtype}")
print(f"Sample value range: [{sample.min()}, {sample.max()}]")
print(f"File size: {tif_path.stat().st_size / (1024**2):.1f} MB")
print(f"\nFile created successfully! ✓")
print("\nNote: This image is GDAL-converted (no geometric correction).")
print("For proper map projection, install ISIS3 (see ISIS3_SETUP.md)")
