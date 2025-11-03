#!/usr/bin/env python3
"""Download a storage object image, trim transparent/white borders, and save locally."""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import requests
from PIL import Image


DEFAULT_BASE_URL = "https://api-storage.arkturian.com"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trim transparent borders from a storage-api image")
    parser.add_argument("--api-key", required=True, help="Storage API key")
    parser.add_argument("--object-id", type=int, required=True, help="Storage object ID")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Storage API base URL")
    parser.add_argument("--output", type=Path, default=Path("trimmed.png"), help="Output file path")
    return parser.parse_args()


def fetch_json(base_url: str, path: str, api_key: str) -> dict:
    resp = requests.get(f"{base_url}{path}", headers={"X-API-KEY": api_key}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_bytes(url: str) -> bytes:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content


def trimmed_bbox(img: Image.Image) -> tuple[int, int, int, int]:
    rgba = img.convert("RGBA")
    alpha = np.array(rgba)[:, :, 3]
    mask = alpha > 0

    if not mask.any():
        # fallback: use brightness to detect non-white
        gray = np.array(rgba.convert("L"))
        mask = gray < 250

    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return 0, 0, img.width, img.height

    x0, y0 = xs.min(), ys.min()
    x1, y1 = xs.max() + 1, ys.max() + 1
    return x0, y0, x1, y1


def main() -> None:
    args = parse_args()
    base = args.base_url.rstrip("/")

    obj = fetch_json(base, f"/storage/objects/{args.object_id}", args.api_key)
    img_url = obj.get("webview_url") or obj.get("file_url") or f"{base}/storage/media/{args.object_id}"
    print(f"Downloading image from {img_url}")
    img = Image.open(io.BytesIO(fetch_bytes(img_url)))

    x0, y0, x1, y1 = trimmed_bbox(img)
    print(f"Detected bbox: x:[{x0},{x1}) y:[{y0},{y1}) (width {img.width}, height {img.height})")

    trimmed = img.crop((x0, y0, x1, y1))
    trimmed.save(args.output)
    print(f"Saved trimmed image to {args.output.resolve()}")


if __name__ == "__main__":
    main()


