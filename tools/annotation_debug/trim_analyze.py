#!/usr/bin/env python3
"""Trim a storage image, upload it temporarily, run AI analysis, then show annotations."""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image


DEFAULT_BASE_URL = "https://api-storage.arkturian.com"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trim storage object image and run fresh AI analysis")
    parser.add_argument("--api-key", required=True, help="Storage API key")
    parser.add_argument("--object-id", type=int, required=True, help="Storage object ID to re-process")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Storage API base URL")
    parser.add_argument("--mode", default="quality", help="Analyze mode (fast|quality)")
    parser.add_argument("--vision-mode", default="product", help="Vision mode for AI")
    parser.add_argument(
        "--tasks", default="vision,embedding,kg", help="Tasks to run (comma separated: vision,embedding,kg,...)"
    )
    parser.add_argument("--context-role", default="product", help="Context role hint for AI")
    parser.add_argument("--metadata-json", default=None, help="Extra JSON metadata for the AI prompt")
    parser.add_argument("--save", type=Path, default=None, help="Optional path to save trimmed PNG (defaults to trimmed_<object_id>.png)")
    return parser.parse_args()


def auth_headers(api_key: str) -> dict:
    return {"X-API-KEY": api_key}


def fetch_json(base: str, path: str, api_key: str) -> dict:
    r = requests.get(f"{base}{path}", headers=auth_headers(api_key), timeout=40)
    r.raise_for_status()
    return r.json()


def fetch_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=40)
    r.raise_for_status()
    return r.content


def trimmed_image(img: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
    rgba = img.convert("RGBA")
    alpha = np.array(rgba)[:, :, 3]
    mask = alpha > 0
    if not mask.any():
        gray = np.array(rgba.convert("L"))
        mask = gray < 250

    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return img.copy(), (0, 0, img.width, img.height)

    x0, y0 = xs.min(), ys.min()
    x1, y1 = xs.max() + 1, ys.max() + 1
    return rgba.crop((x0, y0, x1, y1)), (x0, y0, x1, y1)


def upload_temp(base: str, api_key: str, image: Image.Image, origin_meta: dict) -> int:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    files = {"file": (origin_meta.get("original_filename", "trimmed.png"), buf, "image/png")}
    data = {
        "analyze": "false",
        "is_public": "false",
        "context": origin_meta.get("context", "trimmed")
    }
    r = requests.post(
        f"{base}/storage/upload",
        headers=auth_headers(api_key),
        files=files,
        data=data,
        timeout=60,
    )
    r.raise_for_status()
    payload = r.json()
    return payload["id"]


def analyze_object(
    base: str,
    api_key: str,
    object_id: int,
    *,
    mode: str,
    tasks: str,
    vision_mode: str,
    context_role: str,
    metadata_json: Optional[str],
) -> str:
    params = {
        "mode": mode,
        "ai_tasks": tasks,
        "ai_vision_mode": vision_mode,
        "ai_context_role": context_role,
    }
    if metadata_json:
        params["ai_metadata"] = metadata_json

    r = requests.post(
        f"{base}/storage/analyze-async/{object_id}",
        headers=auth_headers(api_key),
        params=params,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["task_id"]


def poll_task(base: str, api_key: str, task_id: str) -> dict:
    while True:
        info = fetch_json(base, f"/storage/tasks/{task_id}", api_key)
        status = info.get("status")
        print(f"Task {task_id}: status={status} progress={info.get('progress')}% phase={info.get('current_phase')}")
        if status in {"completed", "failed"}:
            return info
        time.sleep(3)


def fetch_annotations(base: str, api_key: str, object_id: int) -> list[dict]:
    data = fetch_json(base, f"/storage/objects/{object_id}/annotations", api_key)
    return data.get("annotations", [])


def main() -> None:
    args = parse_args()
    base = args.base_url.rstrip("/")

    origin = fetch_json(base, f"/storage/objects/{args.object_id}", args.api_key)
    origin_url = origin.get("webview_url") or origin.get("file_url") or f"{base}/storage/media/{args.object_id}"
    print(f"Downloading original image from {origin_url}")
    img = Image.open(io.BytesIO(fetch_bytes(origin_url)))

    trimmed, bbox = trimmed_image(img)
    print(f"Trimmed visible bounds: x:[{bbox[0]},{bbox[2]}) y:[{bbox[1]},{bbox[3]})")
    if args.save:
        save_path = args.save
    else:
        save_path = Path(f"trimmed_{args.object_id}.png")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    trimmed.save(save_path)
    print(f"Saved trimmed PNG to {save_path.resolve()}")

    print("Uploading trimmed image as temporary object…")
    temp_id = upload_temp(base, args.api_key, trimmed, origin)
    print(f"Temporary object ID: {temp_id}")

    x0, y0, x1, y1 = [int(v) for v in bbox]
    metadata = {
        "original_object_id": int(args.object_id),
        "trim_bbox": {
            "x1": x0,
            "y1": y0,
            "x2": x1,
            "y2": y1,
            "width": int(img.width),
            "height": int(img.height),
        },
    }
    if args.metadata_json:
        try:
            metadata.update(json.loads(args.metadata_json))
        except json.JSONDecodeError:
            print("Warning: metadata-json is not valid JSON; ignoring")

    task_id = analyze_object(
        base,
        args.api_key,
        temp_id,
        mode=args.mode,
        tasks=args.tasks,
        vision_mode=args.vision_mode,
        context_role=args.context_role,
        metadata_json=json.dumps(metadata),
    )
    print(f"Triggered analysis task {task_id}")
    info = poll_task(base, args.api_key, task_id)
    if info.get("status") != "completed":
        print(f"Task failed: {info.get('error')}")
        return

    annotations = fetch_annotations(base, args.api_key, temp_id)
    print(json.dumps(annotations, indent=2))
    print(f"Finished. Temporary object {temp_id} now holds the annotation result.")


if __name__ == "__main__":
    main()


