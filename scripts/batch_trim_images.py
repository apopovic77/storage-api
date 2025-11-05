#!/usr/bin/env python3
"""Batch-trigger trimmed derivatives for storage images.

This script walks over image objects in storage-api and invokes the async
analysis pipeline with trimming enabled. By default it skips objects that
already contain `trim_bounds` metadata unless --force is provided.

Example:
    python scripts/batch_trim_images.py \
        --api-key $STORAGE_API_KEY \
        --base-url https://api-storage.arkturian.com \
        --tenant oneal

Use --object-id to process a single storage object.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

import requests


DEFAULT_BASE_URL = "https://api-storage.arkturian.com"
POLL_INTERVAL_SECONDS = 5


@dataclass
class StorageObject:
    id: int
    original_filename: str
    mime_type: str
    ai_context_metadata: dict

    @property
    def has_trim(self) -> bool:
        try:
            meta = self.ai_context_metadata or {}
            return bool(meta.get("trim_bounds"))
        except AttributeError:
            return False


def fetch_objects(base_url: str, api_key: str, tenant: Optional[str], limit: int) -> List[StorageObject]:
    params = {
        "mine": "false",
        "limit": str(limit),
    }

    url = f"{base_url.rstrip('/')}/storage/list"
    headers = {"X-API-KEY": api_key}

    resp = requests.get(url, headers=headers, params=params, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to list storage objects: {resp.status_code} {resp.text}")

    data = resp.json()
    items = data.get("items", [])

    results: List[StorageObject] = []
    for item in items:
        if tenant and item.get("tenant_id") != tenant:
            continue
        mime = (item.get("mime_type") or "").lower()
        if not mime.startswith("image/"):
            continue
        results.append(
            StorageObject(
                id=item["id"],
                original_filename=item.get("original_filename", ""),
                mime_type=mime,
                ai_context_metadata=item.get("ai_context_metadata") or {},
            )
        )

    return results


def fetch_single_object(base_url: str, api_key: str, object_id: int) -> Optional[StorageObject]:
    url = f"{base_url.rstrip('/')}/storage/objects/{object_id}"
    headers = {"X-API-KEY": api_key}

    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code == 404:
        return None
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch object {object_id}: {resp.status_code} {resp.text}")

    item = resp.json()
    mime = (item.get("mime_type") or "").lower()
    if not mime.startswith("image/"):
        print(f"Object {object_id} is not an image (mime={mime}), skipping")
        return None

    return StorageObject(
        id=item["id"],
        original_filename=item.get("original_filename", ""),
        mime_type=mime,
        ai_context_metadata=item.get("ai_context_metadata") or {},
    )


def trigger_trim(
    base_url: str,
    api_key: str,
    object_id: int,
    *,
    tasks: Optional[str],
    vision_mode: Optional[str],
    context_role: Optional[str],
    trim_delivery_default: bool,
) -> str:
    params = {
        "mode": "quality",
        "trim_before_analysis": "true",
    }
    if trim_delivery_default:
        params["trim_delivery_default"] = "true"
    if tasks:
        params["ai_tasks"] = tasks
    if vision_mode:
        params["ai_vision_mode"] = vision_mode
    if context_role:
        params["ai_context_role"] = context_role

    url = f"{base_url.rstrip('/')}/storage/analyze-async/{object_id}"
    headers = {"X-API-KEY": api_key}

    resp = requests.post(url, headers=headers, params=params, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Failed to trigger trim for object {object_id}: {resp.status_code} {resp.text}"
        )

    payload = resp.json()
    return payload["task_id"]


def poll_task(base_url: str, api_key: str, task_id: str) -> dict:
    url = f"{base_url.rstrip('/')}/storage/tasks/{task_id}"
    headers = {"X-API-KEY": api_key}

    resp = requests.get(url, headers=headers, timeout=20)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to get task {task_id}: {resp.status_code} {resp.text}")
    return resp.json()


def ensure_trimmed(
    base_url: str,
    api_key: str,
    objects: Iterable[StorageObject],
    *,
    force: bool,
    poll: bool,
    tasks: Optional[str],
    vision_mode: Optional[str],
    context_role: Optional[str],
    trim_delivery_default: bool,
) -> None:
    for index, obj in enumerate(objects, start=1):
        if not force and obj.has_trim:
            print(f"[{index}] Object {obj.id} already trimmed – skipping")
            continue

        print(f"[{index}] Triggering trim for object {obj.id} ({obj.original_filename})")
        try:
            task_id = trigger_trim(
                base_url,
                api_key,
                object_id=obj.id,
                tasks=tasks,
                vision_mode=vision_mode,
                context_role=context_role,
                trim_delivery_default=trim_delivery_default,
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"    ❌ Failed to start trim: {exc}")
            continue

        if not poll:
            print(f"    ✅ Task queued: {task_id}")
            continue

        print(f"    … waiting for task {task_id}")
        while True:
            try:
                info = poll_task(base_url, api_key, task_id)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"    ⚠️ Poll error: {exc}")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            status = info.get("status")
            progress = info.get("progress")
            phase = info.get("current_phase")
            print(f"    status={status} progress={progress}% phase={phase}", end="\r")

            if status in {"completed", "failed"}:
                print()
                if status == "completed":
                    print(f"    ✅ Trim completed for task {task_id}")
                else:
                    print(f"    ❌ Trim failed: {info.get('error')}")
                break

            time.sleep(POLL_INTERVAL_SECONDS)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch trim storage images via analyze-async")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Storage API base URL")
    parser.add_argument("--api-key", required=True, help="Storage API key")
    parser.add_argument("--tenant", default=None, help="Tenant filter (optional)")
    parser.add_argument("--object-id", type=int, default=None, help="Process only this storage object ID")
    parser.add_argument("--limit", type=int, default=5000, help="Max objects to fetch (default 5000)")
    parser.add_argument("--force", action="store_true", help="Re-trim even if trim_bounds already exist")
    parser.add_argument("--no-poll", action="store_true", help="Do not wait for task completion")
    parser.add_argument("--tasks", default="vision", help="AI tasks to run (default: vision)")
    parser.add_argument("--vision-mode", default="product", help="Vision mode for analysis (default: product)")
    parser.add_argument("--context-role", default="product", help="Context role (default: product)")
    parser.add_argument("--set-trim-default", action="store_true", help="Set trim_delivery_default=true when processing")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    try:
        if args.object_id is not None:
            obj = fetch_single_object(args.base_url, args.api_key, args.object_id)
            if obj is None:
                print(f"Object {args.object_id} not found or not an image.")
                return 1
            objects = [obj]
        else:
            objects = fetch_objects(args.base_url, args.api_key, args.tenant, args.limit)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Failed to fetch objects: {exc}")
        return 1

    if not objects:
        print("No matching image objects found.")
        return 0

    if args.object_id is not None:
        print(f"Processing single object {args.object_id}…")
    else:
        print(f"Found {len(objects)} image objects. Starting trim batch…")

    ensure_trimmed(
        args.base_url,
        args.api_key,
        objects,
        force=args.force,
        poll=not args.no_poll,
        tasks=args.tasks,
        vision_mode=args.vision_mode,
        context_role=args.context_role,
        trim_delivery_default=args.set_trim_default,
    )

    print("Batch trim run finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

