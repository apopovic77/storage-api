#!/usr/bin/env python3
"""Trigger AI annotation pipeline for all product images of a tenant.

Usage examples:
  python scripts/batch_product_annotations.py --tenant oneal \
      --api-key $STORAGE_API_KEY --base-url https://api-storage.arkturian.com

By default the script skips objects that already contain annotations. Use
  --force
to re-run analysis regardless of existing results.
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
    ai_category: Optional[str]
    context: Optional[str]
    metadata_json: Optional[dict]

    @property
    def has_annotations(self) -> bool:
        try:
            annotations = (
                self.ai_context_metadata
                .get("embedding_info", {})
                .get("metadata", {})
                .get("annotations", [])
            )
            return bool(annotations)
        except AttributeError:
            return False

    @property
    def has_embeddings(self) -> bool:
        try:
            embedding = (
                self.ai_context_metadata
                .get("embedding_info", {})
                .get("embedding")
            )
            return isinstance(embedding, list) and len(embedding) > 0
        except AttributeError:
            return False

        try:
            annotations = (
                self.ai_context_metadata
                .get("embedding_info", {})
                .get("metadata", {})
                .get("annotations", [])
            )
            return bool(annotations)
        except AttributeError:
            return False


def fetch_objects(
    base_url: str,
    api_key: str,
    tenant: Optional[str],
    limit: int,
    context_contains: Optional[str],
) -> List[StorageObject]:
    """Fetch storage objects for the given tenant."""
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
        # Filter tenant if provided (endpoint already scopes by API key tenant)
        if tenant and item.get("tenant_id") != tenant:
            continue
        mime = (item.get("mime_type") or "").lower()
        if not mime.startswith("image/"):
            continue
        context = item.get("context")
        if context_contains:
            if not context or context_contains.lower() not in context.lower():
                continue
        results.append(
            StorageObject(
                id=item["id"],
                original_filename=item.get("original_filename", ""),
                mime_type=mime,
                ai_context_metadata=item.get("ai_context_metadata") or {},
                ai_category=item.get("ai_category"),
                context=context,
                metadata_json=item.get("metadata_json") or {},
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
        ai_category=item.get("ai_category"),
        context=item.get("context"),
        metadata_json=item.get("metadata_json") or {},
    )


def trigger_analysis(
    base_url: str,
    api_key: str,
    object_id: int,
    vision_mode: str,
    tasks: str,
    context_role: str,
    metadata: Optional[str],
    trim_before_analysis: bool,
    trim_delivery_default: bool,
) -> str:
    params = {
        "mode": "quality",
        "ai_tasks": tasks,
        "ai_vision_mode": vision_mode,
        "ai_context_role": context_role,
    }
    if metadata:
        params["ai_metadata"] = metadata
    if trim_before_analysis:
        params["trim_before_analysis"] = "true"
    if trim_delivery_default:
        params["trim_delivery_default"] = "true"

    url = f"{base_url.rstrip('/')}/storage/analyze-async/{object_id}"
    headers = {"X-API-KEY": api_key}

    resp = requests.post(url, headers=headers, params=params, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Failed to trigger analysis for object {object_id}: {resp.status_code} {resp.text}"
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


def ensure_annotations(
    base_url: str,
    api_key: str,
    objects: Iterable[StorageObject],
    *,
    force: bool,
    poll: bool,
    vision_mode: str,
    tasks: str,
    context_role: str,
    metadata: Optional[str],
    trim_before_analysis: bool,
    trim_delivery_default: bool,
    skip_if_embedding: bool,
) -> None:
    for index, obj in enumerate(objects, start=1):
        if not force and obj.has_annotations:
            print(f"[{index}] Object {obj.id} already has annotations – skipping")
            continue
        if skip_if_embedding and obj.has_embeddings:
            print(f"[{index}] Object {obj.id} already has embeddings – skipping")
            continue

        print(f"[{index}] Triggering analysis for object {obj.id} ({obj.original_filename})")
        try:
            task_id = trigger_analysis(
                base_url,
                api_key,
                object_id=obj.id,
                vision_mode=vision_mode,
                tasks=tasks,
                context_role=context_role,
                metadata=metadata,
                trim_before_analysis=trim_before_analysis,
                trim_delivery_default=trim_delivery_default,
            )
        except Exception as exc:
            print(f"    ❌ Failed to start analysis: {exc}")
            continue

        if not poll:
            print(f"    ✅ Task queued: {task_id}")
            continue

        print(f"    … waiting for task {task_id}")
        while True:
            try:
                info = poll_task(base_url, api_key, task_id)
            except Exception as exc:
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
                    print(f"    ✅ Completed task {task_id}")
                else:
                    print(f"    ❌ Task {task_id} failed: {info.get('error')}")
                break

            time.sleep(POLL_INTERVAL_SECONDS)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch trigger product annotations via storage-api")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Storage API base URL")
    parser.add_argument("--api-key", required=True, help="API key with admin access to the tenant")
    parser.add_argument("--tenant", default=None, help="Tenant ID to filter (optional – inferred from API key)")
    parser.add_argument("--object-id", type=int, default=None, help="Process only this storage object ID")
    parser.add_argument("--limit", type=int, default=5000, help="Maximum objects to scan")
    parser.add_argument(
        "--context-contains",
        default=None,
        help="Only process objects whose context contains this substring",
    )
    parser.add_argument("--force", action="store_true", help="Re-run analysis even if annotations already exist")
    parser.add_argument("--no-poll", action="store_true", help="Do not wait for task completion")
    parser.add_argument("--vision-mode", default="product", help="Vision mode (default: product)")
    parser.add_argument(
        "--tasks",
        default="vision,embedding,kg",
        help="Comma-separated AI tasks to run (default: vision,embedding,kg)",
    )
    parser.add_argument("--context-role", default="product", help="Context role hint (default: product)")
    parser.add_argument(
        "--metadata-json",
        default=None,
        help="Optional JSON string to pass as ai_metadata for every run",
    )
    parser.add_argument(
        "--trim-before-analysis",
        action="store_true",
        help="Request trimming before analysis",
    )
    parser.add_argument(
        "--trim-delivery-default",
        action="store_true",
        help="Set trim_delivery_default flag when triggering analysis",
    )
    parser.add_argument(
        "--skip-if-embedding",
        action="store_true",
        help="Skip objects that already have embeddings",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    try:
        if args.object_id is not None:
            obj = fetch_single_object(args.base_url, args.api_key, args.object_id)
            if obj is None:
                print(f"Object {args.object_id} not found or not accessible.")
                return 1
            objects = [obj]
        else:
            objects = fetch_objects(
                args.base_url,
                args.api_key,
                args.tenant,
                args.limit,
                args.context_contains,
            )
    except Exception as exc:
        print(f"Failed to fetch objects: {exc}")
        return 1

    if not objects:
        print("No matching objects found.")
        return 0

    if args.object_id is not None:
        print(f"Processing single object {args.object_id}…")
    else:
        print(f"Found {len(objects)} image objects. Starting annotation batch…")

    ensure_annotations(
        args.base_url,
        args.api_key,
        objects,
        force=args.force,
        poll=not args.no_poll,
        vision_mode=args.vision_mode,
        tasks=args.tasks,
        context_role=args.context_role,
        metadata=args.metadata_json,
        trim_before_analysis=args.trim_before_analysis,
        trim_delivery_default=args.trim_delivery_default,
        skip_if_embedding=args.skip_if_embedding,
    )

    print("Batch run finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


