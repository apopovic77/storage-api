#!/usr/bin/env python3
"""Idempotent cross-host storage asset migration (HTTP-based).

Copies a scoped set of storage objects from a SOURCE storage-api host to a
TARGET host (e.g. api-storage.arkturian.com -> api-storage.arkserver.arkturian.com)
and emits a machine-readable old->new ID map (JSON) for downstream reference
rewrites (e.g. artrack file_url / storage_host per asset).

Why HTTP (not direct DB/file copy): the two instances have separate DBs +
filesystems; the public API is the portable, safe interface and reuses the
upload pipeline (re-extracts title/GPS from the file, and re-transcodes videos
on the TARGET so their hls_url points at the target host — which is exactly the
hls_url-rewrite the migration needs).

Idempotency: the --out map file is the source of truth. On re-run, an old_id
already mapped to a still-existing target object is skipped. ArTrack's already
-copied assets can be pre-seeded via --seed-map (old:new JSON) so they're not
re-copied.

Scope is an explicit --ids list (the caller knows which assets belong to
Tschepp-AR / GPS-Guide). re-embed is OFF by default (verified no-op for that
scope); enable with --reembed if a target app uses semantic search.

Example:
  ./venv/bin/python scripts/migrate_assets.py \
      --source https://api-storage.arkturian.com \
      --target https://api-storage.arkserver.arkturian.com \
      --key "$MASTER_KEY" \
      --ids 101802-101832,100821 \
      --seed-map track30_seed.json \
      --out migration_map.json --wait-transcode --dry-run
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone

import httpx


def parse_ids(spec: str):
    """'101802-101832,100821' -> [101802,...,101832,100821] (ordered, unique)."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    seen, uniq = set(), []
    for i in out:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    return uniq


def get_object(client, host, key, oid):
    r = client.get(f"{host}/storage/objects/{oid}", headers={"X-API-KEY": key}, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def download_original(client, host, key, oid):
    """Fetch the original bytes (no transform params) + content-type."""
    r = client.get(f"{host}/storage/media/{oid}", headers={"X-API-KEY": key}, timeout=300)
    r.raise_for_status()
    return r.content, r.headers.get("content-type", "application/octet-stream")


def upload(client, host, key, *, filename, data, content_type, src):
    """Upload to the target. ai_mode=none (assets already vetted on source);
    videos still transcode on the target (transcoding is not gated by ai_mode)."""
    form = {
        "original_filename": filename,
        "context": src.get("context") or "",
        "is_public": "true" if src.get("is_public") else "false",
        "ai_mode": "none",
        "storage_mode": "copy",
        "reuse_existing": "false",
    }
    if src.get("collection_id"):
        form["collection_id"] = src["collection_id"]
    if src.get("link_id"):
        form["link_id"] = src["link_id"]
    if src.get("owner_email"):
        form["owner_email"] = src["owner_email"]
    files = {"file": (filename, data, content_type)}
    r = client.post(f"{host}/storage/upload", data=form, files=files,
                    headers={"X-API-KEY": key}, timeout=600)
    r.raise_for_status()
    return r.json()


def patch_object(client, host, key, oid, fields):
    if not fields:
        return
    r = client.patch(f"{host}/storage/objects/{oid}", json=fields,
                     headers={"X-API-KEY": key}, timeout=30)
    if r.status_code >= 400:
        print(f"    WARN: PATCH #{oid} {list(fields)} -> {r.status_code} {r.text[:160]}")


def wait_transcode(client, host, key, oid, timeout_s=900):
    """Poll target object until transcoding completes (videos). Returns the obj."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        o = get_object(client, host, key, oid)
        st = (o or {}).get("transcoding_status")
        if st in (None, "completed", "failed", "disabled", ""):
            return o
        time.sleep(5)
    return get_object(client, host, key, oid)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="source storage-api base URL")
    ap.add_argument("--target", required=True, help="target storage-api base URL")
    ap.add_argument("--key", required=True, help="X-API-KEY (master key, valid on both)")
    ap.add_argument("--ids", required=True, help="scope: e.g. 101802-101832,100821")
    ap.add_argument("--out", required=True, help="path to the old->new map JSON (state + output)")
    ap.add_argument("--seed-map", help="JSON {old:new} of assets already copied (skip re-copy)")
    ap.add_argument("--reembed", action="store_true", help="trigger re-embed on target (default off)")
    ap.add_argument("--wait-transcode", action="store_true", help="poll videos until hls_url ready")
    ap.add_argument("--dry-run", action="store_true", help="report plan, write nothing")
    args = ap.parse_args()

    ids = parse_ids(args.ids)

    # Load existing state (idempotency) + seed map
    state = {"mapping": {}, "errors": []}
    try:
        with open(args.out) as f:
            state = json.load(f)
            state.setdefault("mapping", {})
            state.setdefault("errors", [])
    except FileNotFoundError:
        pass
    if args.seed_map:
        with open(args.seed_map) as f:
            for old, new in json.load(f).items():
                state["mapping"].setdefault(str(old), {"new_id": int(new), "status": "seeded"})

    state["source_host"] = args.source.rstrip("/")
    state["target_host"] = args.target.rstrip("/")

    def save():
        if not args.dry_run:
            with open(args.out, "w") as f:
                json.dump(state, f, indent=2)

    src_host, tgt_host = args.source.rstrip("/"), args.target.rstrip("/")
    migrated = reused = skipped = errors = 0

    with httpx.Client() as client:
        for oid in ids:
            key_s = str(oid)
            existing = state["mapping"].get(key_s)
            # Idempotency: skip if already mapped and the target still exists
            if existing and existing.get("new_id"):
                tgt = get_object(client, tgt_host, args.key, existing["new_id"])
                if tgt:
                    print(f"#{oid} already -> #{existing['new_id']} ({existing.get('status')}); skip")
                    reused += 1
                    continue
                else:
                    print(f"#{oid} mapped to #{existing['new_id']} but target gone; re-copy")

            src = get_object(client, src_host, args.key, oid)
            if not src:
                print(f"#{oid} NOT FOUND on source; skip")
                state["errors"].append({"src_id": oid, "error": "source_not_found"})
                errors += 1
                continue
            if src.get("storage_mode") == "external":
                print(f"#{oid} external storage_mode; skip (no local bytes)")
                state["errors"].append({"src_id": oid, "error": "external_mode_unsupported"})
                errors += 1
                continue

            mime = src.get("mime_type", "")
            fname = src.get("original_filename") or f"obj_{oid}"
            print(f"#{oid} {mime} {fname!r} -> copy to target ...")
            if args.dry_run:
                state["mapping"][key_s] = {"new_id": None, "status": "would-migrate",
                                           "mime": mime, "title": src.get("title")}
                continue

            try:
                data, ctype = download_original(client, src_host, args.key, oid)
                ctype = mime or ctype
                new = upload(client, tgt_host, args.key,
                             filename=fname, data=data, content_type=ctype, src=src)
                new_id = new["id"]

                # Preserve user-set fields the upload pipeline doesn't carry
                # (title from non-embedded sources, description). lat/lon/title
                # from embedded tags are re-extracted automatically on upload.
                patch = {}
                if src.get("title") and not new.get("title"):
                    patch["title"] = src["title"]
                if src.get("description"):
                    patch["description"] = src["description"]
                patch_object(client, tgt_host, args.key, new_id, patch)

                entry = {
                    "new_id": new_id,
                    "status": "migrated",
                    "mime": mime,
                    "title": patch.get("title") or new.get("title"),
                    "checksum": new.get("checksum"),
                    "file_url": f"{tgt_host}/storage/media/{new_id}",
                    "hls_url": new.get("hls_url"),
                    "transcoding_status": new.get("transcoding_status"),
                }

                if mime.startswith("video/") and args.wait_transcode:
                    print(f"    waiting for transcode of #{new_id} ...")
                    o = wait_transcode(client, tgt_host, args.key, new_id)
                    if o:
                        entry["hls_url"] = o.get("hls_url")
                        entry["transcoding_status"] = o.get("transcoding_status")

                if args.reembed:
                    try:
                        client.post(f"{tgt_host}/storage/objects/{new_id}/embed",
                                    headers={"X-API-KEY": args.key}, timeout=120)
                        entry["reembedded"] = True
                    except Exception as e:  # noqa: BLE001
                        print(f"    WARN: re-embed #{new_id} failed: {e}")

                state["mapping"][key_s] = entry
                print(f"    OK #{oid} -> #{new_id}"
                      + (f"  hls={entry['transcoding_status']}" if mime.startswith('video/') else ""))
                migrated += 1
                save()  # persist incrementally for crash-safety
            except Exception as e:  # noqa: BLE001
                print(f"    ERROR #{oid}: {e}")
                state["errors"].append({"src_id": oid, "error": str(e)})
                errors += 1

    state["migrated_at"] = datetime.now(timezone.utc).isoformat()
    save()
    verb = "(dry-run) would migrate" if args.dry_run else "migrated"
    print(f"\nDone. {verb}: {migrated} | reused/seeded: {reused} | errors: {errors}")
    print(f"Map written to: {args.out}" if not args.dry_run else "(dry-run: no map written)")


if __name__ == "__main__":
    main()
