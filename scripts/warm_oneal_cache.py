#!/usr/bin/env python3
"""
O'Neal LOD Cache Warming Script

Pre-warms the image transformation cache for all 616 O'Neal product images.
Storage IDs: 4208 - 4823

LOD Resolutions:
- 150px (low-res): Initial load for fast page rendering
- 1300px (high-res): Full detail upgrade after viewport detection

Usage:
    python scripts/warm_oneal_cache.py
"""

import requests
import time
import urllib3

# Disable SSL warnings for self-signed cert
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration
BASE_URL = "https://api-storage.arkturian.com/storage/media"
START_ID = 4208
END_ID = 4823
TOTAL_OBJECTS = 616

# LOD Resolutions
RESOLUTIONS = [
    {'name': 'low-res', 'width': 150, 'format': 'webp', 'quality': 75},
    {'name': 'high-res', 'width': 1300, 'format': 'webp', 'quality': 90}
]


def warm_variant(object_id, resolution):
    """Warm a single variant. Returns (success, size_bytes)."""
    url = f"{BASE_URL}/{object_id}"
    params = {
        'width': resolution['width'],
        'format': resolution['format'],
        'quality': resolution['quality']
    }

    try:
        response = requests.get(url, params=params, verify=False, timeout=30)
        response.raise_for_status()
        return True, len(response.content)
    except Exception as e:
        return False, 0


def main():
    print("=" * 80)
    print("O'NEAL LOD CACHE WARMING")
    print("=" * 80)
    print(f"Object IDs:  {START_ID} - {END_ID} ({TOTAL_OBJECTS} objects)")
    print(f"Resolutions: 150px (low-res), 1300px (high-res)")
    print(f"Total variants to cache: {TOTAL_OBJECTS * len(RESOLUTIONS)}")
    print("=" * 80)
    print()

    stats = {
        'cached': 0,
        'errors': 0,
        'total': 0,
        'start_time': time.time()
    }

    for obj_id in range(START_ID, END_ID + 1):
        index = obj_id - START_ID + 1
        print(f"\n[{index}/{TOTAL_OBJECTS}] Object ID {obj_id}")

        for resolution in RESOLUTIONS:
            success, size = warm_variant(obj_id, resolution)
            stats['total'] += 1

            if success:
                stats['cached'] += 1
                size_kb = size / 1024
                print(f"   ✅ {resolution['name']:8} | {resolution['width']:4}px | {size_kb:6.1f} KB")
            else:
                stats['errors'] += 1
                print(f"   ❌ {resolution['name']:8} | {resolution['width']:4}px | FAILED")

            # Small delay to avoid overwhelming server
            time.sleep(0.05)

    # Final stats
    elapsed = time.time() - stats['start_time']

    print()
    print("=" * 80)
    print("CACHE WARMING COMPLETE")
    print("=" * 80)
    print(f"Total variants:       {stats['total']}")
    print(f"Successfully cached:  {stats['cached']}")
    print(f"Errors:               {stats['errors']}")
    print(f"Time elapsed:         {elapsed:.1f}s")

    if stats['total'] > 0:
        avg_time = elapsed / stats['total']
        success_rate = (stats['cached'] / stats['total']) * 100
        print(f"Average per variant:  {avg_time:.2f}s")
        print(f"Success rate:         {success_rate:.1f}%")

    print("=" * 80)

    if stats['errors'] > 0:
        print(f"\n⚠️  {stats['errors']} variants failed to cache")
    else:
        print("\n✅ All O'Neal product images successfully cached!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Cache warming interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
