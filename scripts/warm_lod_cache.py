#!/usr/bin/env python3
"""
LOD Cache Warming Script

Pre-warms the image transformation cache for all O'Neal product images
at the two LOD (Level of Detail) resolutions used by the Product Finder:
- 150px (low-res): Initial load for fast page rendering
- 1300px (high-res): Full detail upgrade after viewport detection

This ensures instant image delivery when users visit the Product Finder.

Usage:
    python scripts/warm_lod_cache.py [--tenant TENANT_ID]

Environment:
    Requires STORAGE_API_KEY environment variable or --api-key argument
"""

import requests
import sys
import time
import os
from pathlib import Path
from typing import List, Dict, Tuple
import argparse

# LOD Resolution Configuration
# These must match the settings in productfinder/src/config/lod.config.ts
LOD_RESOLUTIONS = [
    {
        'name': 'low-res (initial)',
        'width': 150,
        'format': 'webp',
        'quality': 75
    },
    {
        'name': 'high-res (upgrade)',
        'width': 1300,
        'format': 'webp',
        'quality': 90
    }
]

# API Configuration
STORAGE_API_URL = "https://api-storage.arkturian.com"
DEFAULT_TENANT = "oneal"


class CacheWarmer:
    def __init__(self, api_key: str, tenant_id: str = DEFAULT_TENANT):
        self.api_key = api_key
        self.tenant_id = tenant_id
        self.headers = {"X-API-Key": api_key}
        self.stats = {
            'total_objects': 0,
            'total_variants': 0,
            'cached': 0,
            'errors': 0,
            'start_time': time.time()
        }

    def get_storage_objects(self) -> List[Dict]:
        """Fetch all storage objects for the tenant."""
        print(f"🔍 Fetching storage objects for tenant '{self.tenant_id}'...")

        url = f"{STORAGE_API_URL}/storage/objects"
        params = {
            'limit': 1000,
            'offset': 0
        }

        all_objects = []

        while True:
            try:
                response = requests.get(url, headers=self.headers, params=params, verify=False)
                response.raise_for_status()
                data = response.json()

                objects = data.get('items', [])
                if not objects:
                    break

                # Filter for images only
                image_objects = [obj for obj in objects if obj.get('mime_type', '').startswith('image/')]
                all_objects.extend(image_objects)

                print(f"   Fetched {len(all_objects)} images so far...")

                # Check if we have more pages
                total = data.get('total', 0)
                if len(all_objects) >= total:
                    break

                params['offset'] += params['limit']

            except Exception as e:
                print(f"❌ Error fetching objects: {e}")
                break

        print(f"✅ Found {len(all_objects)} image objects")
        self.stats['total_objects'] = len(all_objects)
        return all_objects

    def warm_variant(self, object_id: int, resolution: Dict) -> Tuple[bool, int]:
        """
        Warm cache for a specific image variant.
        Returns (success, response_size_bytes)
        """
        url = f"{STORAGE_API_URL}/storage/media/{object_id}"
        params = {
            'width': resolution['width'],
            'format': resolution['format'],
            'quality': resolution['quality']
        }

        try:
            response = requests.get(url, params=params, verify=False, timeout=30)
            response.raise_for_status()

            size = len(response.content)
            return True, size

        except Exception as e:
            print(f"   ⚠️  Error for object {object_id} ({resolution['name']}): {e}")
            return False, 0

    def warm_object(self, obj: Dict, index: int, total: int) -> None:
        """Warm all LOD variants for a single object."""
        object_id = obj['id']
        filename = obj.get('original_filename', 'unknown')

        print(f"\n[{index}/{total}] Object {object_id}: {filename}")

        for resolution in LOD_RESOLUTIONS:
            success, size = self.warm_variant(object_id, resolution)

            if success:
                self.stats['cached'] += 1
                size_kb = size / 1024
                print(f"   ✅ {resolution['name']:20} | {resolution['width']}px | {size_kb:6.1f} KB")
            else:
                self.stats['errors'] += 1

            self.stats['total_variants'] += 1

            # Small delay to avoid overwhelming the server
            time.sleep(0.1)

    def run(self) -> None:
        """Execute the cache warming process."""
        print("=" * 80)
        print("LOD CACHE WARMING")
        print("=" * 80)
        print(f"Tenant:      {self.tenant_id}")
        print(f"Resolutions: {', '.join([f'{r['width']}px' for r in LOD_RESOLUTIONS])}")
        print(f"API:         {STORAGE_API_URL}")
        print("=" * 80)
        print()

        # Get all objects
        objects = self.get_storage_objects()

        if not objects:
            print("❌ No objects found to warm")
            return

        print(f"\n🔥 Starting cache warming for {len(objects)} objects...")
        print(f"   This will create {len(objects) * len(LOD_RESOLUTIONS)} cached variants")
        print()

        # Warm each object
        for index, obj in enumerate(objects, start=1):
            self.warm_object(obj, index, len(objects))

        # Print final statistics
        self.print_stats()

    def print_stats(self) -> None:
        """Print final statistics."""
        elapsed = time.time() - self.stats['start_time']

        print()
        print("=" * 80)
        print("CACHE WARMING COMPLETE")
        print("=" * 80)
        print(f"Total objects:        {self.stats['total_objects']}")
        print(f"Total variants:       {self.stats['total_variants']}")
        print(f"Successfully cached:  {self.stats['cached']}")
        print(f"Errors:               {self.stats['errors']}")
        print(f"Time elapsed:         {elapsed:.1f}s")

        if self.stats['total_variants'] > 0:
            avg_time = elapsed / self.stats['total_variants']
            success_rate = (self.stats['cached'] / self.stats['total_variants']) * 100
            print(f"Average per variant:  {avg_time:.2f}s")
            print(f"Success rate:         {success_rate:.1f}%")

        print("=" * 80)

        if self.stats['errors'] > 0:
            print(f"\n⚠️  {self.stats['errors']} variants failed to cache")
            sys.exit(1)
        else:
            print("\n✅ All variants successfully cached!")


def main():
    parser = argparse.ArgumentParser(description='Warm LOD image cache for Product Finder')
    parser.add_argument('--api-key', help='Storage API key (or set STORAGE_API_KEY env var)')
    parser.add_argument('--tenant', default=DEFAULT_TENANT, help=f'Tenant ID (default: {DEFAULT_TENANT})')
    parser.add_argument('--base-url', default=STORAGE_API_URL, help=f'API base URL (default: {STORAGE_API_URL})')

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.getenv('STORAGE_API_KEY')
    if not api_key:
        print("❌ Error: No API key provided")
        print("   Use --api-key argument or set STORAGE_API_KEY environment variable")
        sys.exit(1)

    # Update global config if custom URL provided
    if args.base_url != STORAGE_API_URL:
        global STORAGE_API_URL
        STORAGE_API_URL = args.base_url

    # Run cache warming
    warmer = CacheWarmer(api_key=api_key, tenant_id=args.tenant)

    try:
        warmer.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Cache warming interrupted by user")
        warmer.print_stats()
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
