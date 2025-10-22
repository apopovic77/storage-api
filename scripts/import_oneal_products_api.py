#!/usr/bin/env python3
"""
Import O'Neal product images via API (no direct database access).

This script:
1. Fetches products from O'Neal API
2. Uploads to Storage API as external references
3. Triggers AI analysis and embeddings automatically
"""

import asyncio
import httpx
import sys


ONEAL_API_URL = "https://oneal-api.arkturian.com/v1/products"
ONEAL_API_KEY = "oneal_demo_token"
STORAGE_API_URL = "https://api.arkturian.com/storage/upload"


async def fetch_oneal_products(limit: int = 10):
    """Fetch products from O'Neal API."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{ONEAL_API_URL}?format=resolved&limit={limit}",
            headers={"X-API-Key": ONEAL_API_KEY}
        )
        response.raise_for_status()
        data = response.json()
        return data.get("results", []), data.get("count", 0)


async def upload_product_to_storage(product: dict):
    """Upload product image to storage via API."""
    product_id = product.get("id")
    name = product.get("name", "Unknown")
    brand = product.get("brand", "")
    season = product.get("season", "")
    category = product.get("category", [])

    if isinstance(category, list):
        category_str = " > ".join(category)
    else:
        category_str = str(category)

    # Extract hero image URL
    media = product.get("media", {})
    hero = media.get("hero", {})
    variants = hero.get("variants", {})
    image_url = variants.get("preview") or variants.get("thumb") or variants.get("print")

    if not image_url:
        return None, "No image URL"

    # Download image bytes
    async with httpx.AsyncClient(timeout=60.0) as client:
        img_response = await client.get(image_url)
        img_response.raise_for_status()
        image_data = img_response.content

    # Prepare metadata
    price = product.get("price", {})
    product_url = product.get("meta", {}).get("product_url", "")

    ai_context_metadata = {
        "product_id": product_id,
        "brand": brand,
        "season": season,
        "category": category_str,
        "name": name,
        "source": "oneal_api",
        "original_url": image_url,
        "price": price,
        "product_url": product_url
    }

    filename = image_url.split("/")[-1].split("?")[0]
    if not filename:
        filename = f"{product_id}.jpg"

    # Upload via Storage API
    async with httpx.AsyncClient(timeout=120.0) as client:
        files = {
            'file': (filename, image_data, 'image/jpeg')
        }
        data = {
            'title': name,
            'description': f"{brand} {season} - {category_str}",
            'context': f"oneal_product_{product_id}",
            'storage_mode': 'external',
            'external_uri': image_url,
            'is_public': 'true',
            'ai_context_metadata': str(ai_context_metadata)
        }

        response = await client.post(
            STORAGE_API_URL,
            headers={"X-API-Key": ONEAL_API_KEY},
            files=files,
            data=data
        )
        response.raise_for_status()
        return response.json(), None


async def import_products(limit: int = 10):
    """Import O'Neal products via Storage API."""
    print(f"🔍 Fetching {limit} products from O'Neal API...")
    products, total_count = await fetch_oneal_products(limit)
    print(f"✅ Found {len(products)} products (out of {total_count} total)\n")

    imported = []
    errors = []

    for idx, product in enumerate(products, 1):
        product_id = product.get("id")
        name = product.get("name", "Unknown")
        brand = product.get("brand", "")

        print(f"📦 [{idx}/{len(products)}] Processing: {name} ({brand})")

        try:
            result, error = await upload_product_to_storage(product)

            if error:
                print(f"   ⚠️  Skipped: {error}\n")
                errors.append((product_id, error))
                continue

            storage_id = result.get("id")
            proxy_url = result.get("file_url")

            print(f"   ✅ Uploaded as storage object #{storage_id}")
            print(f"   🔗 Proxy URL: {proxy_url}\n")

            imported.append({
                "product_id": product_id,
                "storage_id": storage_id,
                "name": name,
                "proxy_url": proxy_url
            })

        except Exception as e:
            print(f"   ❌ Error: {e}\n")
            errors.append((product_id, str(e)))

    print("\n" + "="*60)
    print(f"✅ Import completed: {len(imported)}/{len(products)} products imported")
    if errors:
        print(f"⚠️  {len(errors)} errors")
    print("="*60)

    if imported:
        print(f"\n✅ Successfully imported {len(imported)} products")
        print("🤖 AI analysis and embeddings will be processed in background")

    return imported, errors


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Import O'Neal products via API")
    parser.add_argument("--limit", type=int, default=10, help="Number of products (default: 10)")
    args = parser.parse_args()

    print("="*60)
    print("O'NEAL PRODUCT IMPORT (via API)")
    print("="*60)
    print(f"Settings:")
    print(f"  • Source: {ONEAL_API_URL}")
    print(f"  • Target: {STORAGE_API_URL}")
    print(f"  • Limit: {args.limit} products")
    print(f"  • Storage Mode: external (proxy)")
    print(f"  • AI Analysis: automatic (background)")
    print("="*60 + "\n")

    try:
        await import_products(limit=args.limit)
    except KeyboardInterrupt:
        print("\n\n⚠️  Import cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
