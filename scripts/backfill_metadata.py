#!/usr/bin/env python
import sqlite3
import subprocess
import json
from pathlib import Path

DB_PATH = "/var/lib/api-arkturian/artrack.db"
MEDIA_ROOT = Path("/mnt/backup-disk/uploads/media")

def get_missing_metadata_files(conn):
    cursor = conn.cursor()
    # Get files that are images or videos and are missing width (a good proxy for missing metadata)
    cursor.execute("""
        SELECT id, object_key, mime_type
        FROM storage_objects
        WHERE (mime_type LIKE 'video/%' OR mime_type LIKE 'image/%')
          AND width IS NULL
    """)
    return cursor.fetchall()

def extract_metadata(file_path):
    print(f"  -> Probing {file_path}...")
    try:
        probe_cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", str(file_path)
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        probe_data = json.loads(result.stdout)
        
        duration = probe_data.get('format', {}).get('duration')
        bit_rate = probe_data.get('format', {}).get('bit_rate')

        video_stream = next((s for s in probe_data.get('streams', []) if s.get('codec_type') == 'video'), None)
        if video_stream:
            width = video_stream.get('width')
            height = video_stream.get('height')
            return {
                "width": int(width) if width else None,
                "height": int(height) if height else None,
                "duration_seconds": float(duration) if duration else None,
                "bit_rate": int(bit_rate) if bit_rate else None,
            }
        # Handle audio-only files
        audio_stream = next((s for s in probe_data.get('streams', []) if s.get('codec_type') == 'audio'), None)
        if audio_stream:
             return {
                "duration_seconds": float(duration) if duration else None,
                "bit_rate": int(bit_rate) if bit_rate else None,
            }

    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
        print(f"  -> ERROR probing {file_path}: {e}")
    return None

def update_database(conn, file_id, metadata):
    cursor = conn.cursor()
    print(f"  -> Updating DB for ID {file_id} with: {metadata}")
    try:
        cursor.execute("""
            UPDATE storage_objects
            SET width = ?, height = ?, duration_seconds = ?, bit_rate = ?
            WHERE id = ?
        """, (
            metadata.get('width'),
            metadata.get('height'),
            metadata.get('duration_seconds'),
            metadata.get('bit_rate'),
            file_id
        ))
        conn.commit()
    except Exception as e:
        print(f"  -> ERROR updating database for ID {file_id}: {e}")
        conn.rollback()

def main():
    print("Starting backfill of media metadata...")
    try:
        conn = sqlite3.connect(DB_PATH)
        files_to_process = get_missing_metadata_files(conn)
        
        if not files_to_process:
            print("No files found with missing metadata. All up to date!")
            return

        print(f"Found {len(files_to_process)} files to process.")
        
        for file_id, object_key, mime_type in files_to_process:
            print(f"Processing file ID: {file_id}, Key: {object_key}")
            file_path = MEDIA_ROOT / object_key
            
            if not file_path.exists():
                print(f"  -> WARNING: File not found on disk, skipping: {file_path}")
                continue
            
            metadata = extract_metadata(file_path)
            
            if metadata:
                update_database(conn, file_id, metadata)
            else:
                print("  -> No valid metadata extracted, skipping DB update.")
        
        conn.close()
        print("Metadata backfill complete!")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
