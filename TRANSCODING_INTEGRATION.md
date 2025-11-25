# Video Transcoding Integration

## Overview

This document describes the video transcoding integration in storage-api using the reusable `arkturian-transcoding` package.

## Architecture

### Components

1. **Transcoding Package** (`/Volumes/DatenAP/Code/mac_transcoding_api/transcoding/`)
   - Reusable Python package installed via `pip install -e .`
   - Supports local FFmpeg and remote API transcoding
   - Clean OOP design with factory pattern

2. **Storage API Integration** (`/Volumes/DatenAP/Code/storage-api/`)
   - Configuration in `config.py`
   - Helper module in `storage/transcoding_helper.py`
   - Ready to integrate into `storage/routes.py`

### Transcoding Modes

- **`local`**: Uses FFmpeg directly on the server (default)
- **`remote`**: Delegates to external transcoding API
- **`disabled`**: No transcoding

## Configuration

### Environment Variables

Add to `.env` or systemd service:

```bash
# Required
TRANSCODING_MODE=local        # local, remote, or disabled

# For remote mode only
TRANSCODING_API_URL=http://localhost:8087
TRANSCODING_API_KEY=your_api_key_here
```

### Config File

Already added to `config.py` (lines 88-91):

```python
# Transcoding Settings
TRANSCODING_MODE: str = os.getenv("TRANSCODING_MODE", "local")
TRANSCODING_API_URL: Optional[str] = os.getenv("TRANSCODING_API_URL")
TRANSCODING_API_KEY: Optional[str] = os.getenv("TRANSCODING_API_KEY")
```

## Usage in Upload Endpoint

### Simple Integration

Add to `storage/routes.py` after video file is saved (around line 2300+):

```python
from storage.transcoding_helper import transcode_if_needed
from storage.service import generic_storage
from pathlib import Path

# After saving the file...
# saved_obj = await save_file_and_record(...)

# Trigger transcoding if this is a video
if saved_obj.mime_type and saved_obj.mime_type.startswith("video/"):
    try:
        # Get absolute path to uploaded file
        file_path = generic_storage.absolute_path_for_key(saved_obj.object_key, tenant_id)

        # Trigger transcoding (runs in background)
        await transcode_if_needed(
            source_path=Path(file_path),
            mime_type=saved_obj.mime_type,
            storage_object_id=saved_obj.id
        )
    except Exception as e:
        # Log error but don't fail the upload
        logging.error(f"Transcoding trigger failed: {e}")
```

### Advanced Usage

For more control, use the `TranscodingHelper` class directly:

```python
from storage.transcoding_helper import TranscodingHelper
from pathlib import Path

# Check if transcoding is enabled
if TranscodingHelper.is_enabled():
    # Check if this file should be transcoded
    if TranscodingHelper.should_transcode(mime_type):
        # Get paths
        source_path = Path(file_path)
        output_dir = source_path.parent / source_path.stem

        # Start background transcoding
        TranscodingHelper.start_background_transcoding(
            source_path,
            output_dir,
            storage_object_id
        )
```

## Package Details

### Transcoding Package Structure

```
mac_transcoding_api/
├── transcoding/                   # Python package
│   ├── __init__.py               # Exports
│   ├── models.py                 # Data models
│   ├── base.py                   # BaseTranscoder abstract class
│   ├── analyzer.py               # VideoAnalyzer (ffprobe)
│   ├── local.py                  # LocalFFmpegTranscoder
│   ├── remote.py                 # RemoteAPITranscoder
│   └── factory.py                # TranscoderFactory
├── setup.py                      # Package setup
├── requirements.txt              # Dependencies
└── README.md                     # Documentation
```

### Direct Package Usage

If you need more control, use the package directly:

```python
from transcoding import TranscoderFactory, TranscodingConfig, TranscodingMode
from pathlib import Path

# Create config
config = TranscodingConfig(mode=TranscodingMode.LOCAL)

# Create transcoder
transcoder = TranscoderFactory.create(config)

# Check availability
if await transcoder.check_availability():
    # Transcode
    result = await transcoder.transcode(
        Path("/path/to/video.mp4"),
        Path("/path/to/output")
    )

    # Check results
    if result.success:
        print(f"Created {len(result.variants)} variants")
        for variant in result.variants:
            print(f"{variant.name}: {variant.resolution}")
```

## Deployment

### On arkserver

1. **Install FFmpeg** (already done ✅):
   ```bash
   apt update && apt install -y ffmpeg
   ```

2. **Install transcoding package**:
   ```bash
   cd /var/www/storage-api

   # Option A: Install from local path (development)
   pip install -e /path/to/mac_transcoding_api

   # Option B: Install from git (production)
   pip install git+ssh://git@github.com/apopovic77/mac-transcoding-api.git
   ```

3. **Configure environment**:
   ```bash
   # Edit /etc/systemd/system/storage-api.service
   Environment="TRANSCODING_MODE=local"

   # Reload and restart
   systemctl daemon-reload
   systemctl restart storage-api
   ```

4. **Verify**:
   ```bash
   # Check logs
   journalctl -u storage-api -f
   ```

### On arkturian.com

Same steps as arkserver.

### For Remote Transcoding (Mac API)

```bash
# Configure to use remote API
Environment="TRANSCODING_MODE=remote"
Environment="TRANSCODING_API_URL=http://localhost:8087"
Environment="TRANSCODING_API_KEY=your_api_key"
```

## Testing

### Test Local Transcoding

```bash
# Upload a video file
curl -X POST https://api-storage.arkserver.arkturian.com/storage/upload \
  -H "X-API-KEY: your_api_key" \
  -F "file=@test_video.mp4" \
  -F "context=test"

# Check logs for transcoding output
journalctl -u storage-api -f | grep -i transcode
```

Expected output:
```
🎬 Starting transcoding for storage object 12345
   Mode: local
   Source: /path/to/video.mp4
   Output: /path/to/output
✅ Transcoding completed for storage object 12345
   Created 3 variants
      - 1080p: 1920x1080 @ 5.0 Mbps
      - 720p: 1280x720 @ 2.5 Mbps
      - 480p: 854x480 @ 1.0 Mbps
   Generated 3 thumbnails
```

### Test Configuration

```python
# In Python shell or test script
from config import settings

print(f"Transcoding mode: {settings.TRANSCODING_MODE}")
print(f"API URL: {settings.TRANSCODING_API_URL}")

# Test package import
from transcoding import TranscoderFactory
print("✅ Transcoding package available")
```

## Troubleshooting

### Package Not Found

```bash
# Reinstall package
cd /Volumes/DatenAP/Code/mac_transcoding_api
pip install -e .

# Or on server:
pip install git+ssh://git@github.com/apopovic77/mac-transcoding-api.git
```

### FFmpeg Not Found

```bash
# Install FFmpeg
apt update && apt install -y ffmpeg

# Verify
ffmpeg -version
```

### Transcoding Not Triggering

Check logs:
```bash
journalctl -u storage-api -f | grep -i transcode
```

Common issues:
- `TRANSCODING_MODE=disabled` - Change to `local` or `remote`
- FFmpeg not installed - Install FFmpeg
- Permission errors - Check file permissions

### Remote API Not Reachable

```bash
# Test connection
curl http://localhost:8087/health

# Check SSH tunnel if using remote Mac
ssh -L 8087:localhost:8087 user@mac-host
```

## Next Steps

1. **Integrate into routes.py** - Add transcoding call to upload endpoint
2. **Test on arkserver** - Upload test video and verify transcoding
3. **Monitor performance** - Check transcoding times and resource usage
4. **Deploy to production** - Roll out to arkturian.com

## Benefits

- ✅ **Single source of truth** - One reusable package
- ✅ **Multi-tenant safe** - Each deployment configures independently
- ✅ **Easy testing** - Install anywhere with `pip install -e .`
- ✅ **Flexible** - Supports local and remote transcoding
- ✅ **Future-proof** - Easy to add new transcoder types

## File Locations

- Package: `/Volumes/DatenAP/Code/mac_transcoding_api/transcoding/`
- Config: `/Volumes/DatenAP/Code/storage-api/config.py:88-91`
- Helper: `/Volumes/DatenAP/Code/storage-api/storage/transcoding_helper.py`
- Routes: `/Volumes/DatenAP/Code/storage-api/storage/routes.py:1741` (upload endpoint)

## Support

For issues or questions:
- Check logs: `journalctl -u storage-api -f`
- Review package README: `/Volumes/DatenAP/Code/mac_transcoding_api/README.md`
- Test package directly: `python -m transcoding.local` (if test module exists)
