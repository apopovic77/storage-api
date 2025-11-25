# Local Transcoding API Setup

This guide explains how to set up the local transcoding API on arkserver.

## Architecture

- **arkturian.com**: Storage API calls Mac Transcoding API via SSH tunnel (localhost:8087)
- **arkserver**: Storage API calls Local Transcoding API (localhost:8087) which uses FFmpeg

Both use the same API interface, so the storage-api code remains identical.

## Prerequisites

1. **FFmpeg** (already installed on arkserver ✅)
2. **Python transcoding package** (needs to be installed)

## Installation Steps

### 1. Install Transcoding Package

On arkserver, install the arkturian-transcoding package:

```bash
ssh root@arkserver

cd /var/www/storage-api

# Install from local repo (development)
pip install -e /path/to/mac_transcoding_api

# OR install from git (production)
source venv/bin/activate
pip install git+https://github.com/apopovic77/mac-transcoding-api.git
```

### 2. Install Service File

```bash
# Copy service file
cp /var/www/storage-api/transcoding-api.service /etc/systemd/system/

# Reload systemd
systemctl daemon-reload

# Enable service to start on boot
systemctl enable transcoding-api

# Start service
systemctl start transcoding-api

# Check status
systemctl status transcoding-api
```

### 3. Verify Installation

```bash
# Check if API is running
curl http://localhost:8087/

# Expected output:
# {
#   "service": "Local Transcoding API",
#   "status": "online",
#   "active_jobs": 0,
#   "queue_size": 0,
#   "current_job": null,
#   "mode": "local_ffmpeg"
# }

# Check health
curl http://localhost:8087/health

# Expected output:
# {
#   "status": "healthy",
#   "ffmpeg": "available"
# }
```

### 4. Check Logs

```bash
# Follow logs
journalctl -u transcoding-api -f

# Recent logs
journalctl -u transcoding-api -n 100
```

## Testing

Upload a test video to storage-api on arkserver:

```bash
curl -X POST https://api-storage.arkserver.arkturian.com/storage/upload \
  -H "X-API-KEY: your_api_key" \
  -F "file=@test_video.mp4" \
  -F "context=test"
```

Then check transcoding-api logs:

```bash
journalctl -u transcoding-api -f
```

Expected log output:
```
📤 Transcoding job queued: job_12345_abcd1234 - test_video.mp4
⬇️  Downloading: https://api-storage.arkserver.arkturian.com/storage/files/12345
✅ Downloaded: /tmp/transcode_job_12345_abcd1234/test_video.mp4 (15728640 bytes)
🎬 Transcoding: /tmp/transcode_job_12345_abcd1234/test_video.mp4
✅ Transcoded 3 variants
   - 1080p: 1920x1080 @ 5.0 Mbps
   - 720p: 1280x720 @ 2.5 Mbps
   - 480p: 854x480 @ 1.0 Mbps
✅ Job completed: job_12345_abcd1234
```

## Troubleshooting

### Service won't start

```bash
# Check logs for errors
journalctl -u transcoding-api -n 50

# Common issues:
# 1. Transcoding package not installed
#    Fix: pip install git+https://github.com/apopovic77/mac-transcoding-api.git

# 2. Port 8087 already in use
#    Fix: Check what's using the port: lsof -i :8087

# 3. FFmpeg not found
#    Fix: apt install ffmpeg
```

### Package not found error

```bash
# Reinstall transcoding package
cd /var/www/storage-api
source venv/bin/activate
pip install --force-reinstall git+https://github.com/apopovic77/mac-transcoding-api.git
```

### Jobs failing

```bash
# Check specific job logs
journalctl -u transcoding-api | grep "job_12345"

# Check FFmpeg availability
ffmpeg -version

# Test transcoding manually
cd /tmp
ffmpeg -i test.mp4 -c:v libx264 -preset fast output.mp4
```

## Deployment

The transcoding-api.service file will be automatically deployed when you push to the storage-api repo.

To manually deploy:

```bash
# On local machine
git add transcoding_api_server.py transcoding-api.service TRANSCODING_API_SETUP.md
git commit -m "Add local transcoding API for arkserver"
git push origin main

# GitHub Actions will deploy to arkserver
# Then SSH to arkserver and set up the service:

ssh root@arkserver
cd /var/www/storage-api
pip install git+https://github.com/apopovic77/mac-transcoding-api.git
cp transcoding-api.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable transcoding-api
systemctl start transcoding-api
```

## Monitoring

### Check API Status

```bash
# API health
curl http://localhost:8087/health

# Current jobs
curl http://localhost:8087/
```

### Check Job Status

```bash
# Get status of specific job
curl http://localhost:8087/status/job_12345_abcd1234
```

### Service Status

```bash
# Service status
systemctl status transcoding-api

# Resource usage
systemctl status transcoding-api | grep -E "Memory|CPU"

# Restart if needed
systemctl restart transcoding-api
```

## Files

- `transcoding_api_server.py` - FastAPI server
- `transcoding-api.service` - systemd service file
- `TRANSCODING_API_SETUP.md` - This documentation

## Support

For issues:
1. Check logs: `journalctl -u transcoding-api -f`
2. Verify FFmpeg: `ffmpeg -version`
3. Test API: `curl http://localhost:8087/health`
4. Check package: `pip list | grep transcoding`
