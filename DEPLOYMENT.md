# 🚀 Deployment Guide - Storage API

## ⚠️ KRITISCH: Upload-Verzeichnis Schutz

### Aktuelle Konfiguration

**Uploads-Location:** `/var/www/uploads/storage/` (AUSSERHALB Deploy-Path)
**Deploy-Path:** `/var/www/api-storage.arkturian.com/`

```bash
# .env auf Server:
STORAGE_UPLOAD_DIR=/var/www/uploads/storage
```

### ✅ Was ist geschützt:

```yaml
# .github/workflows/deploy.yml
rsync --delete \
  --exclude='storage.db'    # Datenbank
  --exclude='chroma_db'     # Vector Embeddings
  "$REPO/" "$DEPLOY_PATH/"
```

**WICHTIG:** Uploads liegen in `/var/www/uploads/` - **NICHT** im Deploy-Path!
→ rsync kann sie NICHT löschen ✅

### ❌ Was war früher falsch:

```
/var/www/api-storage.arkturian.com/uploads/  ← ALTE Location (IM Deploy-Path!)
→ rsync --delete hat diese GELÖSCHT bei jedem Deploy
```

## 🔒 Deployment-Sicherheit

### 1. Pre-Deploy Checks

```bash
# VOR jedem Deploy prüfen:
ssh root@arkturian.com "
  echo 'DB:' && ls -lh /var/www/api-storage.arkturian.com/storage.db
  echo 'Uploads:' && du -sh /var/www/uploads/storage/
  echo 'ChromaDB:' && du -sh /var/www/api-storage.arkturian.com/chroma_db/
"
```

### 2. Backup vor Deploy

```bash
# Automatisches Backup (bereits im Workflow):
cp /var/www/api-storage.arkturian.com/storage.db \
   /var/backups/storage-db-$(date +%Y%m%d-%H%M%S).db
```

### 3. Restore bei Problemen

```bash
# DB restore:
cp /var/backups/storage-db-TIMESTAMP.db \
   /var/www/api-storage.arkturian.com/storage.db

# Service restart:
systemctl restart storage-api
```

## 📊 Monitoring

### File-Counts überwachen:

```bash
# Nach jedem Deploy prüfen:
echo "Uploads:" && find /var/www/uploads/storage -type f | wc -l
echo "DB Objects:" && sqlite3 /var/www/api-storage.arkturian.com/storage.db \
  "SELECT COUNT(*) FROM storage_objects;"
```

### Erwartete Werte (27. Okt 2025):
- **Uploads:** 5 Files (2 media + 2 thumbnails + 1 webview)
- **DB Objects:** 3650 (1944 external refs + 1706 missing + 2 actual files)

## 🛡️ Worst-Case Recovery

Falls Uploads gelöscht werden:

1. **Service stoppen:**
   ```bash
   systemctl stop storage-api
   ```

2. **Backup restore:**
   ```bash
   # NUR wenn Backup existiert mit Files!
   cp -r /var/backups/storage-api-TIMESTAMP/uploads/* \
         /var/www/uploads/storage/
   ```

3. **Permissions fixen:**
   ```bash
   chown -R root:root /var/www/uploads/storage
   chmod -R 755 /var/www/uploads/storage
   ```

4. **Service starten:**
   ```bash
   systemctl start storage-api
   ```

## 🔄 Deployment Workflow

```bash
# Manueller Deploy (safe):
cd /Volumes/DatenAP/Code/storage-api
./devops release --no-build

# Was passiert:
# 1. dev → main merge
# 2. Push zu GitHub
# 3. GitHub Actions triggert
# 4. rsync mit excludes
# 5. pip install
# 6. systemctl restart
```

## ⚠️ NIEMALS:

- `rm -rf /var/www/uploads/` ❌
- `rsync --delete` ohne excludes ❌
- Uploads im Deploy-Path ablegen ❌

## ✅ IMMER:

- Uploads AUSSERHALB Deploy-Path ✅
- DB & ChromaDB excluden ✅
- Backups vor Deploy ✅
- File-Count nach Deploy checken ✅

