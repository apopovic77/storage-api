# 🚀 Deployment Guide - Storage API

## 📦 Persistente Daten

| Ressource              | Pfad                              | Hinweis                          |
|------------------------|-----------------------------------|----------------------------------|
| Deploy-Repo            | `/var/www/api-storage.arkturian.com` | Wird bei jedem Deploy gereinigt |
| Application Data Dir   | `/var/lib/storage-api`            | SQLite-DB, WAL, Chroma           |
| Uploads                | `/mnt/backup-disk/uploads/storage`| Unabhängig vom Repo              |

```bash
# .env (Auszug)
STORAGE_DATA_DIR=/var/lib/storage-api
STORAGE_DATABASE_URL=sqlite:////var/lib/storage-api/storage.db
CHROMA_DB_PATH=/var/lib/storage-api/chroma_db
STORAGE_UPLOAD_DIR=/mnt/backup-disk/uploads/storage
```

Die GitHub Action verschiebt bei Bedarf alte `.db`/`chroma_db`-Artefakte automatisch nach `/var/lib/storage-api`, führt anschließend `git clean -fdx -e .env` aus und sorgt damit für ein sauberes Working Directory ohne Datenverlust.

## 🔒 Deployment-Sicherheit

### 1. Pre-Deploy Checks

```bash
# VOR jedem Deploy prüfen:
ssh root@arkturian.com "
echo 'DB:' && ls -lh /var/lib/storage-api/storage.db
  echo 'Uploads:' && du -sh /var/www/uploads/storage/
  echo 'ChromaDB:' && du -sh /var/www/api-storage.arkturian.com/chroma_db/
"
```

### 2. Backup vor Deploy

```bash
# Automatisches Backup (bereits im Workflow):
cp /var/lib/storage-api/storage.db \
   /var/backups/storage-db-$(date +%Y%m%d-%H%M%S).db
```

### 3. Restore bei Problemen

```bash
# DB restore:
cp /var/backups/storage-db-TIMESTAMP.db \
   /var/lib/storage-api/storage.db

# Service restart:
systemctl restart storage-api
```

## 📊 Monitoring

### File-Counts überwachen:

```bash
# Nach jedem Deploy prüfen:
echo "Uploads:" && find /var/www/uploads/storage -type f | wc -l
echo "DB Objects:" && sqlite3 /var/lib/storage-api/storage.db \
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

