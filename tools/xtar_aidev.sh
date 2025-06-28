 tartxt.py \
 /Users/jim/src/apps/c4h_ai_dev/c4h_agents \
 /Users/jim/src/apps/c4h_ai_dev/c4h_services \
 /Users/jim/src/apps/c4h_ai_dev/config \
 -x "**/.pytest_cache/,**/__pycache__/**,**/.git/**,**/*.pyc,**/node_modules/**,**/package-lock.json,**/dist/**,**/.DS_Store,**/README.md,**/workspaces/**,**/*.toml" \
  -f /Users/jim/src/apps/c4h_projects/backup_txt/c4h_ai_full.txt
tartxt.py \
  /Users/jim/src/apps/c4h_projects/docs/design_docs/ \
 -x "**/__pycache__/**,**/.git/**,**/*.pyc,**/node_modules/**,**/package-lock.json,**/dist/**,**/.DS_Store,**/README.md,**/workspaces/**,**/*.toml" \
  -f /Users/jim/src/apps/c4h_projects/backup_txt/c4h_master_doc.txt
