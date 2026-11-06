#!/usr/bin/env bash
# Push this update to the existing repository https://github.com/iitrky/feddt-bayeslstm
# Run inside this folder with git credentials for iitrky (gh auth login, or a personal access token).
set -e
git remote get-url origin >/dev/null 2>&1 || git remote add origin https://github.com/iitrky/feddt-bayeslstm.git
git fetch origin main
git merge-base --is-ancestor origin/main HEAD   # stops here if GitHub has commits this copy does not
git push origin main
echo "Updated: https://github.com/iitrky/feddt-bayeslstm"
