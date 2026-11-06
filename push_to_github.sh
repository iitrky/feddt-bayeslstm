#!/usr/bin/env bash
# Publish this repository as https://github.com/iitrky/feddt-bayeslstm
# Needs the GitHub CLI (https://cli.github.com) logged in as iitrky:  gh auth login
set -e
gh repo create iitrky/feddt-bayeslstm --public \
  --description "Federated Bayesian digital twin for battery health prognosis in electric mining fleets" \
  --source . --remote origin --push
echo "Published: https://github.com/iitrky/feddt-bayeslstm"
# Without the GitHub CLI: create an empty public repo named feddt-bayeslstm on github.com, then run
#   git remote add origin https://github.com/iitrky/feddt-bayeslstm.git && git push -u origin main
