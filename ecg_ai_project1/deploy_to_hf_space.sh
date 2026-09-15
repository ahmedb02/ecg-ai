#!/usr/bin/env bash
# Deploys this project to a Hugging Face Space (Docker SDK) so you get a
# real running link, e.g. https://huggingface.co/spaces/<you>/ecg-ai
#
# Run this FROM THE PROJECT ROOT (the directory containing this script,
# Dockerfile, backend/, frontend/, ecg_ai/) on a machine with normal
# internet access -- it can't run inside the sandboxed session that wrote
# it, which has no route to huggingface.co.
#
# Usage:
#   HF_TOKEN=hf_xxx ./deploy_to_hf_space.sh
#   HF_TOKEN=hf_xxx SPACE_NAME=my-ecg-app ./deploy_to_hf_space.sh
#
# HF_TOKEN must be a Hugging Face access token with Write scope
# (https://huggingface.co/settings/tokens). Pass it as an env var, not a
# script edit -- this script is meant to be safe to keep in version
# control, so it never hardcodes a real token.
#
# Requires: git, curl, python3.

set -euo pipefail

HF_TOKEN="${HF_TOKEN:-}"
SPACE_NAME="${SPACE_NAME:-ecg-ai}"
HF_PRIVATE="${HF_PRIVATE:-false}"

if [ -z "$HF_TOKEN" ]; then
  echo "ERROR: set HF_TOKEN to a Hugging Face write-access token, e.g.:" >&2
  echo "  HF_TOKEN=hf_xxx ./deploy_to_hf_space.sh" >&2
  exit 1
fi

for bin in git curl python3; do
  command -v "$bin" >/dev/null 2>&1 || { echo "ERROR: $bin is required but not found on PATH." >&2; exit 1; }
done

for f in Dockerfile backend frontend ecg_ai; do
  [ -e "$f" ] || { echo "ERROR: '$f' not found here -- run this script from the project root (where you unzipped it)." >&2; exit 1; }
done

echo "==> Verifying Hugging Face token and fetching your username..."
WHOAMI_JSON=$(curl -sS -H "Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/api/whoami-v2)
HF_USERNAME=$(python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('name') or d.get('user', {}).get('name', ''))" <<<"$WHOAMI_JSON")
if [ -z "$HF_USERNAME" ]; then
  echo "ERROR: could not determine your HF username -- token invalid/expired? Response was:" >&2
  echo "$WHOAMI_JSON" >&2
  exit 1
fi
echo "    logged in as: $HF_USERNAME"

echo "==> Creating Space '${HF_USERNAME}/${SPACE_NAME}' (Docker SDK) if it doesn't already exist..."
CREATE_RESPONSE=$(curl -sS -X POST "https://huggingface.co/api/repos/create" \
  -H "Authorization: Bearer ${HF_TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{\"type\":\"space\",\"name\":\"${SPACE_NAME}\",\"sdk\":\"docker\",\"private\":${HF_PRIVATE}}")
if echo "$CREATE_RESPONSE" | grep -qi '"error"'; then
  if echo "$CREATE_RESPONSE" | grep -qi 'already'; then
    echo "    Space already exists -- reusing it."
  else
    echo "ERROR creating Space:" >&2
    echo "$CREATE_RESPONSE" >&2
    exit 1
  fi
else
  echo "    created."
fi

echo "==> Preparing README.md with Space metadata..."
if ! head -n1 README.md 2>/dev/null | grep -q '^---$'; then
  TMP_README=$(mktemp)
  {
    echo "---"
    echo "title: ${SPACE_NAME}"
    echo "emoji: 🫀"
    echo "colorFrom: blue"
    echo "colorTo: purple"
    echo "sdk: docker"
    echo "app_port: 7860"
    echo "pinned: false"
    echo "---"
    echo
    cat README.md 2>/dev/null || true
  } > "$TMP_README"
  mv "$TMP_README" README.md
  echo "    frontmatter added."
else
  echo "    frontmatter already present, leaving as-is."
fi

echo "==> Committing and pushing to the Space (this directory becomes a fresh git repo for the push)..."
if [ ! -d .git ]; then
  git init -q
fi
git add -A
git commit -q -m "Deploy ecg_ai to Hugging Face Space" --allow-empty
git branch -M main

git remote remove hf-space 2>/dev/null || true
git remote add hf-space "https://${HF_USERNAME}:${HF_TOKEN}@huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"
git push hf-space main --force

echo
echo "============================================================"
echo "Deployed. Hugging Face is now building the Docker image --"
echo "this typically takes several minutes on first deploy."
echo
echo "Space page (build logs + the app once ready):"
echo "  https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"
echo
echo "Direct app URL (usually, once the build finishes):"
echo "  https://${HF_USERNAME}-${SPACE_NAME}.hf.space"
echo "============================================================"
echo
echo "Reminder: rotate/revoke the token used here at"
echo "https://huggingface.co/settings/tokens once you've confirmed it's live."
