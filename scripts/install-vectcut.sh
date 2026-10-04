#!/bin/zsh
# One-time, idempotent install of VectCutAPI next to cutlogic, configured for
# CapCut desktop on macOS (draft_profile capcut_legacy writes draft_info.json).
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DIR="${VECTCUT_DIR:-$HERE/../VectCutAPI}"
if [ ! -d "$DIR/.git" ]; then
  git clone --depth 1 https://github.com/sun-guannan/VectCutAPI.git "$DIR"
fi
cd "$DIR"
PY="${VECTCUT_PYTHON:-python3}"
if [ ! -x venv-capcut/bin/python ]; then
  "$PY" -m venv venv-capcut
fi
venv-capcut/bin/pip install -q --upgrade pip
venv-capcut/bin/pip install -q -r requirements.txt
if [ ! -f config.json ]; then
  # config.json.example is not strict JSON (comments); write the real thing.
  cat > config.json <<'JSON'
{
  "draft_profile": "capcut_legacy",
  "is_capcut_env": true,
  "draft_domain": "http://127.0.0.1:9001",
  "port": 9001,
  "preview_router": "/draft/downloader",
  "is_upload_draft": false
}
JSON
fi
echo "VectCutAPI ready at $DIR (profile: $(python3 -c 'import json;print(json.load(open("config.json"))["draft_profile"])'))"
