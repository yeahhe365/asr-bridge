#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

if command -v say >/dev/null 2>&1 && command -v afconvert >/dev/null 2>&1; then
  say -v Ting-Ting -o "$tmp_dir/input.aiff" "欢迎使用阿里云语音识别。"
  afconvert -f WAVE -d LEI16@16000 "$tmp_dir/input.aiff" "$tmp_dir/input.wav"
else
  printf 'RIFFdemo-audio' > "$tmp_dir/input.wav"
fi

curl -sS http://127.0.0.1:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer local" \
  -F model=fun-asr \
  -F language=zh \
  -F "file=@$tmp_dir/input.wav;type=audio/wav"
