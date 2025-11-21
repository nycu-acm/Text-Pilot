#!/usr/bin/env bash
# 檔名：run_generate.sh

PYTHON_SCRIPT="./File_generate.py"
PYTHON_BIN=$(command -v python || command -v python)

while true; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 執行 ${PYTHON_SCRIPT} --range 324-1355"
  "$PYTHON_BIN" "$PYTHON_SCRIPT" --range 324-1355
  sleep 120
done

