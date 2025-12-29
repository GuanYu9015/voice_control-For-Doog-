#!/bin/bash
# 在 NanoLLM 容器內執行語音控制程式

CONTAINERS_DIR="${HOME}/jetson-containers"
VOICE_CONTROL_DIR="/home/jetson/voice_control"

# 掛載語音控制專案目錄
cd "$CONTAINERS_DIR"
jetson-containers run \
    -v "$VOICE_CONTROL_DIR:/workspace/voice_control" \
    $(autotag nano_llm) \
    python3 /workspace/voice_control/src/pipeline.py "$@"
