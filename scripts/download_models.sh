#!/bin/bash
# 下載 Sherpa-onnx 模型
# 包含 VAD、KWS、ASR、TTS 模型

set -e

MODELS_DIR="${1:-/home/jetson/voice_control/models}"
echo "=== Sherpa-onnx 模型下載器 ==="
echo "模型目錄: $MODELS_DIR"

# 建立目錄
mkdir -p "$MODELS_DIR"/{vad,kws,asr,tts,dns}

# ============================
# VAD 模型 (Silero VAD)
# ============================
echo ""
echo "=== 下載 VAD 模型 (Silero VAD) ==="
VAD_DIR="$MODELS_DIR/vad"

if [ ! -f "$VAD_DIR/silero_vad.onnx" ]; then
    wget -O "$VAD_DIR/silero_vad.onnx" \
        https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
    echo "VAD 模型下載完成"
else
    echo "VAD 模型已存在，跳過"
fi

# ============================
# ASR 模型 (中文 Streaming)
# ============================
echo ""
echo "=== 下載 ASR 模型 (中文 Streaming Zipformer) ==="
ASR_DIR="$MODELS_DIR/asr"

# Streaming ASR - 中文
ASR_STREAMING_DIR="$ASR_DIR/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23"
if [ ! -d "$ASR_STREAMING_DIR" ]; then
    cd "$ASR_DIR"
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23.tar.bz2
    tar xvf sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23.tar.bz2
    rm sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23.tar.bz2
    echo "ASR Streaming 模型下載完成"
else
    echo "ASR Streaming 模型已存在，跳過"
fi

# Paraformer (離線, 精度較高) - 可選
echo ""
echo "=== 下載 ASR 模型 (Paraformer 離線) ==="
ASR_OFFLINE_DIR="$ASR_DIR/sherpa-onnx-paraformer-zh-2023-09-14"
if [ ! -d "$ASR_OFFLINE_DIR" ]; then
    cd "$ASR_DIR"
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
    tar xvf sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
    rm sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
    echo "ASR Paraformer 模型下載完成"
else
    echo "ASR Paraformer 模型已存在，跳過"
fi

# ============================
# TTS 模型 (中文 VITS)
# ============================
echo ""
echo "=== 下載 TTS 模型 (中文 VITS) ==="
TTS_DIR="$MODELS_DIR/tts"

# 中文 TTS - VITS aishell3
TTS_ZH_DIR="$TTS_DIR/vits-zh-aishell3"
if [ ! -d "$TTS_ZH_DIR" ]; then
    cd "$TTS_DIR"
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-icefall-zh-aishell3.tar.bz2
    tar xvf vits-icefall-zh-aishell3.tar.bz2
    mv vits-icefall-zh-aishell3 vits-zh-aishell3
    rm vits-icefall-zh-aishell3.tar.bz2
    echo "TTS 中文模型下載完成"
else
    echo "TTS 中文模型已存在，跳過"
fi

# ============================
# KWS 模型 (喚醒詞)
# ============================
echo ""
echo "=== 下載 KWS 模型 (喚醒詞偵測) ==="
KWS_DIR="$MODELS_DIR/kws"

# 中文 KWS 模型
KWS_ZH_DIR="$KWS_DIR/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
if [ ! -d "$KWS_ZH_DIR" ]; then
    cd "$KWS_DIR"
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2
    tar xvf sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2
    rm sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2
    echo "KWS 模型下載完成"
else
    echo "KWS 模型已存在，跳過"
fi

# 建立喚醒詞檔案
KEYWORDS_FILE="$KWS_DIR/keywords.txt"
if [ ! -f "$KEYWORDS_FILE" ]; then
    echo "智護車" > "$KEYWORDS_FILE"
    echo "護理車" >> "$KEYWORDS_FILE"
    echo "喚醒詞檔案已建立: $KEYWORDS_FILE"
fi

# ============================
# 顯示下載結果
# ============================
echo ""
echo "=== 模型下載完成 ==="
echo "目錄結構:"
find "$MODELS_DIR" -type d | head -20

echo ""
echo "模型大小:"
du -sh "$MODELS_DIR"/*

echo ""
echo "下一步驟:"
echo "1. 安裝 sherpa-onnx: pip install sherpa-onnx"
echo "2. 執行測試: python -m src.speech.vad"
