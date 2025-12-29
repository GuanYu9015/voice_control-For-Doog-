#!/bin/bash
# 安裝 llama.cpp 並下載 GGUF 模型
# 適用於 Jetson 平台
# 根據 config/model_config.yaml 下載所有可用模型

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="${1:-$PROJECT_DIR/models/llm}"

echo "============================================================"
echo "llama.cpp 安裝與 GGUF 模型下載"
echo "============================================================"
echo "專案目錄: $PROJECT_DIR"
echo "模型目錄: $MODELS_DIR"

# ============================
# 安裝 llama-cpp-python
# ============================
echo ""
echo "=== 安裝 llama-cpp-python (CUDA) ==="

# 設定 CUDA 環境
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 檢查是否已安裝
if python3 -c "import llama_cpp" 2>/dev/null; then
    echo "llama-cpp-python 已安裝"
else
    echo "安裝 llama-cpp-python with CUDA 支援..."
    CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --upgrade
    echo "llama-cpp-python 安裝完成"
fi

# ============================
# 建立模型目錄
# ============================
mkdir -p "$MODELS_DIR"

# ============================
# 模型下載函式
# ============================
download_model() {
    local name=$1
    local url=$2
    local filename=$3
    local size=$4
    local target="$MODELS_DIR/$filename"
    
    if [ -f "$target" ]; then
        echo "[✓] $name 已存在"
        return 0
    else
        echo "[↓] 下載 $name ($size)..."
        if wget --progress=bar:force -O "$target" "$url" 2>&1; then
            echo "[✓] $name 下載完成"
            return 0
        else
            echo "[✗] $name 下載失敗"
            rm -f "$target"
            return 1
        fi
    fi
}

# ============================
# 可用模型列表
# ============================
echo ""
echo "============================================================"
echo "可用模型列表 (GGUF Q4_K_M 量化)"
echo "============================================================"
echo ""
echo "中文模型:"
echo "  1. qwen2.5-3b          - Qwen2.5-3B-Instruct     (~2.0GB) [推薦]"
echo "  2. qwen2.5-7b          - Qwen2.5-7B-Instruct     (~4.5GB)"
echo "  3. llama-3.2-tw-3b     - 台灣中文 Llama 3.2      (~2.0GB) [台灣中文]"
echo "  4. taide-llama-3-8b    - TAIDE Llama 3 8B       (~5.0GB) [台灣中文]"
echo "  5. taide-llama-3.1-8b  - TAIDE Llama 3.1 8B     (~5.0GB) [台灣中文]"
echo ""
echo "英文/多語言模型:"
echo "  6. llama-3.2-3b        - Llama 3.2 3B           (~2.0GB)"
echo ""
echo "============================================================"

# ============================
# 選擇下載模式
# ============================
echo ""
echo "下載選項:"
echo "  1) 下載推薦模型 (qwen2.5-3b)"
echo "  2) 下載台灣中文模型 (llama-3.2-tw-3b)"
echo "  3) 下載所有小型模型 (3B 系列，約 6GB)"
echo "  4) 下載所有模型 (約 20GB)"
echo "  5) 自訂選擇"
echo "  6) 只安裝 llama-cpp-python（不下載模型）"
echo ""

read -p "請選擇 [1-6] (預設 1): " choice
choice=${choice:-1}

case $choice in
    1)
        echo ""
        echo "=== 下載推薦模型 ==="
        download_model "Qwen2.5-3B-Instruct" \
            "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf" \
            "qwen2.5-3b-instruct-q4_k_m.gguf" \
            "2.0GB"
        ;;
    2)
        echo ""
        echo "=== 下載台灣中文模型 ==="
        download_model "Llama-3.2-Taiwan-3B" \
            "https://huggingface.co/QuantFactory/Llama-3.2-Taiwan-3B-Instruct-GGUF/resolve/main/Llama-3.2-Taiwan-3B-Instruct.Q4_K_M.gguf" \
            "Llama-3.2-Taiwan-3B-Instruct.Q4_K_M.gguf" \
            "2.0GB"
        ;;
    3)
        echo ""
        echo "=== 下載所有小型模型 (3B) ==="
        download_model "Qwen2.5-3B-Instruct" \
            "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf" \
            "qwen2.5-3b-instruct-q4_k_m.gguf" \
            "2.0GB"
        
        download_model "Llama-3.2-Taiwan-3B" \
            "https://huggingface.co/QuantFactory/Llama-3.2-Taiwan-3B-Instruct-GGUF/resolve/main/Llama-3.2-Taiwan-3B-Instruct.Q4_K_M.gguf" \
            "Llama-3.2-Taiwan-3B-Instruct.Q4_K_M.gguf" \
            "2.0GB"
        
        download_model "Llama-3.2-3B-Instruct" \
            "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf" \
            "llama-3.2-3b-instruct-q4_k_m.gguf" \
            "2.0GB"
        ;;
    4)
        echo ""
        echo "=== 下載所有模型 ==="
        # 3B 模型
        download_model "Qwen2.5-3B-Instruct" \
            "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf" \
            "qwen2.5-3b-instruct-q4_k_m.gguf" \
            "2.0GB"
        
        download_model "Llama-3.2-Taiwan-3B" \
            "https://huggingface.co/QuantFactory/Llama-3.2-Taiwan-3B-Instruct-GGUF/resolve/main/Llama-3.2-Taiwan-3B-Instruct.Q4_K_M.gguf" \
            "Llama-3.2-Taiwan-3B-Instruct.Q4_K_M.gguf" \
            "2.0GB"
        
        download_model "Llama-3.2-3B-Instruct" \
            "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf" \
            "llama-3.2-3b-instruct-q4_k_m.gguf" \
            "2.0GB"
        
        # 7B/8B 模型
        download_model "Qwen2.5-7B-Instruct" \
            "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf" \
            "Qwen2.5-7B-Instruct-Q4_K_M.gguf" \
            "4.7GB"
        
        download_model "TAIDE-Llama3-8B" \
            "https://huggingface.co/QuantFactory/Llama3-TAIDE-LX-8B-Chat-Alpha1-GGUF/resolve/main/Llama3-TAIDE-LX-8B-Chat-Alpha1.Q4_K_M.gguf" \
            "Llama3-TAIDE-LX-8B-Chat-Alpha1-Q4_K_M.gguf" \
            "5.0GB"
        
        download_model "TAIDE-Llama3.1-8B" \
            "https://huggingface.co/tetf/Llama-3.1-TAIDE-LX-8B-Chat-GGUF/resolve/main/Llama-3.1-TAIDE-LX-8B-Chat-Q4_K_M.gguf" \
            "Llama-3.1-TAIDE-LX-8B-Chat-Q4_K_M.gguf" \
            "5.0GB"
        ;;
    5)
        echo ""
        echo "=== 自訂選擇 ==="
        echo "請手動執行以下指令下載模型："
        echo ""
        echo "cd $MODELS_DIR"
        echo ""
        echo "# Qwen2.5-3B (推薦)"
        echo "wget https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
        echo ""
        echo "# 台灣中文 Llama 3.2"
        echo "wget https://huggingface.co/QuantFactory/Llama-3.2-Taiwan-3B-Instruct-GGUF/resolve/main/Llama-3.2-Taiwan-3B-Instruct.Q4_K_M.gguf"
        echo ""
        echo "# TAIDE 8B"
        echo "wget https://huggingface.co/QuantFactory/Llama3-TAIDE-LX-8B-Chat-Alpha1-GGUF/resolve/main/Llama3-TAIDE-LX-8B-Chat-Alpha1.Q4_K_M.gguf"
        ;;
    6)
        echo ""
        echo "只安裝 llama-cpp-python，不下載模型"
        ;;
    *)
        echo "無效選項"
        exit 1
        ;;
esac

# ============================
# 顯示結果
# ============================
echo ""
echo "============================================================"
echo "安裝完成"
echo "============================================================"
echo "模型目錄: $MODELS_DIR"
echo ""
echo "已下載的模型:"
ls -lh "$MODELS_DIR"/*.gguf 2>/dev/null || echo "  (無)"

echo ""
echo "使用方式:"
echo "  # 修改 config/model_config.yaml 中的 llm.llama_cpp.model_path"
echo "  # 然後執行："
echo "  python -m src.pipeline"
echo ""
echo "或在 Python 中："
echo "  from src.llm.unified import create_llm"
echo "  llm = create_llm(backend='llama_cpp', model_path='$MODELS_DIR/<model>.gguf')"
echo ""
