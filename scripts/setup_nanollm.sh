#!/bin/bash
# 設定 NanoLLM 環境與下載 LLM 模型
# 需要在 Jetson 上執行

set -e

echo "=== NanoLLM 環境設定 ==="

# 檢查是否為 Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "警告: 此腳本設計給 Jetson 平台使用"
fi

# ============================
# 安裝 jetson-containers
# ============================
CONTAINERS_DIR="${HOME}/jetson-containers"

echo ""
echo "=== 設定 jetson-containers ==="

if [ ! -d "$CONTAINERS_DIR" ]; then
    echo "下載 jetson-containers..."
    cd "$HOME"
    git clone https://github.com/dusty-nv/jetson-containers
    cd jetson-containers
    bash install.sh
else
    echo "jetson-containers 已存在: $CONTAINERS_DIR"
    cd "$CONTAINERS_DIR"
    git pull
fi

# ============================
# 下載 LLM 模型
# ============================
echo ""
echo "=== 下載 LLM 模型 ==="
echo "將在 NanoLLM 容器內下載以下模型:"
echo "  1. Qwen/Qwen2.5-3B-Instruct"
echo "  2. Qwen/Qwen2.5-7B-Instruct"
echo "  3. yentinglin/Llama-3.2-TW-3B"
echo "  4. taide/TAIDE-Llama-3-8B"
echo "  5. taide/Gemma-3-TAIDE-12b-Chat"
echo ""
echo "注意: 大型模型下載需要較長時間和足夠的磁碟空間"
echo ""

# 建立下載腳本（在容器內執行）
DOWNLOAD_SCRIPT="/tmp/download_llm_models.py"
cat > "$DOWNLOAD_SCRIPT" << 'EOF'
#!/usr/bin/env python3
"""
在 NanoLLM 容器內下載 LLM 模型
"""
import subprocess
import sys

MODELS = [
    ("qwen2.5-3b", "Qwen/Qwen2.5-3B-Instruct", "q4f16_ft"),
    ("qwen2.5-7b", "Qwen/Qwen2.5-7B-Instruct", "q4f16_ft"),
    ("llama-3.2-tw-3b", "yentinglin/Llama-3.2-TW-3B", "q4f16_ft"),
    ("taide-llama-3-8b", "taide/TAIDE-Llama-3-8B", "q4f16_ft"),
    ("gemma-3-taide-12b", "taide/Gemma-3-TAIDE-12b-Chat", "q4f16_ft"),
]

def download_model(name, model_path, quantization):
    print(f"\n=== 下載模型: {name} ===")
    print(f"路徑: {model_path}")
    print(f"量化: {quantization}")
    
    try:
        # 使用 nano_llm 下載
        from nano_llm import NanoLLM
        
        model = NanoLLM.from_pretrained(
            model_path,
            quantization=quantization,
            api="mlc"
        )
        
        print(f"模型 {name} 下載完成")
        return True
        
    except Exception as e:
        print(f"下載失敗: {e}")
        return False

def main():
    print("=== NanoLLM 模型下載器 ===")
    
    # 檢查是否指定特定模型
    if len(sys.argv) > 1:
        target = sys.argv[1]
        for name, path, quant in MODELS:
            if name == target:
                download_model(name, path, quant)
                return
        print(f"未知的模型: {target}")
        return
    
    # 下載所有模型
    for name, path, quant in MODELS:
        download_model(name, path, quant)
    
    print("\n=== 所有模型下載完成 ===")

if __name__ == "__main__":
    main()
EOF

echo "要下載模型，請執行以下指令進入 NanoLLM 容器:"
echo ""
echo "  cd $CONTAINERS_DIR"
echo "  jetson-containers run \$(autotag nano_llm)"
echo ""
echo "在容器內執行:"
echo "  python3 /workspace/voice_control/scripts/download_llm_models.py"
echo ""
echo "或下載特定模型:"
echo "  python3 /workspace/voice_control/scripts/download_llm_models.py qwen2.5-3b"
echo ""

# ============================
# 建立啟動腳本
# ============================
VOICE_CONTROL_DIR="/home/jetson/voice_control"
RUN_SCRIPT="$VOICE_CONTROL_DIR/scripts/run_in_container.sh"

cat > "$RUN_SCRIPT" << 'EOF'
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
EOF

chmod +x "$RUN_SCRIPT"

echo "啟動腳本已建立: $RUN_SCRIPT"
echo ""
echo "=== 設定完成 ==="
