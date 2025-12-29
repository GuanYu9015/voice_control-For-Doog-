#!/usr/bin/env python3
"""
NanoLLM 模型下載器

在 NanoLLM 容器內執行此腳本以下載 LLM 模型。
僅包含 JetPack 5.12 MLC 支援的模型。
"""
import sys

# NanoLLM/MLC 支援的模型列表（JetPack 5.12）
# 注意：Qwen2.5、Llama 3.2 等新模型不被 MLC 支援，請使用 llama.cpp
MODELS = [
    # 中文模型
    ("chinese-alpaca-2-7b", "ziqingyang/chinese-alpaca-2-7b", "q4f16_ft", "中文"),
    ("chinese-llama-2-7b", "hfl/chinese-llama-2-7b", "q4f16_ft", "中文"),
    
    # 英文模型
    ("llama-2-7b", "meta-llama/Llama-2-7b-chat-hf", "q4f16_ft", "英文"),
    ("llama-2-13b", "meta-llama/Llama-2-13b-chat-hf", "q4f16_ft", "英文"),
    ("vicuna-7b", "lmsys/vicuna-7b-v1.5", "q4f16_ft", "英文"),
    ("vicuna-13b", "lmsys/vicuna-13b-v1.5", "q4f16_ft", "英文"),
]


def download_model(name: str, model_path: str, quantization: str, lang: str) -> bool:
    """下載單一模型"""
    print(f"\n{'='*60}")
    print(f"下載模型: {name} ({lang})")
    print(f"路徑: {model_path}")
    print(f"量化: {quantization}")
    print('='*60)
    
    try:
        from nano_llm import NanoLLM
        
        model = NanoLLM.from_pretrained(
            model_path,
            quantization=quantization,
            api="mlc"
        )
        
        print(f"✓ 模型 {name} 下載並載入成功")
        return True
        
    except Exception as e:
        print(f"✗ 下載失敗: {e}")
        return False


def list_models():
    """列出可用模型"""
    print("\n" + "="*60)
    print("NanoLLM/MLC 支援的模型（JetPack 5.12）")
    print("="*60)
    print("\n中文模型:")
    for name, path, quant, lang in MODELS:
        if lang == "中文":
            print(f"  {name:25s} -> {path}")
    
    print("\n英文模型:")
    for name, path, quant, lang in MODELS:
        if lang == "英文":
            print(f"  {name:25s} -> {path}")
    
    print("\n" + "-"*60)
    print("注意：Qwen2.5、Llama 3.2 等新模型請使用 llama.cpp")
    print("執行: bash scripts/install_llama_cpp.sh")
    print("-"*60)


def main():
    print("="*60)
    print("NanoLLM 模型下載器 (JetPack 5.12 MLC)")
    print("="*60)
    
    # 顯示幫助
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print("\n用法:")
        print("  python3 download_llm_models.py           # 互動式選擇")
        print("  python3 download_llm_models.py list      # 列出可用模型")
        print("  python3 download_llm_models.py <name>    # 下載指定模型")
        print("  python3 download_llm_models.py all       # 下載所有模型")
        print("\n推薦中文模型:")
        print("  - chinese-alpaca-2-7b (推薦)")
        print("\n對於新模型（Qwen2.5、Llama 3.2、TAIDE 等），請使用 llama.cpp")
        return
    
    # 列出模型
    if len(sys.argv) > 1 and sys.argv[1] == 'list':
        list_models()
        return
    
    # 下載所有模型
    if len(sys.argv) > 1 and sys.argv[1] == 'all':
        results = {}
        for name, path, quant, lang in MODELS:
            results[name] = download_model(name, path, quant, lang)
        
        print("\n" + "="*60)
        print("下載結果:")
        for name, success in results.items():
            status = "✓ 成功" if success else "✗ 失敗"
            print(f"  {name:25s} {status}")
        return
    
    # 下載指定模型
    if len(sys.argv) > 1:
        target = sys.argv[1]
        for name, path, quant, lang in MODELS:
            if name == target:
                success = download_model(name, path, quant, lang)
                sys.exit(0 if success else 1)
        
        print(f"錯誤: 未知的模型 '{target}'")
        list_models()
        sys.exit(1)
    
    # 互動式選擇
    list_models()
    print("\n推薦下載: chinese-alpaca-2-7b (中文最佳)")
    print("\n輸入模型名稱開始下載，或按 Ctrl+C 取消")
    
    try:
        target = input("模型名稱: ").strip()
        if target:
            for name, path, quant, lang in MODELS:
                if name == target:
                    download_model(name, path, quant, lang)
                    return
            print(f"未找到模型: {target}")
    except KeyboardInterrupt:
        print("\n取消下載")


if __name__ == "__main__":
    main()
