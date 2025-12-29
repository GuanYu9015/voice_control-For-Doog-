# Voice Control Robot

語音控制機器人專案 - Jetson Orin NX + JetPack 5.12

## 功能模組

| 模組 | 技術 | 說明 |
|-----|-----|-----|
| DNS | Sherpa-onnx | 動態噪音抑制 |
| VAD | Sherpa-onnx (Silero) | 語音活動偵測 |
| KWS | Sherpa-onnx | 喚醒詞：智護車/護理車 |
| ASR | Sherpa-onnx | 中文語音辨識 |
| TTS | Sherpa-onnx | 中文語音合成 |
| LLM | NanoLLM | 語言模型推理 |

## 專案結構

```
voice_control/
├── config/           # 設定檔
├── models/           # 模型檔案
├── src/              # 原始碼
│   ├── audio/        # 音訊處理
│   ├── speech/       # 語音處理 (VAD/KWS/ASR/TTS)
│   ├── llm/          # LLM 整合
│   └── robot/        # 機器人控制
├── scripts/          # 安裝腳本
└── tests/            # 測試程式
```

## 快速開始

```bash
# 1. 安裝依賴
pip install -r requirements.txt

# 2. 下載模型
bash scripts/download_models.sh

# 3. 執行測試
python -m pytest tests/

# 4. 啟動語音控制
python -m src.pipeline
```

## 環境需求

- Jetson Orin NX
- JetPack 5.12
- Python 3.8+
- CUDA 11.4
