# 護理車語音控制系統 / Nursing Cart Voice Control System

> 基於 Jetson Orin NX 的醫院護理車智能語音控制系統  
> 支援喚醒詞、中文語音辨識、LLM 智能導航與 MQTT 導航控制

---

## 📋 專案簡介

本系統為醫院護理輔助車（Thouzer 載具）設計的語音控制介面，整合語音辨識、大型語言模型（LLM）與 MQTT 導航控制，實現自然語言互動的病房導航功能。

### 核心功能

- **語音喚醒**：支援「智護車」、「護理車」等自定義喚醒詞
- **自然語言導航**：「到五號病房」、「回護理站」等口語化指令
- **跟隨模式**：「開始跟隨」、「停止跟隨」
- **智能路徑規劃**：自動計算起終點路徑，支援 17 個病房 + 護理站導航
- **圖形化介面**：Tkinter UI，顯示樓層地圖、站點選擇、語音對話

### 技術架構

```
喚醒詞偵測 (KWS) → 語音辨識 (ASR) → 大型語言模型 (LLM) → 指令解析 → MQTT 控制 Thouzer 載具
        ↓
    DeepFilterNet 降噪 + Silero VAD 語音活動偵測
```

---

## 🏗️ 系統架構

### 技術棧

| 模組 | 技術方案 | 說明 |
|------|----------|------|
| **DNS** | DeepFilterNet3 | 深度學習噪音抑制（即時串流） |
| **VAD** | Silero VAD (Sherpa-ONNX) | 語音活動偵測 |
| **KWS** | Sherpa-ONNX Zipformer | 喚醒詞偵測（開放詞彙） |
| **ASR** | Sherpa-ONNX Streaming Zipformer | 中文語音辨識（串流） |
| **TTS** | Sherpa-ONNX VITS (AISHELL-3) | 中文語音合成 |
| **LLM** | llama.cpp (Qwen 2.5-7B) | 意圖理解與指令生成 |
| **導航** | MQTT + Thouzer | 載具軌跡執行 |

### 專案結構

```
voice_control/
├── config/
│   ├── model_config.yaml      # 模型路徑設定
│   └── robot_config.yaml      # 載具/站點/MQTT 設定（Single Source of Truth）
├── models/                     # AI 模型檔案
│   ├── dns/                    # DeepFilterNet
│   ├── vad/                    # Silero VAD
│   ├── asr/                    # Streaming Zipformer (中文)
│   ├── tts/                    # VITS 合成器
│   └── llm/                    # LLM 模型 (GGUF)
├── src/
│   ├── async_pipeline.py       # 異步語音控制 Pipeline（核心）
│   ├── main_nursing_cart.py    # 護理車應用程式入口
│   ├── audio/                  # 音訊輸入輸出
│   ├── speech/                 # VAD/KWS/ASR/TTS 模組
│   ├── llm/                    # LLM 推理介面
│   ├── robot/                  # 載具控制與站點映射
│   │   ├── thouzer.py          # Thouzer 載具控制器
│   │   ├── mqtt_controller.py  # MQTT 協議實作
│   │   ├── station_mapper.py   # 站點/路徑映射
│   │   └── nursing_commands.py # LLM Prompt 與指令解析
│   ├── ui/                     # Tkinter 圖形介面
│   └── utils/                  # 工具函式（簡繁轉換等）
├── scripts/                    # 安裝腳本
├── logs/                       # 日誌輸出
└── start_nursing_cart.sh       # 啟動腳本（含日誌記錄）
```

---

## 🚀 快速開始

### 環境需求

- **硬體**：NVIDIA Jetson Orin NX (8GB+)
- **作業系統**：JetPack 5.1.2 (Ubuntu 20.04)
- **Python**：3.10+
- **CUDA**：11.4+ (JetPack 自帶)

### 安裝步驟

#### 1. 安裝 PyTorch（Jetson ARM64）

```bash
# JetPack 5.1.2 適用
pip3 install torch torchvision torchaudio --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/
```

#### 2. 安裝 llama.cpp（CUDA 支援）

```bash
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python
```

#### 3. 安裝其他依賴

```bash
cd /home/jetson/voice_control
pip install -r requirements.txt
```

#### 4. 下載 AI 模型

```bash
# 下載所有必要模型（VAD/ASR/TTS）
bash scripts/download_models.sh

# 下載 LLM 模型（Qwen 2.5-7B，約 4.7GB）
python scripts/download_llm_models.py
```

#### 5. 設定 MQTT 連線

編輯 `config/robot_config.yaml` 中的 MQTT broker 位址：

```yaml
mqtt:
  broker:
    host: "192.168.212.1"  # 修改為實際 IP
    port: 1883
  auth:
    username: "mqtt"
    password: "your_password"  # 或使用環境變數 MQTT_PASSWORD
```

---

## 🎮 使用方式

### 啟動系統

```bash
# 啟動護理車語音控制（含 UI）
./start_nursing_cart.sh

# 使用 Mock Robot（無需實際連線 MQTT，用於測試）
./start_nursing_cart.sh --mock-robot

# 查看日誌
tail -f logs/nursing_cart.log
```

### 語音指令範例

| 使用情境 | 語音指令 | 系統行為 |
|----------|----------|----------|
| 喚醒系統 | 「護理車」 | TTS 回應：「我在，請說」 |
| 導航病房 | 「到五號病房」 | 執行路徑 O→E (護理站→5號病房) |
| 連續導航 | 「從七號到十號病房」 | 執行路徑 G→J |
| 一氣呵成 | 「護理車到十七號病房」 | 喚醒+導航 (O→R) |
| 返回護理站 | 「回護理站」 | 執行路徑 X→O |
| 跟隨模式 | 「開始跟隨」 | 啟動跟隨模式 |
| 停止跟隨 | 「停止跟隨」 | 停止跟隨 |

### UI 操作

- **地圖面板**：點選起點→終點，執行建圖軌跡
- **狀態列**：顯示喚醒狀態、語音連接、MQTT 連線、當前位置
- **語音對話**：即時顯示 LLM 回應與指令執行結果

---

## ⚙️ 系統配置

### 站點對照表

本系統預設配置 17 個病房 + 1 個護理站（來源：`config/robot_config.yaml`）

| 代號 | 病房號碼 | 名稱 | 代號 | 病房號碼 | 名稱 |
|------|----------|------|------|----------|------|
| O | - | 護理站 | J | 3210 | 10號病房 |
| A | 3201 | 1號病房 | K | 3211 | 11號病房 |
| B | 3202 | 2號病房 | L | 3212 | 12號病房 |
| C | 3203 | 3號病房 | M | 3213 | 13號病房 |
| D | 3204 | 4號病房 | N | 3214 | 14號病房 |
| E | 3205 | 5號病房 | P | 3215 | 15號病房 |
| F | 3206 | 6號病房 | Q | 3216 | 16號病房 |
| G | 3207 | 7號病房 | R | 3217 | 17號病房 |
| H | 3208 | 8號病房 | - | - | - |
| I | 3209 | 9號病房 | - | - | - |

> 注意：英文代號跳過 O（護理站專用），所以 15號=P, 16號=Q, 17號=R

### LLM Prompt Engineering

本系統使用 **Chain of Thought (CoT)** 技術提升 LLM 準確度：

1. **垂直映射表**：`1號 -> A` 格式（而非 `A=1號`），降低 LLM 位移錯誤
2. **Reasoning 欄位**：強制 LLM 先輸出推理邏輯（`"reasoning":"Rule: 5號=E. Target is E."`）
3. **負向約束**：明確禁止按字母順序猜測（`DO NOT calculate letter positions`）
4. **Route 自動修正**：程式根據 `current_position + target` 自動計算正確 route

完整 Prompt 定義：[`src/robot/nursing_commands.py`](src/robot/nursing_commands.py#L121)

### ASR 誤辨識修正

常見口語誤辨識自動修正（`src/async_pipeline.py`）：

| 誤辨識 | 修正為 |
|--------|--------|
| 跟水、根水、跟誰、根隨 | 跟隨 |

---

## 🔧 進階配置

### 調整 LLM 模型

編輯 `config/model_config.yaml`：

```yaml
llm:
  llama_cpp:
    model_path: models/llm/Qwen2.5-7B-Instruct-Q4_K_M.gguf  # 替換模型
    n_ctx: 8192        # Context Window（避免 token 溢出）
    n_gpu_layers: -1   # GPU 層數（-1 = 全部）
```

推薦模型：
- **Qwen 2.5-3B**：速度快，適合即時回應（2GB）
- **Qwen 2.5-7B**：準確度高，推薦使用（4.7GB）

### 調整 TTS 語速

在 `src/async_pipeline.py` 第 200 行：

```python
self.tts = create_tts(model_dir=str(tts_dirs[0]), speed=0.8)  # 0.5-2.0
```

### 喚醒詞變體

在 `src/speech/kws.py` 新增諧音變體以提高匹配率：

```python
SIMPLIFIED_VARIANTS = {
    "智護車": ["智护车", "自護車", "之護車", "志護車", ...],
    "護理車": ["护理车", "戶理車", "胡理车", ...],
}
```

---

## 📊 效能指標

| 指標 | 數值 |
|------|------|
| 喚醒詞偵測延遲 | ~200ms |
| ASR 辨識延遲 | ~500ms (串流) |
| LLM 推理延遲 | ~1.5s (Qwen 2.5-7B) |
| TTS 合成延遲 | ~300ms |
| 端到端延遲 | ~2.5s (喚醒→執行) |
| GPU 記憶體使用 | ~5.5GB |

測試環境：Jetson Orin NX (8GB), JetPack 5.1.2

---

## 🐛 疑難排解

### 1. PyTorch CUDA 不可用

```bash
# 檢查 CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# 重新安裝 Jetson PyTorch
pip3 install --no-cache-dir torch torchvision torchaudio --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/
```

### 2. llama.cpp 無 CUDA 加速

```bash
# 重新編譯 llama-cpp-python
pip uninstall llama-cpp-python
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --no-cache-dir
```

### 3. TTS 發音錯誤（OOV）

系統已內建數字轉中文機制（`10號` → `十號`），若仍有問題，可在 `src/async_pipeline.py` 的 `_convert_numbers_to_chinese()` 新增對照表。

### 4. MQTT 連線失敗

```bash
# 測試 MQTT broker 連線
mosquitto_sub -h 192.168.212.1 -p 1883 -u mqtt -P <password> -t '#'
```

---

## 📝 授權與貢獻

本專案為內部研究專案，如需引用或修改請聯繫專案維護者。

### 核心技術貢獻

- **Sherpa-ONNX**：語音處理框架
- **llama.cpp**：LLM 推理引擎
- **DeepFilterNet**：降噪技術
- **Qwen Team**：LLM 模型

---

## 資訊
 
最後更新：2025-12-29
