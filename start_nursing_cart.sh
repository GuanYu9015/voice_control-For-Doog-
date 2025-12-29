#!/bin/bash
# 啟動護理車語音控制系統，並將所有輸出紀錄到 log 檔案

# 建立 logs 目錄
mkdir -p logs

# 設定 log 檔名（包含時間戳記，避免覆蓋）
LOG_FILE="logs/nursing_cart_$(date +%Y%m%d_%H%M%S).log"
LATEST_LOG="logs/nursing_cart.log"

echo "==================================================" | tee -a "$LOG_FILE"
echo "Starting Nursing Cart Voice Control System" | tee -a "$LOG_FILE"
echo "Time: $(date)" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"

# 更新 latest log link
rm -f "$LATEST_LOG"
ln -s "$(basename "$LOG_FILE")" "$LATEST_LOG"

# 啟動程式 (python -u disable buffering)
# 2>&1 將 stderr 導向 stdout，再透過 tee 同時輸出到螢幕和檔案
python -u -m src.main_nursing_cart "$@" 2>&1 | tee -a "$LOG_FILE"
