#!/usr/bin/env python3
"""
護理輔助車語音控制主程式

整合 AsyncVoiceControlPipeline 和 NursingCartUI，
提供完整的語音控制介面。

執行方式：
    python -m src.main_nursing_cart
    python -m src.main_nursing_cart --mock-llm  # 測試模式
"""
import argparse
import os
import sys
import threading
import tkinter as tk
from pathlib import Path

# 確保專案根目錄在 Python 路徑中
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """主程式"""
    parser = argparse.ArgumentParser(description='護理輔助車語音控制系統')
    parser.add_argument('--no-wake', action='store_true',
                        help='停用喚醒詞，持續聆聽')
    parser.add_argument('--mock-llm', action='store_true',
                        help='使用模擬 LLM')
    parser.add_argument('--mock-robot', action='store_true',
                        help='使用模擬機器人（不連接 MQTT）')
    parser.add_argument('--no-ui', action='store_true',
                        help='純命令列模式（不顯示 UI）')
    args = parser.parse_args()
    
    # 匯入模組
    from src.async_pipeline import AsyncVoiceControlPipeline
    
    if args.no_ui:
        # 命令列模式
        _run_cli_mode(args)
    else:
        # UI 模式
        _run_ui_mode(args)


def _run_cli_mode(args):
    """命令列模式"""
    from src.async_pipeline import AsyncVoiceControlPipeline
    
    def on_command(cmd):
        print(f"[Command] {cmd}")
    
    pipeline = AsyncVoiceControlPipeline(
        use_wake_word=not args.no_wake,
        use_mock_llm=args.mock_llm,
        use_mock_robot=args.mock_robot,
        on_command=on_command
    )
    
    pipeline.run()


def _run_ui_mode(args):
    """UI 模式"""
    from src.async_pipeline import AsyncVoiceControlPipeline
    from src.ui.nursing_cart_ui import NursingCartUI
    
    # 建立 Tkinter 視窗
    root = tk.Tk()
    
    # 建立 Pipeline（先建立以便 UI 回調使用）
    pipeline = None
    
    # ========================================
    # Pipeline 回調
    # ========================================
    
    def on_command(cmd):
        cmd_type = cmd.get('type', '')
        text = cmd.get('text', '')
        root.after(0, lambda: ui.add_dialog(f"{cmd_type}: {text}"))
        
        # 更新軌跡顯示：使用已修正的 route（在 async_pipeline 中已經修正）
        route = cmd.get('route', '')
        if route:
            root.after(0, lambda r=route: ui.set_selected_stations(r))
    
    def on_status(status, msg):
        # 更新喚醒狀態指示燈
        if status == 'wake':
            root.after(0, lambda: ui.update_wake_status(True, msg))
        elif status in ['idle', 'processing']:
            root.after(0, lambda: ui.update_wake_status(False))
    
    def on_position(code, name):
        root.after(0, lambda: ui.update_position(code, name))
    
    # 建立 Pipeline
    pipeline = AsyncVoiceControlPipeline(
        use_wake_word=not args.no_wake,
        use_mock_llm=args.mock_llm,
        use_mock_robot=args.mock_robot,
        on_command=on_command,
        on_status=on_status,
        on_position=on_position
    )
    
    # ========================================
    # UI 回調
    # ========================================
    
    def on_execute_trace(route_code):
        """執行記憶軌跡"""
        if pipeline.thouzer and len(route_code) == 2:
            route_number = pipeline.thouzer.station_mapper.get_route_number(
                route_code[0], route_code[1]
            )
            if route_number and pipeline.thouzer.mqtt:
                pipeline.thouzer.mqtt.execute_route(route_number)
                # 更新位置
                pipeline.thouzer.update_position(route_code[1])
    
    def on_create_trace(route_code):
        """建置移動軌跡"""
        if pipeline.thouzer and len(route_code) == 2:
            route_number = pipeline.thouzer.station_mapper.get_route_number(
                route_code[0], route_code[1]
            )
            if route_number and pipeline.thouzer.mqtt:
                pipeline.thouzer.mqtt.start_memory_mode(route_number)
    
    def on_memory_save():
        """記憶存檔"""
        if pipeline.thouzer and pipeline.thouzer.mqtt:
            pipeline.thouzer.mqtt.memory_save()
    
    def on_pause_toggle(is_paused):
        """暫停/繼續"""
        if pipeline.thouzer:
            if is_paused:
                pipeline.thouzer.pause()
            else:
                pipeline.thouzer.resume()
    
    def on_follow_toggle(is_following):
        """跟隨模式"""
        if pipeline.thouzer:
            if is_following:
                pipeline.thouzer.start_follow()
            else:
                pipeline.thouzer.stop_follow()
    
    def on_hold():
        """Hold 點"""
        if pipeline.thouzer and pipeline.thouzer.mqtt:
            pipeline.thouzer.mqtt.set_hold()
    
    def on_close():
        """關閉程式"""
        pipeline.stop()
        if pipeline.thouzer:
            pipeline.thouzer.disconnect()
        root.destroy()
    
    # 建立 UI
    ui = NursingCartUI(
        root,
        on_execute_trace=on_execute_trace,
        on_create_trace=on_create_trace,
        on_memory_save=on_memory_save,
        on_pause_toggle=on_pause_toggle,
        on_follow_toggle=on_follow_toggle,
        on_hold=on_hold,
        on_close=on_close
    )
    
    # 設定視窗關閉處理
    root.protocol("WM_DELETE_WINDOW", on_close)
    
    # 在背景執行 Pipeline
    def run_pipeline():
        pipeline.start()
        root.after(0, lambda: ui.update_voice_status(True))  # 語音：已連接
        root.after(0, lambda: ui.add_dialog("語音控制系統已啟動"))
        
        # 更新 MQTT 狀態
        if pipeline.thouzer:
            mqtt_connected = pipeline.thouzer.is_connected
            root.after(0, lambda: ui.update_mqtt_status(mqtt_connected))
            root.after(0, lambda: ui.update_position(
                pipeline.thouzer.current_position,
                pipeline.thouzer.current_position_name
            ))
        else:
            root.after(0, lambda: ui.update_mqtt_status(False))
    
    pipeline_thread = threading.Thread(target=run_pipeline, daemon=True)
    pipeline_thread.start()
    
    # 定期更新狀態
    def update_status():
        # MQTT 狀態
        if pipeline.thouzer:
            ui.update_mqtt_status(pipeline.thouzer.is_connected)
        
        root.after(500, update_status)
    
    root.after(1000, update_status)
    
    # 執行主迴圈
    root.mainloop()


if __name__ == "__main__":
    main()
