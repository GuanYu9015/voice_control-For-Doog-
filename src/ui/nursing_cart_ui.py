"""
護理輔助車 UI 模組

基於 main_2.7_t6.py 改造的 Tkinter 介面，
整合 AsyncVoiceControlPipeline 實現語音控制。

功能：
- 樓層地圖顯示
- 站點選擇
- 建圖功能（執行軌跡、建置軌跡、記憶存檔）
- 語音/MQTT 狀態顯示
- LLM 對話顯示
"""
import os
import threading
import time
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from typing import Callable, Optional
import yaml


class NursingCartUI:
    """
    護理輔助車 UI
    
    整合語音控制 Pipeline 的圖形介面。
    """
    
    def __init__(
        self,
        root: tk.Tk,
        config_path: str = "config/robot_config.yaml",
        on_station_select: Optional[Callable[[str], None]] = None,
        on_execute_trace: Optional[Callable[[str], None]] = None,
        on_create_trace: Optional[Callable[[str], None]] = None,
        on_memory_save: Optional[Callable[[], None]] = None,
        on_pause_toggle: Optional[Callable[[bool], None]] = None,
        on_follow_toggle: Optional[Callable[[bool], None]] = None,
        on_hold: Optional[Callable[[], None]] = None,
        on_close: Optional[Callable[[], None]] = None
    ):
        """
        初始化 UI
        
        Args:
            root: Tkinter 根視窗
            config_path: 設定檔路徑
            on_station_select: 站點選擇回調
            on_execute_trace: 執行軌跡回調 (route_code)
            on_create_trace: 建置軌跡回調 (route_code)
            on_memory_save: 記憶存檔回調
            on_pause_toggle: 暫停切換回調 (is_paused)
            on_follow_toggle: 跟隨切換回調 (is_following)
            on_hold: Hold 點回調
            on_close: 關閉程式回調
        """
        self.root = root
        self.on_station_select = on_station_select
        self.on_execute_trace = on_execute_trace
        self.on_create_trace = on_create_trace
        self.on_memory_save = on_memory_save
        self.on_pause_toggle = on_pause_toggle
        self.on_follow_toggle = on_follow_toggle
        self.on_hold = on_hold
        self.on_close = on_close
        
        # 載入設定
        self.config = self._load_config(config_path)
        self.ui_config = self.config.get('ui', {})
        self.stations = self.config.get('stations', {})
        
        # 狀態
        self.selected_stations = ""
        self.current_position = "O"
        self.is_following = False
        self.is_paused = False
        self.is_listening = False
        self.mqtt_connected = False
        
        # 圖片資源
        self.images = {}
        
        # 建立 UI
        self._setup_window()
        self._create_ui()
    
    def _load_config(self, path: str) -> dict:
        """載入設定檔"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Failed to load config: {e}")
            return {}
    
    def _setup_window(self) -> None:
        """設定視窗"""
        title = self.ui_config.get('title', '護理輔助車語音控制')
        self.root.title(title)
        
        # 使用 800x600 與原始 main_2.7_t6.py 一致
        window = self.ui_config.get('window', {})
        width = window.get('width', 800)
        height = window.get('height', 600)
        self.root.geometry(f'{width}x{height}')
        
        if window.get('fullscreen', False):
            self.root.attributes('-fullscreen', True)
        
        self.root.resizable(False, False)
    
    def _create_ui(self) -> None:
        """建立 UI 元件"""
        # 底部：狀態列（先建立確保在最底部）
        self._create_status_bar()
        
        # 主框架 - 使用 grid 佈局
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(side='top', fill='both', expand=True)
        
        # 設定 grid 權重
        self.main_frame.columnconfigure(0, weight=3)  # 左側地圖佔更多空間
        self.main_frame.columnconfigure(1, weight=1)  # 右側控制面板
        self.main_frame.rowconfigure(0, weight=1)
        
        # 左側：樓層地圖
        self._create_floor_panel()
        
        # 右側：控制面板（建圖功能）
        self._create_control_panel()
    
    def _create_floor_panel(self) -> None:
        """建立樓層地圖面板"""
        frame = tk.LabelFrame(self.main_frame)
        frame.grid(row=0, column=0, padx=12, pady=10, sticky='nsew', rowspan=3)
        
        # 提示文字
        tk.Label(
            frame,
            text="* 請依序點選起點、終點",
            font=('Microsoft JhengHei', 14, 'bold')
        ).pack(padx=5, pady=3, anchor='w')
        
        # 地圖圖片
        img_path = self.ui_config.get('resources', {}).get('images', '/home/jetson/Documents/img_mmh')
        floor_img_path = os.path.join(img_path, 'mfloor_plan_3f.png')
        
        if os.path.exists(floor_img_path):
            # 載入並放大地圖（zoom 參數：1.2 倍）
            floor_img_original = tk.PhotoImage(file=floor_img_path)
            # PhotoImage 只支援整數倍率，使用 zoom(6).subsample(5) 達到 1.2 倍
            self.floor_img = floor_img_original.zoom(10).subsample(9)
            self.floor_label = tk.Label(frame, image=self.floor_img)
            self.floor_label.pack(ipadx=10, anchor='center')
            
            # 建立站點按鈕（放在 frame 上，使用相對位置）
            self._create_station_buttons(frame, img_path)
        else:
            tk.Label(frame, text=f"地圖圖片未找到\n{floor_img_path}", fg='red').pack()
        
        # 軌跡站點顯示（使用 place 放置在固定位置）
        track_frame = tk.Frame(frame)
        track_frame.place(relx=0.2, rely=0.885)
        
        tk.Label(
            track_frame,
            text="軌跡站點:",
            font=('Microsoft JhengHei', 14, 'bold'),
            fg='#000000'
        ).pack(side='left')
        
        self.station_label = tk.Label(
            track_frame,
            text="尚未點選站點",
            font=('Microsoft JhengHei', 14, 'bold'),
            fg='#0000CD'
        )
        self.station_label.pack(side='left', padx=5)
        
        # 清除按鈕
        delete_img_path = os.path.join(img_path, 'delete.png')
        if os.path.exists(delete_img_path):
            self.delete_img = tk.PhotoImage(file=delete_img_path)
            tk.Button(frame, image=self.delete_img, command=self._clear_stations).pack(padx=10, pady=10, side='right')
        else:
            tk.Button(frame, text="清除", command=self._clear_stations).pack(padx=10, pady=10, side='right')
    
    def _create_station_buttons(self, parent: tk.Widget, img_path: str) -> None:
        """建立站點按鈕"""
        for code, station in self.stations.items():
            img_file = os.path.join(img_path, station.get('image', ''))
            if os.path.exists(img_file):
                img = tk.PhotoImage(file=img_file)
                self.images[code] = img
                
                position = station.get('position', [0, 0])
                btn = tk.Button(
                    parent,
                    image=img,
                    command=lambda c=code: self._select_station(c)
                )
                btn.place(relx=position[0], rely=position[1])
    
    def _create_control_panel(self) -> None:
        """建立控制面板（建圖功能）"""
        frame = tk.LabelFrame(self.main_frame)
        frame.grid(row=0, column=1, padx=5, pady=10, sticky='nsew', rowspan=4)
        
        colors = self.ui_config.get('colors', {})
        btn_style = {
            'font': ('Microsoft JhengHei', 10, 'bold'),
            'fg': 'white',
            'width': 12,
            'height': 1
        }
        
        # 1. 執行記憶軌跡
        tk.Button(
            frame,
            text="執行記憶軌跡",
            bg=colors.get('primary', '#388E3C'),
            command=self._execute_trace,
            **btn_style
        ).pack(pady=5)
        
        # 2. 建置移動軌跡
        tk.Button(
            frame,
            text="建置移動軌跡",
            bg=colors.get('secondary', '#FF9800'),
            command=self._create_trace,
            **btn_style
        ).pack(pady=5)
        
        # 3. 記憶存檔
        tk.Button(
            frame,
            text="記憶存檔",
            bg=colors.get('info', '#2196F3'),
            command=self._memory_save,
            **btn_style
        ).pack(pady=5)
        
        # 4. 暫停控制區域
        pause_section = tk.LabelFrame(
            frame,
            text="暫停控制（建置模式）",
            font=('Microsoft JhengHei', 10, 'bold'),
            padx=5,
            pady=5
        )
        pause_section.pack(padx=5, pady=8, fill='x')
        
        tk.Label(
            pause_section,
            text="在當前位置記錄暫停:",
            font=('Microsoft JhengHei', 9),
            fg='#d32f2f'
        ).pack()
        
        pause_btn_frame = tk.Frame(pause_section)
        pause_btn_frame.pack(pady=5)
        
        # 暫停按鈕
        self.pause_btn = tk.Button(
            pause_btn_frame,
            text="暫停",
            font=('Microsoft JhengHei', 10, 'bold'),
            fg='white',
            bg=colors.get('pause', '#9C27B0'),
            width=7,
            height=1,
            command=self._toggle_pause
        )
        self.pause_btn.pack(side='left', padx=3)
        
        # Hold 按鈕
        tk.Button(
            pause_btn_frame,
            text="Hold",
            font=('Microsoft JhengHei', 10, 'bold'),
            fg='white',
            bg=colors.get('hold', '#E91E63') if 'hold' in colors else '#E91E63',
            width=7,
            height=1,
            command=self._set_hold
        ).pack(side='left', padx=3)
        
        # 提示
        tk.Label(
            pause_section,
            text="※暫停後請等待你想要的秒數\n再按「繼續」來記錄暫停時間",
            font=('Microsoft JhengHei', 8),
            fg='#666666',
            justify='left'
        ).pack(pady=2)
        
        # 5. 跟隨模式
        self.follow_btn = tk.Button(
            frame,
            text="啟動跟隨",
            bg=colors.get('success', '#4CAF50'),
            command=self._toggle_follow,
            **btn_style
        )
        self.follow_btn.pack(pady=5)
        
        # 分隔線
        tk.Frame(frame, height=2, bg='#CCCCCC').pack(fill='x', pady=10)
        
        # 語音對話區
        dialog_frame = tk.LabelFrame(frame, text="語音對話")
        dialog_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.dialog_text = tk.Text(
            dialog_frame,
            height=6,
            width=25,
            font=('Microsoft JhengHei', 9),
            state='disabled'
        )
        self.dialog_text.pack(padx=5, pady=5, fill='both', expand=True)
        
        # 關閉程式按鈕
        tk.Button(
            frame,
            text="關閉程式",
            font=('Microsoft JhengHei', 10, 'bold'),
            fg='white',
            bg='#757575',
            width=14,
            height=1,
            command=self._close_app
        ).pack(pady=10)
    
    def _create_status_bar(self) -> None:
        """建立狀態列"""
        self.status_bar = tk.Frame(self.root, bg='#EEEEEE', bd=1, relief='sunken')
        self.status_bar.pack(side='bottom', fill='x')
        
        # 喚醒狀態指示燈
        self.wake_status = tk.Label(
            self.status_bar,
            text="● 未喚醒",
            font=('Microsoft JhengHei', 11, 'bold'),
            fg='#888888',
            bg='#EEEEEE'
        )
        self.wake_status.pack(side='left', padx=10, pady=5)
        
        # 語音連接狀態
        self.voice_status = tk.Label(
            self.status_bar,
            text="語音：未連接",
            font=('Microsoft JhengHei', 11, 'bold'),
            fg='#888888',
            bg='#EEEEEE'
        )
        self.voice_status.pack(side='left', padx=10, pady=5)
        
        # MQTT 連線狀態
        self.mqtt_status = tk.Label(
            self.status_bar,
            text="MQTT：未連線",
            font=('Microsoft JhengHei', 11, 'bold'),
            fg='#FF0000',
            bg='#EEEEEE'
        )
        self.mqtt_status.pack(side='left', padx=10, pady=5)
        
        # 當前位置
        self.position_label = tk.Label(
            self.status_bar,
            text="位置：護理站",
            font=('Microsoft JhengHei', 11, 'bold'),
            fg='#0066CC',
            bg='#EEEEEE'
        )
        self.position_label.pack(side='left', padx=20, pady=5)
    
    # ========================================
    # 事件處理
    # ========================================
    
    def _select_station(self, code: str) -> None:
        """選擇站點"""
        if len(self.selected_stations) < 2 and code not in self.selected_stations:
            self.selected_stations += code
            self.station_label.config(text=self.selected_stations)
            
            if self.on_station_select:
                self.on_station_select(code)
    
    def _clear_stations(self) -> None:
        """清除站點選擇"""
        self.selected_stations = ""
        self.station_label.config(text="尚未點選")
    
    def _execute_trace(self) -> None:
        """執行記憶軌跡"""
        if len(self.selected_stations) != 2:
            messagebox.showwarning("執行記憶軌跡", "請先點選起點和終點（需選擇2個站點）")
            return
        
        if self.on_execute_trace:
            self.on_execute_trace(self.selected_stations)
        
        self.add_dialog(f"執行軌跡：{self.selected_stations}")
    
    def _create_trace(self) -> None:
        """建置移動軌跡"""
        if len(self.selected_stations) != 2:
            messagebox.showwarning("建置移動軌跡", "請先點選起點和終點（需選擇2個站點）")
            return
        
        result = messagebox.askyesno(
            "建置移動軌跡確認",
            f"確定建置軌跡\n站點: {self.selected_stations}"
        )
        
        if result:
            if self.on_create_trace:
                self.on_create_trace(self.selected_stations)
            
            self.add_dialog(f"建置軌跡：{self.selected_stations}")
            messagebox.showinfo(
                "建置移動軌跡",
                "已發送建置指令\n\n"
                "【重要提示】\n"
                "要記錄暫停點：\n"
                "1. 移動到需要暫停的位置\n"
                "2. 按下「暫停」按鈕\n"
                "3. 等待你想要的秒數\n"
                "4. 按下「繼續」按鈕\n"
                "5. 繼續移動到終點\n"
                "6. 完成後按「記憶存檔」"
            )
    
    def _memory_save(self) -> None:
        """記憶存檔"""
        if self.on_memory_save:
            self.on_memory_save()
        
        self.add_dialog("記憶存檔")
        messagebox.showinfo("記憶存檔", "記憶存檔指令已送出")
    
    def _toggle_pause(self) -> None:
        """切換暫停"""
        self.is_paused = not self.is_paused
        colors = self.ui_config.get('colors', {})
        
        if self.is_paused:
            self.pause_btn.config(text="繼續", bg=colors.get('resume', '#673AB7'))
            self.add_dialog("已暫停")
            messagebox.showinfo(
                "暫停",
                "已暫停機器人\n\n"
                "請等待你想要的秒數\n"
                "然後按「繼續」按鈕\n\n"
                "等待的時間會被記錄在軌跡中"
            )
        else:
            self.pause_btn.config(text="暫停", bg=colors.get('pause', '#9C27B0'))
            self.add_dialog("已繼續")
            messagebox.showinfo("繼續移動", "已發送繼續指令\n暫停時間已記錄在軌跡中")
        
        if self.on_pause_toggle:
            self.on_pause_toggle(self.is_paused)
    
    def _set_hold(self) -> None:
        """設定 Hold 點"""
        if self.on_hold:
            self.on_hold()
        
        self.add_dialog("設定 Hold 點")
    
    def _toggle_follow(self) -> None:
        """切換跟隨模式"""
        self.is_following = not self.is_following
        colors = self.ui_config.get('colors', {})
        
        if self.is_following:
            self.follow_btn.config(text="停止跟隨", bg=colors.get('danger', '#F44336'))
            self.add_dialog("跟隨模式已啟動")
        else:
            self.follow_btn.config(text="啟動跟隨", bg=colors.get('success', '#4CAF50'))
            self.add_dialog("跟隨模式已停止")
        
        if self.on_follow_toggle:
            self.on_follow_toggle(self.is_following)
    
    def _close_app(self) -> None:
        """關閉程式"""
        result = messagebox.askyesno("關閉程式", "確定要關閉程式嗎？")
        if result:
            if self.on_close:
                self.on_close()
            else:
                self.root.destroy()
    
    def _get_station_name(self, code: str) -> str:
        """取得站點名稱"""
        return self.stations.get(code, {}).get('name', f'位置{code}')
    
    # ========================================
    # 公開方法（供外部更新 UI）
    # ========================================
    
    def update_voice_status(self, connected: bool) -> None:
        """更新語音連接狀態"""
        if connected:
            self.voice_status.config(text="語音：已連接", fg='#00AA00')
        else:
            self.voice_status.config(text="語音：未連接", fg='#888888')
    
    def update_mqtt_status(self, connected: bool) -> None:
        """更新 MQTT 連線狀態"""
        self.mqtt_connected = connected
        if connected:
            self.mqtt_status.config(text="MQTT：已連線", fg='#00AA00')
        else:
            self.mqtt_status.config(text="MQTT：未連線", fg='#FF0000')
    
    def update_position(self, code: str, name: str) -> None:
        """更新當前位置"""
        self.current_position = code
        self.position_label.config(text=f"位置：{name}")
    
    def add_dialog(self, text: str) -> None:
        """新增對話"""
        timestamp = time.strftime("%H:%M:%S")
        self.dialog_text.config(state='normal')
        self.dialog_text.insert('end', f"[{timestamp}] {text}\n")
        self.dialog_text.see('end')
        self.dialog_text.config(state='disabled')
    
    def set_listening(self, listening: bool) -> None:
        """設定聆聽狀態"""
        self.is_listening = listening
        if listening:
            self.update_voice_status("聆聽中...", '#00AA00')
        else:
            self.update_voice_status("待機中", '#444444')
    
    def update_wake_status(self, woke: bool, keyword: str = "") -> None:
        """更新喚醒狀態指示燈"""
        if woke:
            self.wake_status.config(text="● 已喚醒", fg='#00AA00')
        else:
            self.wake_status.config(text="● 待喚醒", fg='#888888')
    
    def set_selected_stations(self, route: str) -> None:
        """設定已選站點（供語音控制用）"""
        self.selected_stations = route
        self.station_label.config(text=route)


def create_nursing_cart_ui(root: tk.Tk, **kwargs) -> NursingCartUI:
    """建立護理車 UI"""
    return NursingCartUI(root, **kwargs)


# 測試
if __name__ == "__main__":
    root = tk.Tk()
    
    def on_execute(route):
        print(f"Execute: {route}")
    
    def on_create(route):
        print(f"Create: {route}")
    
    ui = create_nursing_cart_ui(
        root,
        on_execute_trace=on_execute,
        on_create_trace=on_create
    )
    
    # 模擬更新
    ui.update_voice_status("已連接", '#00AA00')
    ui.update_mqtt_status(True)
    ui.add_dialog("系統啟動")
    
    root.mainloop()
