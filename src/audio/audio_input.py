"""
音訊輸入模組

使用 sounddevice 從藍牙麥克風擷取即時音訊串流。
支援 callback 機制將音訊 chunks 傳遞給下游處理模組。
"""
import queue
import threading
from typing import Callable, Optional

import numpy as np
import sounddevice as sd


class AudioInput:
    """
    麥克風音訊輸入類別
    
    為何使用 sounddevice：
    - 跨平台支援良好
    - 低延遲 callback 機制
    - 支援藍牙音訊裝置
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_ms: int = 100,
        device: Optional[str] = None
    ):
        """
        初始化音訊輸入
        
        Args:
            sample_rate: 取樣率 (Hz)，ASR 通常需要 16kHz
            channels: 通道數
            chunk_ms: 每個 chunk 的時間長度 (毫秒)
            device: 音訊裝置名稱，None 表示使用預設裝置
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_ms = chunk_ms
        self.device = device
        
        # 計算每個 chunk 的樣本數
        self.chunk_samples = int(sample_rate * chunk_ms / 1000)
        
        # 音訊緩衝佇列
        self._audio_queue: queue.Queue = queue.Queue()
        
        # 串流物件
        self._stream: Optional[sd.InputStream] = None
        self._is_running = False
        
        # Callback 列表
        self._callbacks: list[Callable[[np.ndarray], None]] = []
    
    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags
    ) -> None:
        """
        音訊串流 callback
        
        潛在風險：若處理速度跟不上音訊輸入，佇列可能會堆積
        """
        if status:
            # 記錄音訊錯誤狀態（如 overflow）
            print(f"Audio input status: {status}")
        
        # 複製音訊資料（避免參照問題）
        audio_chunk = indata.copy().flatten()
        
        # 放入佇列
        self._audio_queue.put(audio_chunk)
        
        # 呼叫已註冊的 callbacks
        for callback in self._callbacks:
            try:
                callback(audio_chunk)
            except Exception as e:
                print(f"Audio callback error: {e}")
    
    def add_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """
        註冊音訊處理 callback
        
        Args:
            callback: 接收 numpy array 的函式
        """
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """移除已註冊的 callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def start(self) -> None:
        """開始音訊串流"""
        if self._is_running:
            return
        
        self._stream = sd.InputStream(
            device=self.device,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
            blocksize=self.chunk_samples,
            callback=self._audio_callback
        )
        
        self._stream.start()
        self._is_running = True
        print(f"Audio input started: {self.sample_rate}Hz, {self.channels}ch")
    
    def stop(self) -> None:
        """停止音訊串流"""
        if not self._is_running:
            return
        
        self._is_running = False
        
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        
        print("Audio input stopped")
    
    def get_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        從佇列取得音訊 chunk
        
        Args:
            timeout: 等待超時時間 (秒)
        
        Returns:
            音訊資料 (numpy array) 或 None (若超時)
        """
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def clear_queue(self) -> None:
        """清空音訊佇列"""
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
    
    @staticmethod
    def list_devices() -> list[dict]:
        """
        列出可用的音訊輸入裝置
        
        Returns:
            裝置資訊列表
        """
        devices = sd.query_devices()
        input_devices = []
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append({
                    'index': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'sample_rate': device['default_samplerate']
                })
        
        return input_devices
    
    def __enter__(self):
        """Context manager 進入"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 離開"""
        self.stop()


# 測試用主程式
if __name__ == "__main__":
    print("=== 可用的音訊輸入裝置 ===")
    for device in AudioInput.list_devices():
        print(f"  [{device['index']}] {device['name']}")
    
    print("\n=== 開始錄音測試 (5秒) ===")
    
    recorded_chunks = []
    
    def save_chunk(chunk: np.ndarray):
        recorded_chunks.append(chunk)
    
    with AudioInput(sample_rate=16000, chunk_ms=100) as audio:
        audio.add_callback(save_chunk)
        
        import time
        time.sleep(5)
    
    # 合併所有 chunks
    if recorded_chunks:
        audio_data = np.concatenate(recorded_chunks)
        print(f"錄製完成: {len(audio_data)} 樣本, {len(audio_data)/16000:.2f} 秒")
