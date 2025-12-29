"""
音訊輸出模組

使用 sounddevice 進行音訊播放，供 TTS 輸出使用。
"""
from typing import Optional

import numpy as np
import sounddevice as sd


class AudioOutput:
    """
    音訊輸出類別
    
    支援同步與異步播放，可用於 TTS 語音輸出。
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        channels: int = 1,
        device: Optional[str] = None
    ):
        """
        初始化音訊輸出
        
        Args:
            sample_rate: 取樣率 (Hz)，TTS 輸出通常為 22050Hz
            channels: 通道數
            device: 音訊裝置名稱，None 表示使用預設裝置
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        
        self._is_playing = False
    
    def play(self, audio_data: np.ndarray, blocking: bool = True) -> None:
        """
        播放音訊
        
        Args:
            audio_data: 音訊資料 (numpy array, float32, -1.0 ~ 1.0)
            blocking: 是否等待播放完成
        """
        if self._is_playing and not blocking:
            # 非阻塞模式下，若已在播放則跳過
            return
        
        self._is_playing = True
        
        try:
            sd.play(
                audio_data,
                samplerate=self.sample_rate,
                device=self.device,
                blocking=blocking
            )
        finally:
            if blocking:
                self._is_playing = False
    
    def stop(self) -> None:
        """停止播放"""
        sd.stop()
        self._is_playing = False
    
    def wait(self) -> None:
        """等待播放完成"""
        sd.wait()
        self._is_playing = False
    
    @property
    def is_playing(self) -> bool:
        """是否正在播放"""
        return self._is_playing
    
    @staticmethod
    def list_devices() -> list[dict]:
        """
        列出可用的音訊輸出裝置
        
        Returns:
            裝置資訊列表
        """
        devices = sd.query_devices()
        output_devices = []
        
        for i, device in enumerate(devices):
            if device['max_output_channels'] > 0:
                output_devices.append({
                    'index': i,
                    'name': device['name'],
                    'channels': device['max_output_channels'],
                    'sample_rate': device['default_samplerate']
                })
        
        return output_devices


# 測試用主程式
if __name__ == "__main__":
    print("=== 可用的音訊輸出裝置 ===")
    for device in AudioOutput.list_devices():
        print(f"  [{device['index']}] {device['name']}")
    
    print("\n=== 播放測試音 (440Hz, 1秒) ===")
    
    # 產生 440Hz 正弦波
    sample_rate = 22050
    duration = 1.0
    frequency = 440.0
    
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    output = AudioOutput(sample_rate=sample_rate)
    output.play(audio, blocking=True)
    
    print("播放完成")
