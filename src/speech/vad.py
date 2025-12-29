"""
語音活動偵測模組 (VAD - Voice Activity Detection)

使用 Sherpa-onnx 的 Silero VAD 模型偵測語音片段。
主要功能：偵測語音起始與結束，將有效語音片段傳遞給 ASR。

設計說明：
1. 使用 Silero VAD 的原因：
   - 輕量且準確
   - sherpa-onnx 原生支援
   - 適合即時處理
"""
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:
    import sherpa_onnx
    SHERPA_AVAILABLE = True
except ImportError:
    SHERPA_AVAILABLE = False
    print("Warning: sherpa_onnx not installed, VAD will be disabled")


class VoiceActivityDetector:
    """
    語音活動偵測器
    
    使用 Silero VAD 模型偵測語音片段，
    當偵測到完整的語音段落時觸發 callback。
    """
    
    def __init__(
        self,
        model_path: str,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_silence_duration: float = 0.5,
        min_speech_duration: float = 0.25,
        window_size: int = 512
    ):
        """
        初始化 VAD
        
        Args:
            model_path: Silero VAD 模型路徑
            sample_rate: 取樣率，必須為 16000
            threshold: 語音偵測閾值 (0.0 ~ 1.0)
            min_silence_duration: 最小靜音時長 (秒)，超過此時間視為語音結束
            min_speech_duration: 最小語音時長 (秒)，短於此時間的語音會被忽略
            window_size: 處理視窗大小
        """
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_silence_duration = min_silence_duration
        self.min_speech_duration = min_speech_duration
        self.window_size = window_size
        
        self._vad = None
        self._is_initialized = False
        
        # 語音片段回調
        self._on_speech_callbacks: list[Callable[[np.ndarray], None]] = []
        
        self._init_model()
    
    def _init_model(self) -> None:
        """初始化 VAD 模型"""
        if not SHERPA_AVAILABLE:
            print("sherpa_onnx not available, VAD disabled")
            return
        
        if not Path(self.model_path).exists():
            print(f"VAD model not found: {self.model_path}")
            return
        
        try:
            # 建立 Silero VAD 設定
            config = sherpa_onnx.VadModelConfig()
            config.silero_vad.model = self.model_path
            config.silero_vad.threshold = self.threshold
            config.silero_vad.min_silence_duration = self.min_silence_duration
            config.silero_vad.min_speech_duration = self.min_speech_duration
            config.silero_vad.window_size = self.window_size
            config.sample_rate = self.sample_rate
            
            self._vad = sherpa_onnx.VoiceActivityDetector(
                config,
                buffer_size_in_seconds=30
            )
            
            self._is_initialized = True
            print(f"VAD model loaded: {self.model_path}")
            
        except Exception as e:
            print(f"Failed to initialize VAD: {e}")
    
    def process(self, audio_chunk: np.ndarray) -> list[np.ndarray]:
        """
        處理音訊 chunk 並偵測語音片段
        
        Args:
            audio_chunk: 音訊資料 (float32, 16kHz)
        
        Returns:
            偵測到的語音片段列表
        """
        if not self._is_initialized or self._vad is None:
            return []
        
        # 確保格式正確
        audio_chunk = audio_chunk.astype(np.float32)
        
        # 餵入 VAD
        self._vad.accept_waveform(audio_chunk)
        
        # 取得完成的語音片段
        speech_segments = []
        
        while not self._vad.empty():
            segment = self._vad.front()
            speech_segments.append(
                np.array(segment.samples, dtype=np.float32)
            )
            self._vad.pop()
            
            # 觸發回調
            for callback in self._on_speech_callbacks:
                try:
                    callback(speech_segments[-1])
                except Exception as e:
                    print(f"VAD callback error: {e}")
        
        return speech_segments
    
    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        檢查當前 chunk 是否包含語音
        
        Args:
            audio_chunk: 音訊資料
        
        Returns:
            是否為語音
        """
        if not self._is_initialized or self._vad is None:
            return False
        
        return self._vad.is_speech(audio_chunk.astype(np.float32))
    
    def on_speech(self, callback: Callable[[np.ndarray], None]) -> None:
        """
        註冊語音片段偵測回調
        
        Args:
            callback: 當偵測到完整語音片段時呼叫
        """
        self._on_speech_callbacks.append(callback)
    
    def reset(self) -> None:
        """重置 VAD 狀態"""
        if self._vad:
            self._vad.clear()
    
    @property
    def is_available(self) -> bool:
        """VAD 是否可用"""
        return self._is_initialized


def create_vad(
    model_path: Optional[str] = None,
    models_dir: str = "models",
    **kwargs
) -> Optional[VoiceActivityDetector]:
    """
    建立 VAD 實例的工廠函式
    
    Args:
        model_path: 模型路徑，若為 None 則使用預設路徑
        models_dir: 模型目錄
        **kwargs: 傳遞給 VoiceActivityDetector 的其他參數
    
    Returns:
        VAD 實例或 None
    """
    if model_path is None:
        # 使用預設路徑
        model_path = str(Path(models_dir) / "vad" / "silero_vad.onnx")
    
    if not Path(model_path).exists():
        print(f"VAD model not found: {model_path}")
        print("Please download the model first:")
        print("  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx")
        return None
    
    return VoiceActivityDetector(model_path, **kwargs)


# 測試用主程式
if __name__ == "__main__":
    print("=== VAD 模組測試 ===")
    
    if not SHERPA_AVAILABLE:
        print("sherpa_onnx not installed, skipping test")
        exit(1)
    
    # 嘗試建立 VAD
    vad = create_vad(models_dir="/home/jetson/voice_control/models")
    
    if vad and vad.is_available:
        print("VAD initialized successfully")
        
        # 測試處理
        test_audio = np.random.randn(16000).astype(np.float32) * 0.1
        segments = vad.process(test_audio)
        print(f"Detected {len(segments)} speech segments")
    else:
        print("VAD not available, please download the model")
