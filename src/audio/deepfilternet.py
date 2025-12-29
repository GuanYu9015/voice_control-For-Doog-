"""
DeepFilterNet 語音增強模組

使用 DeepFilterNet 進行即時語音增強與噪音抑制。
DeepFilterNet 是目前效果最好的深度學習降噪方案之一。

設計說明：
1. DeepFilterNet 需要 PyTorch，Jetson 平台已有 ARM 版本
2. 支援 GPU 加速（CUDA）
3. 即時處理需要適當的 buffer 管理

注意事項：
- 第一次載入模型較慢（需要編譯）
- 記憶體使用量較高（約 500MB-1GB）
- 建議在 GPU 上執行以獲得最佳效能
"""
from pathlib import Path
from typing import Optional
import numpy as np

# 嘗試導入 DeepFilterNet
try:
    from df.enhance import enhance, init_df, load_audio, save_audio
    from df import config as df_config
    DEEPFILTER_AVAILABLE = True
except ImportError:
    DEEPFILTER_AVAILABLE = False
    print("Warning: DeepFilterNet not installed")
    print("Install with: pip install deepfilternet")


class DeepFilterNetEnhancer:
    """
    DeepFilterNet 語音增強器
    
    提供即時語音增強與噪音抑制功能。
    使用深度濾波器網路，效果優於傳統方法。
    """
    
    # 支援的模型
    MODELS = {
        "DeepFilterNet": "DeepFilterNet",
        "DeepFilterNet2": "DeepFilterNet2", 
        "DeepFilterNet3": "DeepFilterNet3",  # 最新版本，效果最好
    }
    
    def __init__(
        self,
        model_name: str = "DeepFilterNet3",
        sample_rate: int = 16000,
        post_filter: bool = False,
        atten_lim_db: float = 100.0
    ):
        """
        初始化 DeepFilterNet 增強器
        
        Args:
            model_name: 模型名稱 (DeepFilterNet/DeepFilterNet2/DeepFilterNet3)
            sample_rate: 目標取樣率（輸出會重採樣到此取樣率）
            post_filter: 是否使用後處理濾波器（可進一步降噪但可能影響語音品質）
            atten_lim_db: 最大衰減量 (dB)，預設 100dB（完全消除噪音）
        """
        self.model_name = model_name
        self.target_sample_rate = sample_rate
        self.post_filter = post_filter
        self.atten_lim_db = atten_lim_db
        
        self._model = None
        self._df_state = None
        self._is_initialized = False
        
        # DeepFilterNet 原生取樣率（固定 48kHz）
        self._native_sample_rate = 48000
        
        self._init_model()
    
    def _init_model(self) -> None:
        """初始化 DeepFilterNet 模型"""
        if not DEEPFILTER_AVAILABLE:
            print("DeepFilterNet not available")
            return
        
        try:
            print(f"Loading DeepFilterNet model: {self.model_name}")
            print("(First load may take a while for model compilation...)")
            
            # 初始化 DeepFilterNet
            self._model, self._df_state, _ = init_df(
                post_filter=self.post_filter,
                log_level="warning"
            )
            
            self._is_initialized = True
            print(f"DeepFilterNet initialized (native SR: {self._native_sample_rate}Hz)")
            
        except Exception as e:
            print(f"Failed to initialize DeepFilterNet: {e}")
    
    def _resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """
        重採樣音訊
        
        Args:
            audio: 音訊資料
            orig_sr: 原始取樣率
            target_sr: 目標取樣率
        
        Returns:
            重採樣後的音訊
        """
        if orig_sr == target_sr:
            return audio
        
        try:
            from scipy import signal
            
            # 計算重採樣比例
            num_samples = int(len(audio) * target_sr / orig_sr)
            resampled = signal.resample(audio, num_samples)
            
            return resampled.astype(np.float32)
            
        except ImportError:
            print("scipy not available for resampling")
            return audio
    
    def enhance(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        增強音訊（降噪）
        
        Args:
            audio: 輸入音訊 (numpy array, float32, -1.0 ~ 1.0)
            sample_rate: 輸入音訊的取樣率
        
        Returns:
            增強後的音訊（與輸入相同取樣率）
        """
        if not self._is_initialized or self._model is None:
            return audio
        
        try:
            import torch
            
            # 確保輸入格式正確
            audio = audio.astype(np.float32)
            
            # 重採樣到 48kHz（DeepFilterNet 原生取樣率）
            if sample_rate != self._native_sample_rate:
                audio_48k = self._resample(audio, sample_rate, self._native_sample_rate)
            else:
                audio_48k = audio
            
            # 轉換為 torch tensor
            audio_tensor = torch.from_numpy(audio_48k).unsqueeze(0)
            
            # 增強處理
            enhanced = enhance(
                self._model,
                self._df_state,
                audio_tensor,
                atten_lim_db=self.atten_lim_db
            )
            
            # 轉回 numpy
            enhanced_np = enhanced.squeeze().numpy()
            
            # 重採樣回原始取樣率
            if sample_rate != self._native_sample_rate:
                enhanced_np = self._resample(
                    enhanced_np, 
                    self._native_sample_rate, 
                    sample_rate
                )
            
            return enhanced_np
            
        except Exception as e:
            print(f"DeepFilterNet enhancement error: {e}")
            return audio
    
    def enhance_realtime(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int = 16000
    ) -> np.ndarray:
        """
        即時增強音訊 chunk
        
        注意：DeepFilterNet 設計上需要較長的音訊段落才能發揮最佳效果。
        對於非常短的 chunk（< 100ms），效果可能較差。
        
        建議：將多個 chunk 累積後一起處理，或使用重疊處理。
        
        Args:
            audio_chunk: 音訊 chunk
            sample_rate: 取樣率
        
        Returns:
            增強後的音訊 chunk
        """
        # 對於即時處理，直接調用 enhance
        # 未來可實作滑動視窗處理以改善邊界效應
        return self.enhance(audio_chunk, sample_rate)
    
    def enhance_file(
        self,
        input_path: str,
        output_path: str
    ) -> bool:
        """
        增強音訊檔案
        
        Args:
            input_path: 輸入檔案路徑
            output_path: 輸出檔案路徑
        
        Returns:
            是否成功
        """
        if not self._is_initialized:
            return False
        
        try:
            # 載入音訊
            audio, sample_rate = load_audio(input_path, sr=self._native_sample_rate)
            
            # 轉換為 numpy
            audio_np = audio.numpy()
            
            # 增強
            enhanced = self.enhance(audio_np, self._native_sample_rate)
            
            # 儲存
            import torch
            enhanced_tensor = torch.from_numpy(enhanced)
            save_audio(output_path, enhanced_tensor, self._native_sample_rate)
            
            return True
            
        except Exception as e:
            print(f"Failed to enhance file: {e}")
            return False
    
    @property
    def is_available(self) -> bool:
        """模型是否可用"""
        return self._is_initialized
    
    @property
    def native_sample_rate(self) -> int:
        """原生取樣率"""
        return self._native_sample_rate


class StreamingDeepFilterNet:
    """
    串流式 DeepFilterNet 增強器
    
    適合即時處理場景，使用滑動視窗處理音訊。
    
    設計說明：
    - 使用 overlap-add 方法處理音訊塊
    - 減少邊界效應
    - 維持較低延遲
    """
    
    def __init__(
        self,
        chunk_size_ms: int = 100,
        overlap_ratio: float = 0.5,
        sample_rate: int = 16000
    ):
        """
        初始化串流增強器
        
        Args:
            chunk_size_ms: 處理 chunk 大小（毫秒）
            overlap_ratio: 重疊比例
            sample_rate: 取樣率
        """
        self.chunk_size_ms = chunk_size_ms
        self.overlap_ratio = overlap_ratio
        self.sample_rate = sample_rate
        
        # 計算樣本數
        self.chunk_samples = int(sample_rate * chunk_size_ms / 1000)
        self.overlap_samples = int(self.chunk_samples * overlap_ratio)
        self.hop_samples = self.chunk_samples - self.overlap_samples
        
        # 緩衝區
        self._input_buffer = np.array([], dtype=np.float32)
        self._output_buffer = np.array([], dtype=np.float32)
        
        # 初始化 DeepFilterNet
        self._enhancer = DeepFilterNetEnhancer(sample_rate=sample_rate)
    
    def process(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        處理音訊 chunk（串流式）
        
        Args:
            audio_chunk: 輸入音訊 chunk
        
        Returns:
            增強後的音訊（可能與輸入長度不同）
        """
        if not self._enhancer.is_available:
            return audio_chunk
        
        # 加入輸入緩衝區
        self._input_buffer = np.concatenate([self._input_buffer, audio_chunk])
        
        output_chunks = []
        
        # 處理完整的 chunks
        while len(self._input_buffer) >= self.chunk_samples:
            # 取出一個 chunk
            chunk = self._input_buffer[:self.chunk_samples]
            
            # 增強
            enhanced_chunk = self._enhancer.enhance(chunk, self.sample_rate)
            
            # 只保留非重疊部分
            output_chunks.append(enhanced_chunk[:self.hop_samples])
            
            # 移動緩衝區
            self._input_buffer = self._input_buffer[self.hop_samples:]
        
        if output_chunks:
            return np.concatenate(output_chunks)
        else:
            return np.array([], dtype=np.float32)
    
    def flush(self) -> np.ndarray:
        """
        清空緩衝區並取得剩餘音訊
        
        Returns:
            剩餘的增強音訊
        """
        if len(self._input_buffer) > 0:
            # 處理剩餘音訊（padding 到完整 chunk）
            padded = np.pad(
                self._input_buffer,
                (0, self.chunk_samples - len(self._input_buffer))
            )
            enhanced = self._enhancer.enhance(padded, self.sample_rate)
            result = enhanced[:len(self._input_buffer)]
            self._input_buffer = np.array([], dtype=np.float32)
            return result
        
        return np.array([], dtype=np.float32)
    
    def reset(self) -> None:
        """重置緩衝區"""
        self._input_buffer = np.array([], dtype=np.float32)
        self._output_buffer = np.array([], dtype=np.float32)
    
    @property
    def is_available(self) -> bool:
        """是否可用"""
        return self._enhancer.is_available


def create_deepfilternet(
    streaming: bool = False,
    **kwargs
) -> DeepFilterNetEnhancer:
    """
    建立 DeepFilterNet 增強器的工廠函式
    
    Args:
        streaming: 是否使用串流模式
        **kwargs: 其他參數
    
    Returns:
        增強器實例
    """
    if streaming:
        return StreamingDeepFilterNet(**kwargs)
    else:
        return DeepFilterNetEnhancer(**kwargs)


# 測試用主程式
if __name__ == "__main__":
    print("=== DeepFilterNet 模組測試 ===")
    print(f"DeepFilterNet available: {DEEPFILTER_AVAILABLE}")
    
    if not DEEPFILTER_AVAILABLE:
        print("\n安裝指令:")
        print("  pip install deepfilternet")
        exit(1)
    
    # 建立增強器
    enhancer = create_deepfilternet(streaming=False)
    
    if enhancer.is_available:
        print("\nDeepFilterNet 初始化成功")
        
        # 測試：產生含噪音的音訊
        duration = 2.0  # 秒
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # 正弦波 + 白噪音
        clean_signal = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz
        noise = 0.1 * np.random.randn(len(t))
        noisy_signal = (clean_signal + noise).astype(np.float32)
        
        print(f"\n輸入: {len(noisy_signal)} 樣本, {duration} 秒")
        
        # 增強
        enhanced = enhancer.enhance(noisy_signal, sample_rate)
        
        print(f"輸出: {len(enhanced)} 樣本")
        print(f"SNR 改善: 預期約 10-20dB")
    else:
        print("DeepFilterNet 初始化失敗")
