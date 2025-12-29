"""
動態噪音抑制模組 (DNS - Dynamic Noise Suppression)

整合多種降噪方案：
1. DeepFilterNet：效果最好，推薦使用
2. ONNX 模型（如 NSNet2）：輕量替代方案
3. Passthrough：無降噪（備用）

設計說明：
優先使用 DeepFilterNet（需要 PyTorch），若不可用則嘗試 ONNX 模型，
最後使用 Passthrough 確保系統可運作。
"""
from pathlib import Path
from typing import Optional, Union

import numpy as np

# 嘗試導入 DeepFilterNet
try:
    from src.audio.deepfilternet import (
        DeepFilterNetEnhancer,
        StreamingDeepFilterNet,
        DEEPFILTER_AVAILABLE
    )
except ImportError:
    DEEPFILTER_AVAILABLE = False


class OnnxDNS:
    """
    ONNX 模型降噪
    
    使用 ONNX Runtime 執行降噪模型（如 NSNet2）。
    當 DeepFilterNet 不可用時的備用方案。
    """
    
    def __init__(
        self,
        model_path: str,
        sample_rate: int = 16000,
        use_gpu: bool = True
    ):
        """
        初始化 ONNX DNS
        
        Args:
            model_path: ONNX 模型路徑
            sample_rate: 取樣率
            use_gpu: 是否使用 GPU
        """
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.use_gpu = use_gpu
        
        self._model = None
        self._is_initialized = False
        
        self._init_model()
    
    def _init_model(self) -> None:
        """初始化 ONNX 模型"""
        try:
            import onnxruntime as ort
            
            if not Path(self.model_path).exists():
                print(f"ONNX DNS model not found: {self.model_path}")
                return
            
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if not self.use_gpu:
                providers = ['CPUExecutionProvider']
            
            self._model = ort.InferenceSession(
                self.model_path,
                providers=providers
            )
            
            self._is_initialized = True
            print(f"ONNX DNS loaded: {self.model_path}")
            
        except ImportError:
            print("ONNX Runtime not available")
        except Exception as e:
            print(f"Failed to load ONNX DNS: {e}")
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """處理音訊降噪"""
        if not self._is_initialized or self._model is None:
            return audio
        
        try:
            audio = audio.astype(np.float32)
            
            if audio.ndim == 1:
                audio = audio.reshape(1, -1)
            
            input_name = self._model.get_inputs()[0].name
            output_name = self._model.get_outputs()[0].name
            
            result = self._model.run(
                [output_name],
                {input_name: audio}
            )[0]
            
            return result.flatten()
            
        except Exception as e:
            print(f"ONNX DNS error: {e}")
            return audio.flatten()
    
    @property
    def is_available(self) -> bool:
        return self._is_initialized


class PassthroughDNS:
    """
    Passthrough DNS（無降噪）
    
    當沒有降噪模型時使用，直接回傳原始音訊。
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        return audio
    
    def enhance(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """相容 DeepFilterNet 介面"""
        return audio
    
    @property
    def is_available(self) -> bool:
        return True


def create_dns(
    backend: str = "auto",
    model_path: Optional[str] = None,
    sample_rate: int = 16000,
    use_gpu: bool = True,
    streaming: bool = False
) -> Union[DeepFilterNetEnhancer, OnnxDNS, PassthroughDNS]:
    """
    建立 DNS 實例的工廠函式
    
    Args:
        backend: 後端選擇
            - "auto": 自動選擇（優先 DeepFilterNet）
            - "deepfilternet": 強制使用 DeepFilterNet
            - "onnx": 使用 ONNX 模型
            - "passthrough": 不降噪
        model_path: ONNX 模型路徑（僅 backend="onnx" 時使用）
        sample_rate: 取樣率
        use_gpu: 是否使用 GPU
        streaming: 是否使用串流模式（僅 DeepFilterNet）
    
    Returns:
        DNS 實例
    """
    # 自動選擇後端
    if backend == "auto":
        if DEEPFILTER_AVAILABLE:
            backend = "deepfilternet"
        elif model_path and Path(model_path).exists():
            backend = "onnx"
        else:
            backend = "passthrough"
    
    # 根據後端建立實例
    if backend == "deepfilternet":
        if not DEEPFILTER_AVAILABLE:
            print("DeepFilterNet not available, falling back to passthrough")
            return PassthroughDNS(sample_rate)
        
        if streaming:
            print("Using StreamingDeepFilterNet for speech enhancement")
            return StreamingDeepFilterNet(
                sample_rate=sample_rate,
                chunk_size_ms=100
            )
        else:
            print("Using DeepFilterNet for speech enhancement")
            return DeepFilterNetEnhancer(sample_rate=sample_rate)
    
    elif backend == "onnx":
        if not model_path or not Path(model_path).exists():
            print(f"ONNX model not found: {model_path}, using passthrough")
            return PassthroughDNS(sample_rate)
        
        print("Using ONNX DNS for speech enhancement")
        return OnnxDNS(model_path, sample_rate, use_gpu)
    
    else:
        print("Using passthrough (no speech enhancement)")
        return PassthroughDNS(sample_rate)


# 測試用主程式
if __name__ == "__main__":
    print("=== DNS 模組測試 ===")
    print(f"DeepFilterNet available: {DEEPFILTER_AVAILABLE}")
    
    # 自動選擇後端
    dns = create_dns(backend="auto")
    print(f"Selected backend: {type(dns).__name__}")
    print(f"DNS available: {dns.is_available}")
    
    # 測試處理
    test_audio = np.random.randn(16000).astype(np.float32) * 0.1
    
    if hasattr(dns, 'enhance'):
        processed = dns.enhance(test_audio, 16000)
    else:
        processed = dns.process(test_audio)
    
    print(f"Input shape: {test_audio.shape}")
    print(f"Output shape: {processed.shape}")
