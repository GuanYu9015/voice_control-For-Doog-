"""
語音合成模組 (TTS - Text-to-Speech)

使用 Sherpa-onnx 實作中文語音合成。
將 LLM 輸出的文字轉換為語音回應。

設計說明：
1. 使用 VITS 模型，音質自然
2. 支援中文語音合成
3. 可調整語速
"""
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import sherpa_onnx
    SHERPA_AVAILABLE = True
except ImportError:
    SHERPA_AVAILABLE = False
    print("Warning: sherpa_onnx not installed, TTS will be disabled")


class TextToSpeech:
    """
    語音合成器
    
    將文字轉換為語音，支援中文。
    """
    
    def __init__(
        self,
        model_path: str,
        tokens_path: str,
        lexicon_path: Optional[str] = None,
        data_dir: Optional[str] = None,
        dict_dir: Optional[str] = None,
        sample_rate: int = 22050,
        num_threads: int = 2,
        provider: str = "cuda",
        speed: float = 0.5,
        sid: int = 0
    ):
        """
        初始化 TTS
        
        Args:
            model_path: VITS 模型路徑
            tokens_path: Tokens 檔案路徑
            lexicon_path: 詞典路徑（若模型需要）
            data_dir: 資料目錄（若模型需要）
            dict_dir: 字典目錄（若模型需要）
            sample_rate: 輸出取樣率
            num_threads: 推理執行緒數
            provider: 推理後端 ("cuda" 或 "cpu")
            speed: 語速 (0.5 ~ 2.0)
            sid: 說話者 ID（若模型支援多說話者）
        """
        self.model_path = model_path
        self.tokens_path = tokens_path
        self.lexicon_path = lexicon_path
        self.data_dir = data_dir
        self.dict_dir = dict_dir
        self.sample_rate = sample_rate
        self.num_threads = num_threads
        self.provider = provider
        self.speed = speed
        self.sid = sid
        
        self._tts = None
        self._is_initialized = False
        
        self._init_model()
    
    def _init_model(self) -> None:
        """初始化 TTS 模型"""
        if not SHERPA_AVAILABLE:
            print("sherpa_onnx not available, TTS disabled")
            return
        
        if not Path(self.model_path).exists():
            print(f"TTS model not found: {self.model_path}")
            return
        
        try:
            # 建立 TTS 設定
            # 根據不同的模型類型調整設定
            vits_config = sherpa_onnx.OfflineTtsVitsModelConfig(
                model=self.model_path,
                tokens=self.tokens_path,
                lexicon=self.lexicon_path or "",
                data_dir=self.data_dir or "",
                dict_dir=self.dict_dir or ""
            )
            
            model_config = sherpa_onnx.OfflineTtsModelConfig(
                vits=vits_config,
                num_threads=self.num_threads,
                provider=self.provider
            )
            
            config = sherpa_onnx.OfflineTtsConfig(
                model=model_config,
                max_num_sentences=1
            )
            
            self._tts = sherpa_onnx.OfflineTts(config)
            
            # 更新實際取樣率
            self.sample_rate = self._tts.sample_rate
            
            self._is_initialized = True
            print(f"TTS initialized (sample_rate={self.sample_rate})")
            
        except Exception as e:
            print(f"Failed to initialize TTS: {e}")
    
    def synthesize(self, text: str, speed: Optional[float] = None) -> np.ndarray:
        """
        合成語音
        
        Args:
            text: 要合成的文字
            speed: 語速（若為 None 則使用預設值）
        
        Returns:
            音訊資料 (numpy array, float32)
        """
        if not self._is_initialized or self._tts is None:
            return np.array([], dtype=np.float32)
        
        speed = speed or self.speed
        
        try:
            audio = self._tts.generate(
                text,
                sid=self.sid,
                speed=speed
            )
            
            return np.array(audio.samples, dtype=np.float32)
            
        except Exception as e:
            print(f"TTS synthesis error: {e}")
            return np.array([], dtype=np.float32)
    
    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        speed: Optional[float] = None
    ) -> bool:
        """
        合成語音並儲存為檔案
        
        Args:
            text: 要合成的文字
            output_path: 輸出檔案路徑
            speed: 語速
        
        Returns:
            是否成功
        """
        try:
            import soundfile as sf
            
            audio = self.synthesize(text, speed)
            
            if len(audio) == 0:
                return False
            
            sf.write(output_path, audio, self.sample_rate)
            return True
            
        except Exception as e:
            print(f"Failed to save TTS output: {e}")
            return False
    
    def stream_synthesize(
        self,
        text: str,
        chunk_size: int = 4096
    ):
        """
        串流合成語音（產生器）
        
        Args:
            text: 要合成的文字
            chunk_size: 每個 chunk 的樣本數
        
        Yields:
            音訊 chunk (numpy array)
            
        注意：目前 sherpa-onnx 的 TTS 不支援真正的串流，
        這裡是將完整音訊切成 chunks 回傳。
        """
        audio = self.synthesize(text)
        
        for i in range(0, len(audio), chunk_size):
            yield audio[i:i + chunk_size]
    
    @property
    def is_available(self) -> bool:
        """TTS 是否可用"""
        return self._is_initialized
    
    @property
    def num_speakers(self) -> int:
        """可用的說話者數量"""
        if self._tts:
            return self._tts.num_speakers
        return 0


def create_tts(
    model_dir: Optional[str] = None,
    **kwargs
) -> Optional[TextToSpeech]:
    """
    建立 TTS 實例
    
    Args:
        model_dir: 模型目錄
        **kwargs: 其他參數
    
    Returns:
        TTS 實例或 None
    """
    if model_dir is None:
        model_dir = "models/tts/vits-zh-aishell3"
    
    model_dir = Path(model_dir)
    
    model = str(model_dir / "model.onnx")
    tokens = str(model_dir / "tokens.txt")
    lexicon = str(model_dir / "lexicon.txt")
    
    # 檢查最基本的檔案
    if not Path(model).exists():
        print(f"TTS model not found: {model}")
        print("Please download the model first")
        return None
    
    return TextToSpeech(
        model_path=model,
        tokens_path=tokens,
        lexicon_path=lexicon if Path(lexicon).exists() else None,
        **kwargs
    )


# 測試用主程式
if __name__ == "__main__":
    print("=== TTS 模組測試 ===")
    
    if not SHERPA_AVAILABLE:
        print("sherpa_onnx not installed, skipping test")
        exit(1)
    
    # 嘗試建立 TTS
    tts = create_tts(model_dir="/home/jetson/voice_control/models/tts")
    
    if tts and tts.is_available:
        print(f"TTS initialized (speakers={tts.num_speakers})")
        
        # 測試合成
        test_text = "你好，我是智護車，請問有什麼需要幫忙的嗎？"
        audio = tts.synthesize(test_text)
        
        if len(audio) > 0:
            duration = len(audio) / tts.sample_rate
            print(f"Synthesized: {len(audio)} samples, {duration:.2f} seconds")
        else:
            print("Synthesis failed")
    else:
        print("TTS not available, please download the model")
