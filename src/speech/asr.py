"""
語音辨識模組 (ASR - Automatic Speech Recognition)

使用 Sherpa-onnx 實作中文即時語音辨識。
支援 Streaming（即時）和 Offline（離線）兩種模式。

設計說明：
1. 使用 Streaming ASR 以達到即時辨識效果
2. 支援中文模型，適合護理機器人場景
3. 可切換為 Paraformer 等離線模型獲得更高精度
"""
from pathlib import Path
from typing import Callable, Generator, Optional

import numpy as np

try:
    import sherpa_onnx
    SHERPA_AVAILABLE = True
except ImportError:
    SHERPA_AVAILABLE = False
    print("Warning: sherpa_onnx not installed, ASR will be disabled")


class StreamingASR:
    """
    串流語音辨識器
    
    即時處理音訊並輸出辨識結果，
    適合與 VAD 結合使用於對話系統。
    """
    
    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        joiner_path: str,
        tokens_path: str,
        sample_rate: int = 16000,
        num_threads: int = 4,
        provider: str = "cuda",
        decoding_method: str = "greedy_search"
    ):
        """
        初始化 Streaming ASR
        
        Args:
            encoder_path: Encoder 模型路徑
            decoder_path: Decoder 模型路徑
            joiner_path: Joiner 模型路徑
            tokens_path: Tokens 檔案路徑
            sample_rate: 取樣率 (必須為 16000)
            num_threads: 推理執行緒數
            provider: 推理後端 ("cuda" 或 "cpu")
            decoding_method: 解碼方法 ("greedy_search" 或 "modified_beam_search")
        """
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.joiner_path = joiner_path
        self.tokens_path = tokens_path
        self.sample_rate = sample_rate
        self.num_threads = num_threads
        self.provider = provider
        self.decoding_method = decoding_method
        
        self._recognizer = None
        self._stream = None
        self._is_initialized = False
        
        # 辨識結果回調
        self._on_result_callbacks: list[Callable[[str, bool], None]] = []
        
        self._init_model()
    
    def _init_model(self) -> None:
        """初始化 ASR 模型"""
        if not SHERPA_AVAILABLE:
            print("sherpa_onnx not available, ASR disabled")
            return
        
        # 檢查模型檔案
        for path in [self.encoder_path, self.decoder_path, 
                     self.joiner_path, self.tokens_path]:
            if not Path(path).exists():
                print(f"ASR model file not found: {path}")
                return
        
        try:
            # 使用新版 API: from_transducer 類方法
            self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                encoder=self.encoder_path,
                decoder=self.decoder_path,
                joiner=self.joiner_path,
                tokens=self.tokens_path,
                num_threads=self.num_threads,
                sample_rate=self.sample_rate,
                feature_dim=80,
                enable_endpoint_detection=True,
                rule1_min_trailing_silence=2.4,
                rule2_min_trailing_silence=1.2,
                rule3_min_utterance_length=20.0,
                decoding_method=self.decoding_method,
                provider=self.provider
            )
            
            self._stream = self._recognizer.create_stream()
            
            self._is_initialized = True
            print(f"Streaming ASR initialized")
            
        except Exception as e:
            print(f"Failed to initialize ASR: {e}")
    
    def process(self, audio_chunk: np.ndarray) -> tuple[str, bool]:
        """
        處理音訊 chunk 並取得辨識結果
        
        Args:
            audio_chunk: 音訊資料 (float32, 16kHz)
        
        Returns:
            (辨識文字, 是否為最終結果)
        """
        if not self._is_initialized or self._recognizer is None:
            return "", False
        
        # 確保格式正確
        audio_chunk = audio_chunk.astype(np.float32)
        
        # 餵入 ASR
        self._stream.accept_waveform(self.sample_rate, audio_chunk)
        
        # 處理
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)
        
        # 取得結果 (新版 API 返回字串，舊版返回物件)
        result = self._recognizer.get_result(self._stream)
        if isinstance(result, str):
            text = result.strip()
        else:
            text = result.text.strip() if result.text else ""
        
        # 檢查是否為端點（語句結束）
        is_endpoint = self._recognizer.is_endpoint(self._stream)
        
        if is_endpoint and text:
            # 重置 stream
            self._recognizer.reset(self._stream)
            
            # 觸發回調
            for callback in self._on_result_callbacks:
                try:
                    callback(text, True)
                except Exception as e:
                    print(f"ASR callback error: {e}")
        
        return text, is_endpoint
    
    def recognize(self, audio: np.ndarray) -> str:
        """
        辨識完整音訊段落
        
        Args:
            audio: 完整音訊資料
        
        Returns:
            辨識文字
        """
        if not self._is_initialized:
            return ""
        
        # 建立新的 stream
        stream = self._recognizer.create_stream()
        stream.accept_waveform(self.sample_rate, audio.astype(np.float32))
        
        # 送入尾端靜音以觸發端點偵測
        tail_padding = np.zeros(int(self.sample_rate * 0.5), dtype=np.float32)
        stream.accept_waveform(self.sample_rate, tail_padding)
        stream.input_finished()
        
        # 處理
        while self._recognizer.is_ready(stream):
            self._recognizer.decode_stream(stream)
        
        result = self._recognizer.get_result(stream)
        if isinstance(result, str):
            return result.strip()
        return result.text.strip() if result.text else ""
    
    def on_result(self, callback: Callable[[str, bool], None]) -> None:
        """
        註冊辨識結果回調
        
        Args:
            callback: 回調函式，接收 (文字, 是否最終結果)
        """
        self._on_result_callbacks.append(callback)
    
    def reset(self) -> None:
        """重置 ASR 狀態"""
        if self._recognizer:
            self._stream = self._recognizer.create_stream()
    
    @property
    def is_available(self) -> bool:
        """ASR 是否可用"""
        return self._is_initialized


class OfflineASR:
    """
    離線語音辨識器
    
    處理完整音訊檔案，精度較高但延遲較大。
    適合用於辨識預錄音訊或 VAD 切出的完整語音段落。
    """
    
    def __init__(
        self,
        model_path: str,
        tokens_path: str,
        sample_rate: int = 16000,
        num_threads: int = 4,
        provider: str = "cuda"
    ):
        """
        初始化 Offline ASR (Paraformer)
        
        Args:
            model_path: 模型路徑
            tokens_path: Tokens 檔案路徑
            sample_rate: 取樣率
            num_threads: 推理執行緒數
            provider: 推理後端
        """
        self.model_path = model_path
        self.tokens_path = tokens_path
        self.sample_rate = sample_rate
        self.num_threads = num_threads
        self.provider = provider
        
        self._recognizer = None
        self._is_initialized = False
        
        self._init_model()
    
    def _init_model(self) -> None:
        """初始化 Offline ASR 模型"""
        if not SHERPA_AVAILABLE:
            return
        
        if not Path(self.model_path).exists():
            print(f"ASR model not found: {self.model_path}")
            return
        
        try:
            config = sherpa_onnx.OfflineRecognizerConfig(
                feat_config=sherpa_onnx.FeatureExtractorConfig(
                    sample_rate=self.sample_rate,
                    feature_dim=80
                ),
                model_config=sherpa_onnx.OfflineModelConfig(
                    paraformer=sherpa_onnx.OfflineParaformerModelConfig(
                        model=self.model_path
                    ),
                    tokens=self.tokens_path,
                    num_threads=self.num_threads,
                    provider=self.provider
                )
            )
            
            self._recognizer = sherpa_onnx.OfflineRecognizer(config)
            self._is_initialized = True
            print("Offline ASR (Paraformer) initialized")
            
        except Exception as e:
            print(f"Failed to initialize Offline ASR: {e}")
    
    def recognize(self, audio: np.ndarray) -> str:
        """
        辨識音訊
        
        Args:
            audio: 音訊資料 (float32, 16kHz)
        
        Returns:
            辨識文字
        """
        if not self._is_initialized or self._recognizer is None:
            return ""
        
        audio = audio.astype(np.float32)
        stream = self._recognizer.create_stream()
        stream.accept_waveform(self.sample_rate, audio)
        
        self._recognizer.decode_stream(stream)
        
        result = stream.result
        return result.text.strip() if result.text else ""
    
    @property
    def is_available(self) -> bool:
        """ASR 是否可用"""
        return self._is_initialized


def create_streaming_asr(
    model_dir: Optional[str] = None,
    **kwargs
) -> Optional[StreamingASR]:
    """
    建立 Streaming ASR 實例
    
    Args:
        model_dir: 模型目錄
        **kwargs: 其他參數
    
    Returns:
        StreamingASR 實例或 None
    """
    if model_dir is None:
        model_dir = "models/asr/sherpa-onnx-streaming-zipformer-zh"
    
    model_dir = Path(model_dir)
    
    # 優先使用 kwargs 傳入的路徑
    encoder = kwargs.pop('encoder_path', None)
    decoder = kwargs.pop('decoder_path', None)
    joiner = kwargs.pop('joiner_path', None)
    tokens = kwargs.pop('tokens_path', None)
    
    # 如果未指定，嘗試自動偵測
    if not encoder:
        # 嘗試常見的檔名模式
        candidates = [
            model_dir / "encoder.onnx",
            model_dir / "encoder.int8.onnx",
            model_dir / "encoder.fp16.onnx",
            model_dir / "encoder-epoch-99-avg-1.onnx",
            model_dir / "encoder-epoch-99-avg-1.int8.onnx",
        ]
        # 也嘗試 glob 搜尋
        if not any(p.exists() for p in candidates):
            glob_files = list(model_dir.glob("encoder*.onnx"))
            if glob_files:
                candidates.insert(0, glob_files[0])
        
        for p in candidates:
            if p.exists():
                encoder = str(p)
                break
    
    if not decoder:
        # 根據 encoder 檔名推斷 decoder (e.g., encoder.onnx -> decoder.onnx)
        if encoder:
            enc_name = Path(encoder).name
            dec_name = enc_name.replace("encoder", "decoder")
            if (model_dir / dec_name).exists():
                decoder = str(model_dir / dec_name)
        
        # 如果推斷失敗，嘗試常見檔名
        if not decoder:
             candidates = [
                model_dir / "decoder.onnx",
                model_dir / "decoder.int8.onnx",
                model_dir / "decoder.fp16.onnx",
                model_dir / "decoder-epoch-99-avg-1.onnx",
                model_dir / "decoder-epoch-99-avg-1.int8.onnx",
            ]
             for p in candidates:
                if p.exists():
                    decoder = str(p)
                    break

    if not joiner:
        if encoder:
            enc_name = Path(encoder).name
            join_name = enc_name.replace("encoder", "joiner")
            if (model_dir / join_name).exists():
                joiner = str(model_dir / join_name)
        
        if not joiner:
            candidates = [
                model_dir / "joiner.onnx",
                model_dir / "joiner.int8.onnx",
                model_dir / "joiner.fp16.onnx",
                model_dir / "joiner-epoch-99-avg-1.onnx",
                model_dir / "joiner-epoch-99-avg-1.int8.onnx",
            ]
            for p in candidates:
                if p.exists():
                    joiner = str(p)
                    break
    
    if not tokens:
        tokens = str(model_dir / "tokens.txt")

    # 檢查檔案是否存在
    missing = []
    if not encoder or not Path(encoder).exists(): missing.append("encoder")
    if not decoder or not Path(decoder).exists(): missing.append("decoder")
    if not joiner or not Path(joiner).exists(): missing.append("joiner")
    if not tokens or not Path(tokens).exists(): missing.append("tokens")
    
    if missing:
        print(f"ASR model files missing ({', '.join(missing)}) in: {model_dir}")
        print("Please download the model first or check config")
        # 列出目錄內容以便除錯
        if model_dir.exists():
            print(f"Contents of {model_dir}:")
            for f in model_dir.iterdir():
                print(f"  {f.name}")
        return None
    
    return StreamingASR(
        encoder_path=str(encoder),
        decoder_path=str(decoder),
        joiner_path=str(joiner),
        tokens_path=str(tokens),
        **kwargs
    )


# 測試用主程式
if __name__ == "__main__":
    print("=== ASR 模組測試 ===")
    
    if not SHERPA_AVAILABLE:
        print("sherpa_onnx not installed, skipping test")
        exit(1)
    
    # 嘗試建立 ASR
    asr = create_streaming_asr(model_dir="/home/jetson/voice_control/models/asr")
    
    if asr and asr.is_available:
        print("ASR initialized successfully")
        
        # 測試處理（靜音）
        silence = np.zeros(16000, dtype=np.float32)
        text, is_final = asr.process(silence)
        print(f"Test result: '{text}' (final={is_final})")
    else:
        print("ASR not available, please download the model")
