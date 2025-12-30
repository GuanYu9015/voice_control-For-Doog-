"""
喚醒詞偵測模組 (KWS - Keyword Spotting)

使用 Sherpa-onnx 實作喚醒詞偵測。
喚醒詞：「智護車」、「護理車」

設計說明：
1. 喚醒詞採用開放詞彙方式，不需重新訓練模型
2. 偵測到喚醒詞後觸發後續語音辨識流程
"""
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

try:
    import sherpa_onnx
    SHERPA_AVAILABLE = True
except ImportError:
    SHERPA_AVAILABLE = False
    print("Warning: sherpa_onnx not installed, KWS will be disabled")


class KeywordSpotter:
    """
    喚醒詞偵測器
    
    使用 Sherpa-onnx 的 Keyword Spotting 功能偵測自定義喚醒詞。
    """
    
    # 預設喚醒詞
    DEFAULT_KEYWORDS = ["智護車", "護理車"]
    
    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        joiner_path: str,
        tokens_path: str,
        keywords: Optional[List[str]] = None,
        keywords_file: Optional[str] = None,  # 新增：外部 keywords 檔案路徑
        sample_rate: int = 16000,
        threshold: float = 0.25,
        num_threads: int = 2,
        provider: str = "cuda"
    ):
        """
        初始化喚醒詞偵測器
        
        Args:
            encoder_path: Encoder 模型路徑
            decoder_path: Decoder 模型路徑
            joiner_path: Joiner 模型路徑
            tokens_path: Tokens 檔案路徑
            keywords: 喚醒詞列表（若有 keywords_file 則忽略）
            keywords_file: 外部 keywords 檔案路徑（拼音格式）
            sample_rate: 取樣率
            threshold: 偵測閾值
            num_threads: 執行緒數
            provider: 推理後端 ("cuda" 或 "cpu")
        """
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.joiner_path = joiner_path
        self.tokens_path = tokens_path
        self.keywords = keywords or self.DEFAULT_KEYWORDS
        self.keywords_file = keywords_file  # 新增
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.num_threads = num_threads
        self.provider = provider
        
        self._kws = None
        self._stream = None
        self._is_initialized = False
        
        # 喚醒回調
        self._on_wake_callbacks: list[Callable[[str], None]] = []
        
        self._init_model()
    
    def _init_model(self) -> None:
        """初始化 KWS 模型"""
        if not SHERPA_AVAILABLE:
            print("sherpa_onnx not available, KWS disabled")
            return
        
        # 檢查模型檔案
        for path in [self.encoder_path, self.decoder_path, 
                     self.joiner_path, self.tokens_path]:
            if not Path(path).exists():
                print(f"KWS model file not found: {path}")
                return
        
        try:
            # 決定 keywords 檔案路徑
            # 優先使用外部指定的 keywords_file
            if self.keywords_file and Path(self.keywords_file).exists():
                keywords_file = Path(self.keywords_file)
                print(f"Using external keywords file: {keywords_file}")
            else:
                # Fallback: 使用模型目錄中的 keywords.txt
                model_dir = Path(self.encoder_path).parent
                keywords_file = model_dir / "keywords.txt"
                
                # 如果沒有 keywords.txt，動態創建一個（但格式可能不正確）
                if not keywords_file.exists():
                    print(f"Warning: keywords.txt not found, creating default")
                    with open(keywords_file, 'w', encoding='utf-8') as f:
                        for kw in self.keywords:
                            chars = ' '.join(kw)
                            f.write(f"{chars}\n")
            
            # 使用新版 sherpa-onnx API (1.12.x+)
            # 注意：sample_rate 必須是 int，provider 使用 cpu（sherpa-onnx 未編譯 GPU）
            self._kws = sherpa_onnx.KeywordSpotter(
                tokens=self.tokens_path,
                encoder=self.encoder_path,
                decoder=self.decoder_path,
                joiner=self.joiner_path,
                keywords_file=str(keywords_file),
                num_threads=self.num_threads,
                sample_rate=int(self.sample_rate),  # 必須是 int
                feature_dim=80,
                keywords_threshold=self.threshold,
                num_trailing_blanks=1,
                provider="cpu"  # sherpa-onnx 未啟用 GPU
            )
            self._stream = self._kws.create_stream()
            
            self._is_initialized = True
            print(f"KWS initialized with keywords: {self.keywords}")
            
        except Exception as e:
            print(f"Failed to initialize KWS: {e}")
    
    def process(self, audio_chunk: np.ndarray) -> Optional[str]:
        """
        處理音訊 chunk 並偵測喚醒詞
        
        Args:
            audio_chunk: 音訊資料 (float32, 16kHz)
        
        Returns:
            偵測到的喚醒詞，若無則回傳 None
        """
        if not self._is_initialized or self._kws is None:
            return None
        
        # 確保格式正確
        audio_chunk = audio_chunk.astype(np.float32)
        
        # 餵入 KWS
        self._stream.accept_waveform(self.sample_rate, audio_chunk)
        
        # 檢查是否偵測到關鍵字（使用新版 API）
        while self._kws.is_ready(self._stream):
            self._kws.decode_stream(self._stream)  # 新版 API: decode_stream
        
        result = self._kws.get_result(self._stream)
        
        if result:
            keyword = result.strip()
            
            # 重置 stream 以便繼續偵測
            # 使用 create_stream 完全重建（reset_stream 可能不完全清理）
            self._stream = self._kws.create_stream()
            
            # 觸發回調
            for callback in self._on_wake_callbacks:
                try:
                    callback(keyword)
                except Exception as e:
                    print(f"KWS callback error: {e}")
            
            return keyword
        
        return None
    
    def on_wake(self, callback: Callable[[str], None]) -> None:
        """
        註冊喚醒詞偵測回調
        
        Args:
            callback: 當偵測到喚醒詞時呼叫，參數為偵測到的詞
        """
        self._on_wake_callbacks.append(callback)
    
    def reset(self) -> None:
        """重置 KWS 狀態"""
        if self._kws:
            self._stream = self._kws.create_stream()
    
    @property
    def is_available(self) -> bool:
        """KWS 是否可用"""
        return self._is_initialized


class SimpleKeywordMatcher:
    """
    簡易喚醒詞比對器
    
    當 KWS 模型不可用時的備案，
    直接從 ASR 結果中比對喚醒詞。
    
    支援：
    - 繁體/簡體中文匹配
    - 模糊匹配（部分匹配）
    """
    
    # 簡繁體對照表（喚醒詞相關）
    # 包含可能的諧音/誤聽變體（增加更多變體提高匹配率）
    SIMPLIFIED_VARIANTS = {
        "智護車": [
            "智护车", "智護車", "自护车", "自護車",
            "之护车", "之護車", "志护车", "志護車",
            "智互车", "智护", "支护车", "知乎车"
        ],
        "護理車": [
            "护理车", "護理車", "戶理車", "户理车",
            "胡理车", "虎理车"
        ],
    }
    
    def __init__(self, keywords: Optional[List[str]] = None):
        """
        初始化
        
        Args:
            keywords: 喚醒詞列表
        """
        self.keywords = keywords or KeywordSpotter.DEFAULT_KEYWORDS
        self._on_wake_callbacks: list[Callable[[str], None]] = []
        
        # 建立擴展關鍵詞列表（包含簡體變體）
        self._expanded_keywords = {}
        for kw in self.keywords:
            variants = self.SIMPLIFIED_VARIANTS.get(kw, [kw])
            for variant in variants:
                self._expanded_keywords[variant.lower()] = kw
            # 確保原詞也在列表中
            self._expanded_keywords[kw.lower()] = kw
    
    def check(self, text: str) -> Optional[tuple[str, str]]:
        """
        檢查文字中是否包含喚醒詞
        
        Args:
            text: ASR 辨識結果
        
        Returns:
            (喚醒詞, 剩餘文字) 元組，若無則回傳 None
        """
        if not text:
            return None
        
        text_lower = text.lower()
        
        # 檢查所有變體
        for variant, original_kw in self._expanded_keywords.items():
            idx = text_lower.find(variant)
            if idx != -1:
                # 觸發回調
                for callback in self._on_wake_callbacks:
                    try:
                        callback(original_kw)
                    except Exception as e:
                        print(f"Keyword matcher callback error: {e}")
                
                # 提取剩餘文字（喚醒詞之後的部分）
                # 注意：這裡假設中文變體長度與原文一致
                end_idx = idx + len(variant)
                remaining = text[end_idx:].strip()
                
                # 如果剩餘文字只包含標點符號，視為空
                if remaining in [",", ".", "。", "，", "！", "!", "?", "？"]:
                    remaining = ""
                
                return original_kw, remaining
        
        return None
    
    def on_wake(self, callback: Callable[[str], None]) -> None:
        """註冊喚醒詞偵測回調"""
        self._on_wake_callbacks.append(callback)
    
    @property
    def is_available(self) -> bool:
        """永遠可用"""
        return True


def create_kws(
    model_dir: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    **kwargs
) -> KeywordSpotter:
    """
    建立 KWS 實例的工廠函式
    
    Args:
        model_dir: 模型目錄
        keywords: 喚醒詞列表
        **kwargs: 其他參數
    
    Returns:
        KWS 實例
    """
    if model_dir is None:
        model_dir = "models/kws"
    
    model_dir = Path(model_dir)
    
    # 預設使用 zipformer 模型
    encoder = str(model_dir / "encoder-epoch-12-avg-2-chunk-16-left-64.onnx")
    decoder = str(model_dir / "decoder-epoch-12-avg-2-chunk-16-left-64.onnx")
    joiner = str(model_dir / "joiner-epoch-12-avg-2-chunk-16-left-64.onnx")
    tokens = str(model_dir / "tokens.txt")
    
    return KeywordSpotter(
        encoder_path=encoder,
        decoder_path=decoder,
        joiner_path=joiner,
        tokens_path=tokens,
        keywords=keywords,
        **kwargs
    )


# 測試用主程式
if __name__ == "__main__":
    print("=== KWS 模組測試 ===")
    print(f"預設喚醒詞: {KeywordSpotter.DEFAULT_KEYWORDS}")
    
    # 測試簡易比對器
    matcher = SimpleKeywordMatcher()
    
    test_texts = [
        "你好智護車請幫我",
        "護理車請過來",
        "今天天氣很好"
    ]
    
    for text in test_texts:
        result = matcher.check(text)
        if result:
            print(f"  '{text}' -> 偵測到: {result}")
        else:
            print(f"  '{text}' -> 無喚醒詞")
