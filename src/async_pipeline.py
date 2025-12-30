"""
非同步多執行緒語音控制 Pipeline

使用多執行緒和 Queue 實作非阻塞處理流程：
- 語音處理執行緒：麥克風 → DNS → VAD → ASR
- LLM 處理執行緒：接收 ASR 結果 → 生成回應
- TTS 輸出執行緒：接收 LLM 回應 → 語音合成 → 播放

各執行緒透過 Queue 通訊，不互相等待。

設計說明：
1. 語音處理持續運作，即使 LLM 正在處理
2. LLM 回應可以串流輸出到 TTS
3. 支援中斷當前回應（偵測到新喚醒詞時）
"""
import threading
import queue
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import yaml


class MessageType(Enum):
    """訊息類型"""
    ASR_RESULT = "asr_result"      # ASR 辨識結果
    LLM_RESPONSE = "llm_response"  # LLM 回應
    TTS_REQUEST = "tts_request"    # TTS 請求
    COMMAND = "command"            # 機器人指令
    WAKE_WORD = "wake_word"        # 喚醒詞
    STOP = "stop"                  # 停止信號


@dataclass
class Message:
    """執行緒間通訊訊息"""
    type: MessageType
    data: any
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class AsyncVoiceControlPipeline:
    """
    非同步多執行緒語音控制 Pipeline
    
    架構：
    ┌────────────────┐    ┌────────────────┐    ┌────────────────┐
    │  語音處理執行緒  │───▶│  LLM 處理執行緒  │───▶│  TTS 輸出執行緒  │
    │  (audio_worker) │    │  (llm_worker)  │    │  (tts_worker)  │
    └────────────────┘    └────────────────┘    └────────────────┘
           │                      │                     │
        asr_queue             llm_queue            tts_queue
    """
    
    def __init__(
        self,
        config_path: str = "config/audio_config.yaml",
        model_config_path: str = "config/model_config.yaml",
        robot_config_path: str = "config/robot_config.yaml",
        use_wake_word: bool = True,
        use_mock_llm: bool = False,
        use_mock_robot: bool = False,
        on_command: Optional[Callable[[dict], None]] = None,
        on_status: Optional[Callable[[str, str], None]] = None,
        on_position: Optional[Callable[[str, str], None]] = None
    ):
        """
        初始化非同步 Pipeline
        
        Args:
            config_path: 音訊設定檔路徑
            model_config_path: 模型設定檔路徑
            robot_config_path: 機器人設定檔路徑
            use_wake_word: 是否使用喚醒詞模式
            use_mock_llm: 是否使用模擬 LLM
            use_mock_robot: 是否使用模擬機器人
            on_command: 指令回調函式
            on_status: 狀態變更回調
            on_position: 位置變更回調
        """
        self.use_wake_word = use_wake_word
        self.use_mock_llm = use_mock_llm
        self.use_mock_robot = use_mock_robot
        self.on_command = on_command
        self.on_status = on_status
        self.on_position = on_position
        
        # 載入設定
        self.audio_config = self._load_config(config_path)
        self.model_config = self._load_config(model_config_path)
        self.robot_config = self._load_config(robot_config_path)
        
        # 執行緒間通訊佇列
        self.asr_queue: queue.Queue = queue.Queue(maxsize=10)
        self.llm_queue: queue.Queue = queue.Queue(maxsize=10)
        self.tts_queue: queue.Queue = queue.Queue(maxsize=10)
        
        # 控制旗標
        self._is_running = False
        self._is_listening = False  # 喚醒後聆聽中
        self._is_llm_processing = False  # LLM 處理中
        self._last_wake_time = 0.0
        self._listen_timeout = 10.0
        
        # 中斷控制
        self._interrupt_llm = threading.Event()  # 中斷 LLM
        self._interrupt_tts = threading.Event()  # 中斷 TTS
        
        # 執行緒
        self._threads: list[threading.Thread] = []
        
        # 載具控制器
        self.thouzer = None
        
        # 初始化模組
        self._init_modules()
    
    def _load_config(self, path: str) -> dict:
        """載入 YAML 設定檔"""
        path = Path(path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}
    
    def _get_model_path(self, relative_path: str) -> Path:
        """
        將相對路徑轉換為絕對路徑
        
        Args:
            relative_path: 相對於 models/ 的路徑
        
        Returns:
            完整的絕對路徑
        """
        # 如果已經是絕對路徑，直接返回
        path = Path(relative_path)
        if path.is_absolute():
            return path
        
        # 否則，假設是相對於 models/ 目錄
        return Path("models") / relative_path
    
    
    def _init_modules(self) -> None:
        """初始化各模組"""
        print("=== 初始化非同步語音控制模組 ===")
        
        # 音訊設定
        audio_cfg = self.audio_config.get('audio', {})
        sample_rate = audio_cfg.get('sample_rate', 16000)
        
        # 1. 音訊輸入
        from src.audio.audio_input import AudioInput
        self.audio_input = AudioInput(
            sample_rate=sample_rate,
            channels=audio_cfg.get('channels', 1),
            chunk_ms=audio_cfg.get('chunk_ms', 100)
        )
        print("  [OK] AudioInput")
        
        # 2. 音訊輸出
        from src.audio.audio_output import AudioOutput
        self.audio_output = AudioOutput(sample_rate=22050)
        print("  [OK] AudioOutput")
        
        # 3. DNS 語音增強
        from src.audio.dns import create_dns
        dns_cfg = self.model_config.get('dns', {})
        backend = dns_cfg.get('backend', 'auto')
        self.dns = create_dns(
            backend=backend,
            sample_rate=sample_rate
        )
        
        # 顯示 DNS 實際使用的模型
        dns_model = 'Unknown'
        if hasattr(self.dns, '__class__'):
            class_name = self.dns.__class__.__name__
            if 'DeepFilterNet' in class_name:
                dns_model = 'DeepFilterNet3'
            elif 'Onnx' in class_name:
                dns_model = 'ONNX'
            elif 'Passthrough' in class_name:
                dns_model = 'Passthrough'
        
        print(f"  [{'OK' if self.dns.is_available else '--'}] DNS ({dns_model})")
        
        # 4. VAD
        from src.speech.vad import create_vad
        vad_cfg = self.model_config.get('vad', {})
        
        # 從 config 讀取模型路徑
        model_path = vad_cfg.get('model_path', '')
        vad_display = ''
        if model_path:
            full_model_path = self._get_model_path(model_path)
            vad_display = model_path
            self.vad = create_vad(
                model_path=str(full_model_path),
                threshold=vad_cfg.get('threshold', 0.5)
            )
        else:
            # Fallback: 使用預設路徑（models/vad/silero_vad.onnx）
            vad_display = 'silero_vad.onnx'
            self.vad = create_vad(
                models_dir="models",
                threshold=vad_cfg.get('threshold', 0.5)
            )
        
        print(f"  [{'OK' if self.vad and self.vad.is_available else '--'}] VAD ({vad_display})")
        
        # 5. KWS
        if self.use_wake_word:
            # 嘗試使用 Sherpa-ONNX KWS（效能更好）
            try:
                from src.speech.kws import KeywordSpotter
                kws_cfg = self.model_config.get('kws', {})
                
                # 從 model_config.yaml 讀取模型路徑
                model_path = self._get_model_path(kws_cfg.get('model_path', 'kws/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01'))
                
                # 自動偵測模型檔案（支援不同版本的模型）
                model_dir = Path(model_path)
                encoder_files = list(model_dir.glob("encoder-*.onnx"))
                decoder_files = list(model_dir.glob("decoder-*.onnx"))
                joiner_files = list(model_dir.glob("joiner-*.onnx"))
                
                # 優先選擇非 int8 版本的 chunk-16 模型
                def select_model(files, prefer_pattern="chunk-16"):
                    non_int8 = [f for f in files if "int8" not in f.name]
                    preferred = [f for f in non_int8 if prefer_pattern in f.name]
                    return str(preferred[0]) if preferred else str(non_int8[0]) if non_int8 else str(files[0])
                
                encoder_path = select_model(encoder_files) if encoder_files else str(model_dir / "encoder.onnx")
                decoder_path = select_model(decoder_files) if decoder_files else str(model_dir / "decoder.onnx")
                joiner_path = select_model(joiner_files) if joiner_files else str(model_dir / "joiner.onnx")
                tokens_path = str(model_dir / "tokens.txt")
                
                # 從 config 讀取 keywords_file（拼音格式）
                keywords_file_cfg = kws_cfg.get('keywords_file', '')
                keywords_file = str(self._get_model_path(keywords_file_cfg)) if keywords_file_cfg else None
                
                # 建立 KeywordSpotter
                self.kws = KeywordSpotter(
                    encoder_path=encoder_path,
                    decoder_path=decoder_path,
                    joiner_path=joiner_path,
                    tokens_path=tokens_path,
                    keywords=None,  # 使用 keywords_file
                    keywords_file=keywords_file,  # 從 config 讀取
                    threshold=kws_cfg.get('threshold', 0.25),
                    provider="cpu"  # sherpa-onnx 未編譯 GPU
                )
                
                # 顯示 KWS 資訊（單行）
                kws_model = kws_cfg.get('model_path', 'default').split('/')[-1]  # 只顯示最後一段
                print(f"  [{'OK' if self.kws.is_available else 'FAIL'}] KWS (Sherpa-ONNX: {kws_model})")
                
                # 如果 Sherpa-ONNX KWS 初始化失敗，fallback 到 SimpleKeywordMatcher
                if not self.kws.is_available:
                    raise Exception("Sherpa-ONNX KWS not available")
                    
            except Exception as e:
                print(f"  [WARN] Sherpa-ONNX KWS failed: {e}")
                print(f"  [INFO] Falling back to SimpleKeywordMatcher")
                
                # Fallback: 使用 SimpleKeywordMatcher
                from src.speech.kws import SimpleKeywordMatcher
                kws_cfg = self.audio_config.get('kws', {})
                keywords = kws_cfg.get('keywords', ['智護車', '護理車'])
                self.kws = SimpleKeywordMatcher(keywords=keywords)
                print(f"  [OK] KWS (Simple, keywords: {keywords})")
        else:
            self.kws = None
            print("  [--] KWS (disabled)")
        
        # 6. ASR
        from src.speech.asr import create_streaming_asr
        asr_cfg = self.model_config.get('asr', {}).get('streaming', {})
        
        # 從 config 讀取 encoder 路徑，推斷模型目錄
        encoder_path = asr_cfg.get('encoder', '')
        asr_display = ''
        if encoder_path:
            # 從 encoder 路徑推斷模型目錄
            full_encoder_path = self._get_model_path(encoder_path)
            model_dir = str(Path(full_encoder_path).parent)
            asr_display = str(Path(encoder_path).parent).split('/')[-1]  # 只顯示目錄名
            
            # 從 config 讀取其他模型路徑（如果有）
            decoder_cfg = asr_cfg.get('decoder', '')
            joiner_cfg = asr_cfg.get('joiner', '')
            tokens_cfg = asr_cfg.get('tokens', '')
            
            decoder_path = str(self._get_model_path(decoder_cfg)) if decoder_cfg else None
            joiner_path = str(self._get_model_path(joiner_cfg)) if joiner_cfg else None
            tokens_path = str(self._get_model_path(tokens_cfg)) if tokens_cfg else None
            
            self.asr = create_streaming_asr(
                model_dir=model_dir,
                encoder_path=str(full_encoder_path),
                decoder_path=decoder_path,
                joiner_path=joiner_path,
                tokens_path=tokens_path
            )
        else:
            # Fallback: glob 搜尋
            asr_dir = Path("models/asr")
            model_dirs = list(asr_dir.glob("*streaming*zipformer*"))
            if model_dirs:
                asr_display = model_dirs[0].name
                self.asr = create_streaming_asr(model_dir=str(model_dirs[0]))
            else:
                asr_display = 'default'
                self.asr = create_streaming_asr()
        
        print(f"  [{'OK' if self.asr and self.asr.is_available else '--'}] ASR ({asr_display})")
        
        # 7. TTS
        from src.speech.tts import create_tts
        tts_cfg = self.model_config.get('tts', {})
        
        # 從 config 讀取模型路徑
        model_path = tts_cfg.get('model_path', '')
        tts_display = ''
        if model_path:
            full_model_path = self._get_model_path(model_path)
            if Path(full_model_path).exists():
                tts_display = model_path.split('/')[-1]  # 只顯示最後一段
                self.tts = create_tts(model_dir=str(full_model_path), speed=0.8)
            else:
                # Fallback: glob 搜尋
                tts_dir = Path("models/tts")
                tts_dirs = list(tts_dir.glob("*vits*"))
                if tts_dirs:
                    tts_display = tts_dirs[0].name
                    self.tts = create_tts(model_dir=str(tts_dirs[0]), speed=0.8)
                else:
                    tts_display = 'default'
                    self.tts = create_tts(speed=0.6)
        else:
            # Fallback: glob 搜尋
            tts_dir = Path("models/tts")
            tts_dirs = list(tts_dir.glob("*vits*"))
            if tts_dirs:
                tts_display = tts_dirs[0].name
                self.tts = create_tts(model_dir=str(tts_dirs[0]), speed=0.8)
            else:
                tts_display = 'default'
                self.tts = create_tts(speed=0.6)
        
        print(f"  [{'OK' if self.tts and self.tts.is_available else '--'}] TTS ({tts_display}, speed=0.8x)")
        
        # 8. LLM
        self._init_llm()
        
        # 9. Thouzer 載具控制
        self._init_thouzer()
        
        print("=== 模組初始化完成 ===")
    
    def _init_llm(self) -> None:
        """初始化 LLM"""
        # 護理車使用動態 Prompt（每次呼叫時建立），不使用固定 system_prompt
        llm_cfg = self.model_config.get('llm', {})
        
        if self.use_mock_llm:
            from src.llm.unified import create_llm
            self.llm = create_llm(use_mock=True, system_prompt="")
        else:
            from src.llm.unified import create_llm
            llm_backend = llm_cfg.get('backend', 'llama_cpp')
            
            if llm_backend == 'llama_cpp':
                llama_cfg = llm_cfg.get('llama_cpp', {})
                model_path = llama_cfg.get('model_path', '')
                n_ctx = llama_cfg.get('n_ctx', 4096)
                
                self.llm = create_llm(
                    backend='llama_cpp',
                    model_path=model_path,
                    system_prompt="",  # 動態 Prompt
                    n_ctx=n_ctx
                )
                
                # 顯示 LLM 資訊（單行）
                llm_display = model_path.split('/')[-1] if model_path else 'default'
                print(f"  [{'OK' if self.llm.is_available else '--'}] LLM (llama.cpp: {llm_display}, ctx={n_ctx})")
                return
            else:
                self.llm = create_llm(use_mock=True, system_prompt="")
        
        print(f"  [{'OK' if self.llm.is_available else '--'}] LLM")
    
    def _init_thouzer(self) -> None:
        """初始化 Thouzer 載具控制器"""
        if self.use_mock_robot:
            self.thouzer = None
            print("  [--] Thouzer (mock mode)")
            return
        
        try:
            from src.robot.thouzer import ThouzerController
            
            # 建立回調包裝
            def on_status(status: str, message: str):
                print(f"[Thouzer] {status}: {message}")
                if self.on_status:
                    self.on_status(status, message)
            
            def on_position(code: str, name: str):
                print(f"[Thouzer] 位置更新: {name} ({code})")
                if self.on_position:
                    self.on_position(code, name)
            
            self.thouzer = ThouzerController(
                on_status_change=on_status,
                on_position_change=on_position
            )
            
            # 連接 MQTT
            if self.thouzer.connect():
                print(f"  [OK] Thouzer (position: {self.thouzer.current_position_name})")
            else:
                print("  [!!] Thouzer (MQTT connection failed)")
                
        except Exception as e:
            print(f"  [!!] Thouzer init error: {e}")
            self.thouzer = None
    
    # ========================================
    # 音訊處理執行緒
    # ========================================
    
    def _audio_worker(self) -> None:
        """
        音訊處理 Worker（執行緒 1）
        
        持續處理音訊，不受 LLM 影響。
        流程：音訊 → DNS → VAD → ASR → 放入佇列
        """
        print("[AudioWorker] Started")
        audio_queue: queue.Queue = queue.Queue()
        
        def audio_callback(chunk: np.ndarray):
            """音訊輸入回調"""
            try:
                audio_queue.put_nowait(chunk)
            except queue.Full:
                pass  # 忽略佇列滿的情況
        
        # 註冊回調
        self.audio_input.add_callback(audio_callback)
        self.audio_input.start()
        
        while self._is_running:
            try:
                # 取得音訊
                audio_chunk = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            # DNS 語音增強
            if self.dns.is_available:
                if hasattr(self.dns, 'enhance'):
                    audio_chunk = self.dns.enhance(audio_chunk, 16000)
                elif hasattr(self.dns, 'process'):
                    audio_chunk = self.dns.process(audio_chunk)
            
            # 喚醒詞模式檢查
            if self.use_wake_word and not self._is_listening:
                # 未喚醒：嘗試用 Sherpa-ONNX KWS 直接處理音訊
                if self.kws and hasattr(self.kws, 'process'):
                    # Sherpa-ONNX KeywordSpotter（直接處理音訊）
                    keyword = self.kws.process(audio_chunk)
                    if keyword:
                        self._on_wake(keyword)
                else:
                    # SimpleKeywordMatcher（需要 ASR 文字）
                    if self.asr and self.asr.is_available:
                        text, _ = self.asr.process(audio_chunk)
                        if text and self.kws:
                            kws_result = self.kws.check(text)
                            if kws_result:
                                # SimpleKeywordMatcher.check() 回傳 (keyword, remaining) tuple
                                if isinstance(kws_result, tuple):
                                    keyword, _ = kws_result
                                else:
                                    keyword = kws_result
                                self._on_wake(keyword)
            else:
                # 已喚醒或持續聆聽模式
                if self.asr and self.asr.is_available:
                    text, is_final = self.asr.process(audio_chunk)
                    
                    if is_final and text:
                        # 檢查是否為喚醒詞（可能是新的喚醒詞中斷）
                        # 只有 SimpleKeywordMatcher 有 check() 方法
                        if self.kws and hasattr(self.kws, 'check'):
                            kws_result = self.kws.check(text)
                            if kws_result:
                                keyword, remaining = kws_result
                                self._interrupt_current()
                                self._on_wake(keyword)
                                
                                # 如果喚醒詞後還有文字（如「護理車到七號病房」），直接處理
                                if remaining:
                                    print(f"[AudioWorker] 偵測到連續指令: {remaining}")
                                    # 簡體轉繁體
                                    remaining_traditional = self._convert_to_traditional(remaining)
                                    self._send_message(
                                        self.asr_queue,
                                        Message(MessageType.ASR_RESULT, remaining_traditional)
                                    )
                                    
                                    # 重置聆聽狀態（因為已經收到指令了）
                                    if self.use_wake_word:
                                        self._is_listening = False
                                        self.asr.reset()
                                        # 通知 UI 回到等待狀態
                                        if self.on_status:
                                            self.on_status('processing', None)
                                continue
                        
                        # 簡體轉繁體
                        text_traditional = self._convert_to_traditional(text)
                        
                        # 送入 ASR 佇列（繁體）
                        print(f"[AudioWorker] ASR: {text} → {text_traditional}")
                        self._send_message(
                            self.asr_queue,
                            Message(MessageType.ASR_RESULT, text_traditional)
                        )
                        
                        # 重置聆聽狀態
                        if self.use_wake_word:
                            self._is_listening = False
                            self.asr.reset()
                            # 通知 UI 進入處理狀態
                            if self.on_status:
                                self.on_status('processing', text_traditional)
            
            # 檢查聆聽超時
            if self._is_listening:
                if time.time() - self._last_wake_time > self._listen_timeout:
                    print("[AudioWorker] 聆聽超時")
                    self._is_listening = False
                    if self.asr:
                        self.asr.reset()
                    # 通知 UI 回到等待狀態
                    if self.on_status:
                        self.on_status('idle', None)
        
        self.audio_input.stop()
        print("[AudioWorker] Stopped")
    
    def _on_wake(self, keyword: str) -> None:
        """喚醒詞偵測"""
        print(f"[AudioWorker] 喚醒詞: {keyword}")
        self._is_listening = True
        self._last_wake_time = time.time()
        
        if self.asr:
            self.asr.reset()
        
        # 通知 UI 已喚醒（用於狀態指示燈）
        if self.on_status:
            self.on_status('wake', keyword)
        
        # 送 TTS 提示
        self._send_message(
            self.tts_queue,
            Message(MessageType.TTS_REQUEST, "我在，請說")
        )
    
    def _interrupt_current(self) -> None:
        """中斷當前處理"""
        self._interrupt_llm.set()
        self._interrupt_tts.set()
        
        # 清空佇列
        self._clear_queue(self.asr_queue)
        self._clear_queue(self.llm_queue)
        self._clear_queue(self.tts_queue)
    
    # ========================================
    # 工具函式
    # ========================================
    
    def _convert_numbers_to_chinese(self, text: str) -> str:
        """
        將阿拉伯數字轉換為中文數字（TTS 專用）
        
        避免 sherpa-onnx TTS 的 OOV 錯誤
        """
        import re
        
        # 數字對照表
        digit_map = {
            '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
            '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
        }
        
        # 常見組合直接替換
        number_map = {
            '10': '十', '11': '十一', '12': '十二', '13': '十三', '14': '十四',
            '15': '十五', '16': '十六', '17': '十七', '18': '十八', '19': '十九',
            '01': '一', '02': '二', '03': '三', '04': '四', '05': '五',
            '06': '六', '07': '七', '08': '八', '09': '九'
        }
        
        # 先處理常見組合（如 10號 -> 十號）
        for num, cn in sorted(number_map.items(), key=lambda x: -len(x[0])):
            text = text.replace(num, cn)
        
        # 再處理單個數字
        for digit, cn in digit_map.items():
            text = text.replace(digit, cn)
        
        return text
    
    # ========================================
    # LLM 處理執行緒
    # ========================================
    
    def _llm_worker(self) -> None:
        """
        LLM 處理 Worker（執行緒 2）
        
        從 ASR 佇列接收文字，生成回應，送入 TTS 佇列。
        支援中斷。
        """
        print("[LLMWorker] Started")
        
        # 導入指令解析
        from src.robot.nursing_commands import parse_nursing_command
        
        while self._is_running:
            try:
                msg = self.asr_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            if msg.type == MessageType.STOP:
                break
            
            if msg.type != MessageType.ASR_RESULT:
                continue
            
            user_input = msg.data
            print(f"[LLMWorker] Processing: {user_input}")
            
            # 重置中斷旗標
            self._interrupt_llm.clear()
            self._is_llm_processing = True
            
            try:
                # 取得當前位置（用於動態 Prompt）
                current_code = "O"
                current_name = "護理站"
                if self.thouzer:
                    current_code = self.thouzer.current_position
                    current_name = self.thouzer.current_position_name
                
                # 建立動態 Prompt
                from src.robot.nursing_commands import build_nursing_cart_prompt
                prompt = build_nursing_cart_prompt(current_code, current_name, user_input)
                
                # 呼叫 LLM
                response = self.llm.chat(prompt)
                
                if self._interrupt_llm.is_set():
                    print("[LLMWorker] Interrupted")
                    continue
                
                print(f"[LLMWorker] Response: {response}")
                
                # 解析護理車指令
                action = parse_nursing_command(response)
                
                if action:
                    # 自動修正 route：不依賴 LLM 輸出，用 current_position + target 計算
                    original_route = action.route
                    if action.is_navigation() and action.target and self.thouzer:
                        correct_route = f"{self.thouzer.current_position}{action.target}"
                        action.route = correct_route  # 修正 route
                        if original_route != correct_route:
                            print(f"[LLMWorker] Route 修正: {original_route} → {correct_route}")
                    
                    print(f"[LLMWorker] Action: type={action.type}, route={action.route}, target={action.target}")
                    
                    if self.thouzer:
                        if action.is_navigation():
                            # 導航指令：使用 target
                            dest = action.target or action.end_code
                            if dest:
                                success = self.thouzer.navigate_to(dest)
                                dest_name = self.thouzer.station_mapper.code_to_name(dest)
                                print(f"[LLMWorker] Navigate to {dest_name}: {'OK' if success else 'FAIL'}")
                        
                        elif action.is_follow():
                            # 跟隨指令
                            if action.action == "START":
                                success = self.thouzer.start_follow()
                            else:
                                success = self.thouzer.stop_follow()
                            print(f"[LLMWorker] Follow {action.action}: {'OK' if success else 'FAIL'}")
                    
                    # 回調
                    if self.on_command:
                        self.on_command(action.to_dict())
                    
                    # TTS 回應
                    tts_text = action.text or response
                else:
                    tts_text = response
                
                # 送入 TTS 佇列
                self._send_message(
                    self.tts_queue,
                    Message(MessageType.TTS_REQUEST, tts_text)
                )
                
            except Exception as e:
                print(f"[LLMWorker] Error: {e}")
            finally:
                self._is_llm_processing = False
                # 通知 UI 回到待機狀態
                if self.on_status:
                    self.on_status('idle', None)
        
        print("[LLMWorker] Stopped")
    
    def _parse_command(self, text: str) -> Optional[dict]:
        """解析 LLM 回應中的指令"""
        import json
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None
    
    # ========================================
    # TTS 輸出執行緒
    # ========================================
    
    def _tts_worker(self) -> None:
        """
        TTS 輸出 Worker（執行緒 3）
        
        從 TTS 佇列接收文字，合成語音並播放。
        支援中斷。
        """
        print("[TTSWorker] Started")
        
        while self._is_running:
            try:
                msg = self.tts_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            if msg.type == MessageType.STOP:
                break
            
            if msg.type != MessageType.TTS_REQUEST:
                continue
            
            text = msg.data
            
            # 重置中斷旗標
            self._interrupt_tts.clear()
            
            if False and self.tts and self.tts.is_available and text:  # TTS 暫時關閉
                # 將阿拉伯數字轉換成中文數字（避免 TTS OOV 錯誤）
                text = self._convert_numbers_to_chinese(text)
                print(f"[TTSWorker] Synthesizing: {text}")
                
                try:
                    audio = self.tts.synthesize(text)
                    
                    if self._interrupt_tts.is_set():
                        print("[TTSWorker] Interrupted")
                        continue
                    
                    if len(audio) > 0:
                        # 非阻塞播放
                        self.audio_output.play(audio, blocking=False)
                        
                except Exception as e:
                    print(f"[TTSWorker] Error: {e}")
        
        print("[TTSWorker] Stopped")
    
    # ========================================
    # 工具方法
    # ========================================
    
    def _send_message(self, q: queue.Queue, msg: Message) -> bool:
        """送入佇列（非阻塞）"""
        try:
            q.put_nowait(msg)
            return True
        except queue.Full:
            return False
    
    def _clear_queue(self, q: queue.Queue) -> None:
        """清空佇列"""
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break
    
    def _convert_to_traditional(self, text: str) -> str:
        """
        簡體中文轉繁體中文，並修正 ASR 常見誤辨識
        
        Args:
            text: 簡體中文文字
        
        Returns:
            繁體中文文字（已修正誤辨識）
        """
        # 1. 簡體轉繁體
        try:
            from src.utils.s2t import s2t
            result = s2t(text)
        except ImportError:
            result = text
        
        # 2. 拼音校正（自動校正同音誤辨識）
        try:
            from src.utils.pinyin_corrector import correct_pinyin
            result = correct_pinyin(result)
        except ImportError:
            pass
        
        # 3. 字典補充校正（拼音校正可能漏掉的特殊情況）
        asr_corrections = {
            "跟水": "跟隨", "根水": "跟隨",
            "跟谁": "跟隨", "跟誰": "跟隨",
            "根随": "跟隨", "根隨": "跟隨",
        }
        for wrong, correct in asr_corrections.items():
            result = result.replace(wrong, correct)
        
        return result
    
    # ========================================
    # 控制方法
    # ========================================
    
    def start(self) -> None:
        """啟動 Pipeline"""
        if self._is_running:
            return
        
        print("\n=== 啟動非同步語音控制 ===")
        self._is_running = True
        
        # 啟動 Worker 執行緒
        workers = [
            ("AudioWorker", self._audio_worker),
            ("LLMWorker", self._llm_worker),
            ("TTSWorker", self._tts_worker),
        ]
        
        for name, target in workers:
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        
        if self.use_wake_word:
            keywords = self.kws.keywords if self.kws else ["智護車", "護理車"]
            print(f"等待喚醒詞: {keywords}")
        else:
            self._is_listening = True
            print("持續聆聽模式")
    
    def stop(self) -> None:
        """停止 Pipeline"""
        if not self._is_running:
            return
        
        print("\n=== 停止非同步語音控制 ===")
        self._is_running = False
        
        # 送停止信號
        stop_msg = Message(MessageType.STOP, None)
        self._send_message(self.asr_queue, stop_msg)
        self._send_message(self.tts_queue, stop_msg)
        
        # 等待執行緒結束
        for t in self._threads:
            t.join(timeout=2.0)
        
        self._threads.clear()
    
    def run(self) -> None:
        """執行 Pipeline（阻塞式）"""
        self.start()
        
        try:
            while self._is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n收到中斷信號")
        finally:
            self.stop()
    
    @property
    def is_processing(self) -> bool:
        """是否正在處理 LLM"""
        return self._is_llm_processing


def main():
    """主程式進入點"""
    import argparse
    
    parser = argparse.ArgumentParser(description='護理輔助車語音控制系統')
    parser.add_argument('--no-wake', action='store_true',
                        help='停用喚醒詞，持續聆聽')
    parser.add_argument('--mock-llm', action='store_true',
                        help='使用模擬 LLM（測試用）')
    parser.add_argument('--mock-robot', action='store_true',
                        help='使用模擬機器人（不連接 MQTT）')
    parser.add_argument('--test', action='store_true',
                        help='執行測試模式')
    args = parser.parse_args()
    
    # 指令回調
    def on_command(cmd: dict):
        print(f"[Command] {cmd}")
    
    # 狀態回調
    def on_status(status: str, msg: str):
        print(f"[Status] {status}: {msg}")
    
    # 位置回調
    def on_position(code: str, name: str):
        print(f"[Position] {code}: {name}")
    
    # 建立 Pipeline
    pipeline = AsyncVoiceControlPipeline(
        use_wake_word=not args.no_wake,
        use_mock_llm=args.mock_llm or args.test,
        use_mock_robot=args.mock_robot or args.test,
        on_command=on_command,
        on_status=on_status,
        on_position=on_position
    )
    
    if args.test:
        print("\n=== 測試模式 ===")
        print("模組狀態:")
        print(f"  DNS: {pipeline.dns.is_available}")
        print(f"  VAD: {pipeline.vad.is_available if pipeline.vad else False}")
        print(f"  ASR: {pipeline.asr.is_available if pipeline.asr else False}")
        print(f"  TTS: {pipeline.tts.is_available if pipeline.tts else False}")
        print(f"  LLM: {pipeline.llm.is_available}")
        print(f"  Thouzer: {pipeline.thouzer is not None}")
        if pipeline.thouzer:
            print(f"    Position: {pipeline.thouzer.current_position_name}")
            print(f"    MQTT: {pipeline.thouzer.is_connected}")
        return
    
    # 執行
    pipeline.run()


if __name__ == "__main__":
    main()

