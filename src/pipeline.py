"""
語音控制整合流程 (Pipeline)

整合所有模組的完整語音控制流程：
麥克風 -> DNS -> VAD -> KWS/ASR -> LLM -> TTS/Robot

設計說明：
1. 使用事件驅動架構，各模組透過 callback 連接
2. 支援兩種模式：
   - 喚醒詞模式：偵測到喚醒詞後才開始辨識
   - 持續模式：持續辨識（測試用）
3. ASR 使用 streaming_zipformer 即時處理音訊
"""
import threading
import queue
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import yaml

# 導入各模組
from src.audio.audio_input import AudioInput
from src.audio.audio_output import AudioOutput
from src.audio.dns import create_dns
from src.speech.vad import VoiceActivityDetector, create_vad
from src.speech.kws import KeywordSpotter, SimpleKeywordMatcher, create_kws
from src.speech.asr import StreamingASR, create_streaming_asr
from src.speech.tts import TextToSpeech, create_tts
# LLM 在 _init_modules 中動態導入，支援 llama_cpp 和 nanollm 雙後端


class VoiceControlPipeline:
    """
    語音控制整合流程
    
    整合音訊輸入、語音處理、LLM 和語音輸出。
    """
    
    def __init__(
        self,
        config_path: str = "config/audio_config.yaml",
        model_config_path: str = "config/model_config.yaml",
        use_wake_word: bool = True,
        use_mock_llm: bool = False,
        on_command: Optional[Callable[[dict], None]] = None
    ):
        """
        初始化 Pipeline
        
        Args:
            config_path: 音訊設定檔路徑
            model_config_path: 模型設定檔路徑
            use_wake_word: 是否使用喚醒詞模式
            use_mock_llm: 是否使用模擬 LLM
            on_command: 指令回調函式（用於機器人控制）
        """
        self.use_wake_word = use_wake_word
        self.use_mock_llm = use_mock_llm
        self.on_command = on_command
        
        # 載入設定
        self.audio_config = self._load_config(config_path)
        self.model_config = self._load_config(model_config_path)
        
        # 狀態
        self._is_running = False
        self._is_listening = False  # 喚醒後是否正在聆聽
        self._listen_timeout = 10.0  # 喚醒後聆聽時間
        self._last_wake_time = 0.0
        
        # 音訊佇列
        self._audio_queue: queue.Queue = queue.Queue()
        
        # 初始化各模組
        self._init_modules()
    
    def _load_config(self, path: str) -> dict:
        """載入 YAML 設定檔"""
        path = Path(path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}
    
    def _init_modules(self) -> None:
        """初始化各模組"""
        print("=== 初始化語音控制模組 ===")
        
        # 音訊設定
        audio_cfg = self.audio_config.get('audio', {})
        sample_rate = audio_cfg.get('sample_rate', 16000)
        
        # 1. 音訊輸入
        self.audio_input = AudioInput(
            sample_rate=sample_rate,
            channels=audio_cfg.get('channels', 1),
            chunk_ms=audio_cfg.get('chunk_ms', 100)
        )
        print("  [OK] AudioInput")
        
        # 2. 音訊輸出
        self.audio_output = AudioOutput(sample_rate=22050)
        print("  [OK] AudioOutput")
        
        # 3. DNS 語音增強（DeepFilterNet 或 ONNX）
        dns_cfg = self.model_config.get('dns', {})
        dns_backend = dns_cfg.get('backend', 'auto')
        dns_streaming = dns_cfg.get('deepfilternet', {}).get('streaming', True)
        onnx_path = dns_cfg.get('onnx', {}).get('model_path')
        
        self.dns = create_dns(
            backend=dns_backend,
            model_path=onnx_path,
            sample_rate=sample_rate,
            streaming=dns_streaming
        )
        
        dns_type = type(self.dns).__name__
        print(f"  [{'OK' if self.dns.is_available else '--'}] DNS ({dns_type})")
        
        # 4. VAD
        vad_cfg = self.audio_config.get('vad', {})
        vad_model = self.model_config.get('vad', {}).get('model_path')
        self.vad = create_vad(
            model_path=vad_model,
            threshold=vad_cfg.get('threshold', 0.5),
            min_silence_duration=vad_cfg.get('silence_duration', 0.5)
        )
        print(f"  [{'OK' if self.vad and self.vad.is_available else '--'}] VAD")
        
        # 5. KWS 喚醒詞
        if self.use_wake_word:
            kws_cfg = self.audio_config.get('kws', {})
            keywords = kws_cfg.get('keywords', ['智護車', '護理車'])
            
            # 嘗試使用完整 KWS，失敗則用簡易比對器
            self.kws = create_kws(keywords=keywords)
            if not self.kws.is_available:
                self.kws = SimpleKeywordMatcher(keywords=keywords)
            print(f"  [OK] KWS (keywords: {keywords})")
        else:
            self.kws = None
            print("  [--] KWS (disabled)")
        
        # 6. ASR (Streaming Zipformer)
        asr_cfg = self.model_config.get('asr', {}).get('streaming', {})
        asr_dir = Path("models/asr")
        
        # 尋找模型目錄
        asr_model_dirs = list(asr_dir.glob("*streaming*zipformer*"))
        if asr_model_dirs:
            self.asr = create_streaming_asr(model_dir=str(asr_model_dirs[0]))
        else:
            self.asr = create_streaming_asr()
        print(f"  [{'OK' if self.asr and self.asr.is_available else '--'}] ASR (Streaming)")
        
        # 7. TTS
        tts_dir = Path("models/tts")
        tts_model_dirs = list(tts_dir.glob("*vits*"))
        if tts_model_dirs:
            self.tts = create_tts(model_dir=str(tts_model_dirs[0]))
        else:
            self.tts = create_tts()
        print(f"  [{'OK' if self.tts and self.tts.is_available else '--'}] TTS")
        
        # 8. LLM (支援 llama.cpp 和 NanoLLM 雙後端)
        llm_cfg = self.model_config.get('llm', {})
        llm_backend = llm_cfg.get('backend', 'llama_cpp')
        
        # 取得系統提示詞
        system_prompt = llm_cfg.get('system_prompt', """你是護理機器人「智護車」的語音控制助手。
用戶會給你語音指令，你需要理解並回應。
若指令涉及移動，請用 JSON 格式輸出控制命令。""")
        
        # 根據後端設定建立 LLM
        if self.use_mock_llm:
            from src.llm.unified import create_llm
            self.llm = create_llm(use_mock=True, system_prompt=system_prompt)
        elif llm_backend == 'llama_cpp':
            from src.llm.unified import create_llm
            llama_cfg = llm_cfg.get('llama_cpp', {})
            model_path = llama_cfg.get('model_path', '')
            
            # 處理相對路徑
            if model_path and not model_path.startswith('/'):
                model_path = str(Path(model_path))
            
            self.llm = create_llm(
                backend='llama_cpp',
                model_path=model_path,
                system_prompt=system_prompt,
                n_ctx=llama_cfg.get('n_ctx', 4096),
                n_gpu_layers=llama_cfg.get('n_gpu_layers', -1)
            )
        else:
            # NanoLLM 後端
            from src.llm.unified import create_llm
            nanollm_cfg = llm_cfg.get('nanollm', {})
            self.llm = create_llm(
                backend='nanollm',
                model_name=nanollm_cfg.get('default_model', 'chinese-alpaca-2-7b'),
                system_prompt=system_prompt
            )
        
        llm_type = llm_backend if not self.use_mock_llm else 'mock'
        print(f"  [{'OK' if self.llm.is_available else '--'}] LLM ({llm_type})")
        
        print("=== 模組初始化完成 ===")
    
    def _process_audio(self, audio_chunk: np.ndarray) -> None:
        """
        處理音訊 chunk（主處理流程）
        
        即時接收音訊，逐幀處理。
        """
        # 1. DNS 語音增強
        if self.dns.is_available:
            # 支援 DeepFilterNet 和 ONNX 兩種介面
            if hasattr(self.dns, 'enhance'):
                audio_chunk = self.dns.enhance(audio_chunk, 16000)
            elif hasattr(self.dns, 'process'):
                audio_chunk = self.dns.process(audio_chunk)
        
        # 2. 放入佇列
        self._audio_queue.put(audio_chunk)
    
    def _processing_loop(self) -> None:
        """
        處理迴圈（在獨立執行緒中運行）
        """
        accumulated_audio = []
        
        while self._is_running:
            try:
                # 取得音訊 chunk
                audio_chunk = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            # 檢查喚醒詞模式
            if self.use_wake_word and not self._is_listening:
                # 未喚醒狀態：偵測喚醒詞
                if self.kws:
                    if isinstance(self.kws, SimpleKeywordMatcher):
                        # 簡易比對器需要 ASR 結果
                        if self.asr and self.asr.is_available:
                            text, _ = self.asr.process(audio_chunk)
                            if text:
                                keyword = self.kws.check(text)
                                if keyword:
                                    self._on_wake(keyword)
                    else:
                        # 完整 KWS
                        keyword = self.kws.process(audio_chunk)
                        if keyword:
                            self._on_wake(keyword)
            else:
                # 聆聽狀態：進行語音辨識
                if self.asr and self.asr.is_available:
                    text, is_final = self.asr.process(audio_chunk)
                    
                    if is_final and text:
                        print(f"[ASR] {text}")
                        self._process_command(text)
                        
                        # 重置聆聽狀態
                        if self.use_wake_word:
                            self._is_listening = False
                            self.asr.reset()
            
            # 檢查聆聽超時
            if self._is_listening:
                if time.time() - self._last_wake_time > self._listen_timeout:
                    print("[Pipeline] 聆聽超時，等待喚醒...")
                    self._is_listening = False
                    self.asr.reset()
    
    def _on_wake(self, keyword: str) -> None:
        """喚醒詞偵測回調"""
        print(f"[KWS] 偵測到喚醒詞: {keyword}")
        self._is_listening = True
        self._last_wake_time = time.time()
        
        # 重置 ASR
        if self.asr:
            self.asr.reset()
        
        # 播放提示音或語音
        if self.tts and self.tts.is_available:
            audio = self.tts.synthesize("我在，請說")
            if len(audio) > 0:
                self.audio_output.play(audio, blocking=False)
    
    def _process_command(self, text: str) -> None:
        """處理語音指令"""
        # 送給 LLM 處理
        response = self.llm.chat(text)
        print(f"[LLM] {response}")
        
        # 嘗試解析 JSON 指令
        try:
            import json
            command = json.loads(response)
            
            # 回調機器人控制
            if self.on_command:
                self.on_command(command)
            
            # 語音回應
            message = command.get('message', '好的')
            
        except json.JSONDecodeError:
            # 非 JSON 回應，直接作為語音輸出
            message = response
        
        # TTS 輸出
        if self.tts and self.tts.is_available and message:
            audio = self.tts.synthesize(message)
            if len(audio) > 0:
                self.audio_output.play(audio, blocking=True)
    
    def start(self) -> None:
        """啟動 pipeline"""
        if self._is_running:
            return
        
        print("\n=== 啟動語音控制 ===")
        self._is_running = True
        
        # 啟動處理執行緒
        self._process_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )
        self._process_thread.start()
        
        # 啟動音訊輸入
        self.audio_input.add_callback(self._process_audio)
        self.audio_input.start()
        
        if self.use_wake_word:
            print(f"等待喚醒詞: {self.kws.keywords if hasattr(self.kws, 'keywords') else '智護車/護理車'}")
        else:
            print("持續聆聽模式")
    
    def stop(self) -> None:
        """停止 pipeline"""
        if not self._is_running:
            return
        
        print("\n=== 停止語音控制 ===")
        self._is_running = False
        
        # 停止音訊輸入
        self.audio_input.stop()
        
        # 等待處理執行緒結束
        if hasattr(self, '_process_thread'):
            self._process_thread.join(timeout=2.0)
    
    def run(self) -> None:
        """執行 pipeline（阻塞式）"""
        self.start()
        
        try:
            while self._is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n收到中斷信號")
        finally:
            self.stop()


def main():
    """主程式進入點"""
    import argparse
    
    parser = argparse.ArgumentParser(description='語音控制機器人')
    parser.add_argument('--no-wake', action='store_true', 
                        help='停用喚醒詞，持續聆聽')
    parser.add_argument('--mock-llm', action='store_true',
                        help='使用模擬 LLM（測試用）')
    parser.add_argument('--test', action='store_true',
                        help='執行測試模式')
    args = parser.parse_args()
    
    # 指令回調（測試用）
    def on_command(cmd: dict):
        print(f"[Robot] 收到指令: {cmd}")
    
    # 建立 pipeline
    pipeline = VoiceControlPipeline(
        use_wake_word=not args.no_wake,
        use_mock_llm=args.mock_llm or args.test,
        on_command=on_command
    )
    
    if args.test:
        # 測試模式：只檢查模組
        print("\n=== 測試模式 ===")
        print("模組狀態:")
        print(f"  DNS: {pipeline.dns.is_available}")
        print(f"  VAD: {pipeline.vad.is_available if pipeline.vad else False}")
        print(f"  ASR: {pipeline.asr.is_available if pipeline.asr else False}")
        print(f"  TTS: {pipeline.tts.is_available if pipeline.tts else False}")
        print(f"  LLM: {pipeline.llm.is_available}")
        return
    
    # 執行
    pipeline.run()


if __name__ == "__main__":
    main()
