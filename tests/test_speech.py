"""
語音處理模組測試

測試 VAD、KWS、ASR、TTS 各模組功能。
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# 確保可以導入 src 模組
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_vad():
    """測試 VAD 模組"""
    print("\n=== VAD 測試 ===")
    
    from src.speech.vad import create_vad, SHERPA_AVAILABLE
    
    if not SHERPA_AVAILABLE:
        print("sherpa_onnx 未安裝")
        return False
    
    vad = create_vad(models_dir="models")
    
    if vad and vad.is_available:
        print("VAD 初始化成功")
        
        # 測試處理
        silence = np.zeros(16000, dtype=np.float32)
        segments = vad.process(silence)
        print(f"處理 1 秒靜音: {len(segments)} 個語音片段")
        return True
    else:
        print("VAD 模型未載入，請先下載模型")
        return False


def test_asr():
    """測試 ASR 模組"""
    print("\n=== ASR 測試 (Streaming Zipformer) ===")
    
    from src.speech.asr import create_streaming_asr, SHERPA_AVAILABLE
    
    if not SHERPA_AVAILABLE:
        print("sherpa_onnx 未安裝")
        return False
    
    # 尋找模型
    asr_dir = Path("models/asr")
    model_dirs = list(asr_dir.glob("*streaming*zipformer*"))
    
    if not model_dirs:
        print("ASR 模型未找到，請先下載模型")
        return False
    
    asr = create_streaming_asr(model_dir=str(model_dirs[0]))
    
    if asr and asr.is_available:
        print(f"ASR 初始化成功: {model_dirs[0].name}")
        
        # 測試處理
        silence = np.zeros(16000, dtype=np.float32)
        text, is_final = asr.process(silence)
        print(f"處理 1 秒靜音: '{text}' (final={is_final})")
        return True
    else:
        print("ASR 初始化失敗")
        return False


def test_tts():
    """測試 TTS 模組"""
    print("\n=== TTS 測試 ===")
    
    from src.speech.tts import create_tts, SHERPA_AVAILABLE
    
    if not SHERPA_AVAILABLE:
        print("sherpa_onnx 未安裝")
        return False
    
    # 尋找模型
    tts_dir = Path("models/tts")
    model_dirs = list(tts_dir.glob("*vits*"))
    
    if not model_dirs:
        print("TTS 模型未找到，請先下載模型")
        return False
    
    tts = create_tts(model_dir=str(model_dirs[0]))
    
    if tts and tts.is_available:
        print(f"TTS 初始化成功: {model_dirs[0].name}")
        print(f"說話者數量: {tts.num_speakers}")
        
        # 測試合成
        test_text = "你好，我是智護車"
        audio = tts.synthesize(test_text)
        
        if len(audio) > 0:
            duration = len(audio) / tts.sample_rate
            print(f"合成 '{test_text}': {len(audio)} 樣本, {duration:.2f} 秒")
            return True
        else:
            print("合成失敗")
            return False
    else:
        print("TTS 初始化失敗")
        return False


def test_kws():
    """測試 KWS 模組"""
    print("\n=== KWS 測試 ===")
    
    from src.speech.kws import SimpleKeywordMatcher
    
    matcher = SimpleKeywordMatcher(keywords=["智護車", "護理車"])
    
    test_texts = [
        "你好智護車請幫我",
        "護理車請過來",
        "今天天氣很好"
    ]
    
    for text in test_texts:
        result = matcher.check(text)
        status = f"偵測到: {result}" if result else "無喚醒詞"
        print(f"  '{text}' -> {status}")
    
    return True


def test_audio():
    """測試音訊模組"""
    print("\n=== 音訊測試 ===")
    
    from src.audio.audio_input import AudioInput
    from src.audio.audio_output import AudioOutput
    
    # 列出裝置
    print("音訊輸入裝置:")
    for device in AudioInput.list_devices():
        print(f"  [{device['index']}] {device['name']}")
    
    print("\n音訊輸出裝置:")
    for device in AudioOutput.list_devices():
        print(f"  [{device['index']}] {device['name']}")
    
    return True


def test_llm():
    """測試 LLM 模組"""
    print("\n=== LLM 測試 (Mock) ===")
    
    from src.llm.nano_llm import create_llm
    
    llm = create_llm(use_mock=True)
    
    test_inputs = ["請向前走", "左轉", "停止"]
    
    for user_input in test_inputs:
        response = llm.chat(user_input)
        print(f"  User: {user_input}")
        print(f"  LLM:  {response}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='語音處理模組測試')
    parser.add_argument('--module', choices=['vad', 'asr', 'tts', 'kws', 'audio', 'llm', 'all'],
                        default='all', help='要測試的模組')
    args = parser.parse_args()
    
    results = {}
    
    if args.module in ['audio', 'all']:
        results['audio'] = test_audio()
    
    if args.module in ['vad', 'all']:
        results['vad'] = test_vad()
    
    if args.module in ['kws', 'all']:
        results['kws'] = test_kws()
    
    if args.module in ['asr', 'all']:
        results['asr'] = test_asr()
    
    if args.module in ['tts', 'all']:
        results['tts'] = test_tts()
    
    if args.module in ['llm', 'all']:
        results['llm'] = test_llm()
    
    # 顯示結果
    print("\n=== 測試結果 ===")
    for module, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {module}: {status}")


if __name__ == "__main__":
    main()
