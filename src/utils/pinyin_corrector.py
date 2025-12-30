"""
拼音校正模組

使用拼音比對來校正 ASR 同音誤辨識。
例如：「惡號病房」→「二號病房」（拼音都是 è hào bìng fáng）
"""

from typing import Dict, List, Optional, Tuple
import re

# 嘗試導入 pypinyin
try:
    from pypinyin import pinyin, Style
    PYPINYIN_AVAILABLE = True
except ImportError:
    PYPINYIN_AVAILABLE = False
    print("Warning: pypinyin not installed, pinyin correction disabled")


class PinyinCorrector:
    """
    拼音校正器
    
    將 ASR 輸出中的同音誤辨識詞彙校正為正確詞彙。
    """
    
    def __init__(self):
        """初始化拼音校正器"""
        # 正確詞彙 → 拼音對照表
        # key: 正確詞彙, value: 拼音（用空格分隔）
        self._correct_words: Dict[str, str] = {}
        
        # 拼音 → 正確詞彙（反向查詢表）
        self._pinyin_to_word: Dict[str, str] = {}
        
        # 初始化預設詞彙
        self._init_default_words()
    
    def _init_default_words(self) -> None:
        """初始化預設詞彙庫"""
        
        # 直接映射表（處理拼音不完全相同的誤辨識）
        # 這些是 ASR 常見的誤辨識，需要直接替換
        self._direct_mappings = {
            # 複合數字誤辨識（十幾號的連讀問題）
            "時候病房": "十號病房", "時候號": "十號", "时候": "十号",
            "时尚号": "十三號", "時尚號": "十三號", "十三个": "十三號",
            "时候四号": "十四號", "時候四號": "十四號", 
            "時間四號": "十四號", "时间四号": "十四號",
            "时候五号": "十五號", "時候五號": "十五號",
            "時間五號": "十五號", "时间五号": "十五號",
            "时候六号": "十六號", "時候六號": "十六號",
            "时候七号": "十七號", "時候七號": "十七號",
            "时候八号": "十八號", "時候八號": "十八號",
            "时候九号": "十九號", "時候九號": "十九號",
            "十呃号": "十二號", "十惡號": "十二號", "十餓號": "十二號",
            "十一号": "十一號", "十二号": "十二號", "十三号": "十三號",
            "十四号": "十四號", "十五号": "十五號", "十六号": "十六號",
            "十七号": "十七號", "十八号": "十八號", "十九号": "十九號",
            "二十号": "二十號",
            
            # 單一數字誤辨識（病房號碼）
            "惡號": "二號", "噩耗": "二號", "餓號": "二號", "呃號": "二號",
            "醫號": "一號", "衣號": "一號", "依號": "一號",
            "散號": "三號", "傘號": "三號",
            "死號": "四號", "寺號": "四號", "似號": "四號",
            "午號": "五號", "舞號": "五號", "武號": "五號",
            "溜號": "六號", "流號": "六號", "留號": "六號",
            "其號": "七號", "騎號": "七號", "奇號": "七號",
            "拔號": "八號", "霸號": "八號",
            "酒號": "九號", "久號": "九號", "舊號": "九號",
            "石號": "十號", "實號": "十號", "時號": "十號",
            
            # 護理站相關
            "為護理戰": "回護理站", "微護理站": "回護理站", 
            "圍護理站": "回護理站", "維護理站": "回護理站",
            "護理戰": "護理站", "護立站": "護理站", "護利站": "護理站",
            "物理站": "護理站", "物理戰": "護理站",
            
            # 智護車相關
            "知護車": "智護車", "知後車": "智護車", "知乎車": "智護車",
            "自護車": "智護車", "自顧車": "智護車", "自物車": "智護車",
            "治步車": "智護車", "智步車": "智護車",
            
            # 護理車相關
            "物理車": "護理車", "烏里車": "護理車",
            
            # 動作指令
            "跟水": "跟隨", "根水": "跟隨", "根隨": "跟隨",
            "停支": "停止", "停滯": "停止",
            "繼許": "繼續", "機許": "繼續",
        }
        
        # 護理車常用詞彙（用於拼音比對）
        default_words = [
            # 病房號碼（1-20）
            "一號病房", "二號病房", "三號病房", "四號病房", "五號病房",
            "六號病房", "七號病房", "八號病房", "九號病房", "十號病房",
            "十一號病房", "十二號病房", "十三號病房", "十四號病房", "十五號病房",
            "十六號病房", "十七號病房", "十八號病房", "十九號病房", "二十號病房",
            
            # 簡稱
            "一號", "二號", "三號", "四號", "五號",
            "六號", "七號", "八號", "九號", "十號",
            "十一號", "十二號", "十三號", "十四號", "十五號",
            "十六號", "十七號", "十八號", "十九號", "二十號",
            
            # 地點
            "護理站", "護理車", "智護車",
            "回護理站", "到護理站", "去護理站",
            "藥局", "手術室", "急診室", "檢驗室",
            
            # 動作指令
            "跟隨", "跟我走", "跟著我",
            "停止", "停下來", "停",
            "前進", "後退", "左轉", "右轉",
            "繼續", "取消", "確認",
            "導航到", "到", "去",
            
            # 常用語
            "護理車", "智護車",
            "病房", "病人", "患者",
            "藥物", "檢查", "報告",
        ]
        
        for word in default_words:
            self.add_word(word)
    
    def _get_pinyin(self, text: str) -> str:
        """
        取得文字的拼音（不帶聲調）
        
        Args:
            text: 中文文字
        
        Returns:
            拼音字串（用空格分隔）
        """
        if not PYPINYIN_AVAILABLE:
            return ""
        
        # 使用 NORMAL 風格（不帶聲調的拼音）
        py_list = pinyin(text, style=Style.NORMAL)
        return " ".join([p[0] for p in py_list])
    
    def add_word(self, word: str) -> None:
        """
        添加正確詞彙
        
        Args:
            word: 正確詞彙
        """
        py = self._get_pinyin(word)
        if py:
            self._correct_words[word] = py
            self._pinyin_to_word[py] = word
    
    def correct(self, text: str) -> str:
        """
        校正文字中的同音誤辨識
        
        Args:
            text: ASR 輸出的文字
        
        Returns:
            校正後的文字
        """
        result = text
        
        # 1. 先使用直接映射表（處理拼音不同但發音相近的情況）
        # 按詞長降序排序，長詞優先替換避免部分匹配問題
        sorted_mappings = sorted(self._direct_mappings.items(), 
                                  key=lambda x: len(x[0]), reverse=True)
        for wrong, correct in sorted_mappings:
            result = result.replace(wrong, correct)
        
        # 2. 再使用拼音比對（處理完全同音的情況）
        if not PYPINYIN_AVAILABLE:
            return result
        
        # 對每個正確詞彙，檢查是否有同音替代
        for correct_word, correct_py in self._correct_words.items():
            # 使用滑動視窗找同音詞
            word_len = len(correct_word)
            
            i = 0
            while i <= len(result) - word_len:
                # 取出待檢測片段
                segment = result[i:i + word_len]
                segment_py = self._get_pinyin(segment)
                
                # 如果拼音相同但文字不同，進行替換
                if segment_py == correct_py and segment != correct_word:
                    result = result[:i] + correct_word + result[i + word_len:]
                
                i += 1
        
        return result
    
    def correct_with_log(self, text: str) -> Tuple[str, List[str]]:
        """
        校正文字並返回校正記錄
        
        Args:
            text: ASR 輸出的文字
        
        Returns:
            (校正後文字, 校正記錄列表)
        """
        if not PYPINYIN_AVAILABLE:
            return text, []
        
        result = text
        corrections = []
        
        for correct_word, correct_py in self._correct_words.items():
            word_len = len(correct_word)
            
            i = 0
            while i <= len(result) - word_len:
                segment = result[i:i + word_len]
                segment_py = self._get_pinyin(segment)
                
                if segment_py == correct_py and segment != correct_word:
                    corrections.append(f"{segment} → {correct_word}")
                    result = result[:i] + correct_word + result[i + word_len:]
                
                i += 1
        
        return result, corrections
    
    @property
    def is_available(self) -> bool:
        """拼音校正是否可用"""
        return PYPINYIN_AVAILABLE
    
    @property
    def word_count(self) -> int:
        """詞彙庫詞彙數量"""
        return len(self._correct_words)


# 全域校正器實例
_corrector: Optional[PinyinCorrector] = None


def get_corrector() -> PinyinCorrector:
    """取得全域校正器實例"""
    global _corrector
    if _corrector is None:
        _corrector = PinyinCorrector()
    return _corrector


def correct_pinyin(text: str) -> str:
    """
    拼音校正便捷函式
    
    Args:
        text: ASR 輸出的文字
    
    Returns:
        校正後的文字
    """
    return get_corrector().correct(text)


# 測試
if __name__ == "__main__":
    print("=== 拼音校正測試 ===\n")
    
    corrector = PinyinCorrector()
    print(f"詞彙庫數量: {corrector.word_count}")
    print(f"pypinyin 可用: {corrector.is_available}\n")
    
    test_cases = [
        "惡號病房",      # → 二號病房
        "噩耗病房",      # → 二號病房
        "為護理戰",      # → 回護理站
        "微護理站",      # → 回護理站
        "跟水",          # → 跟隨
        "一跳一塊到護理站",  # 混合文字
        "到十七號病房",  # 正確文字不應改變
    ]
    
    print("校正結果:")
    for text in test_cases:
        result, logs = corrector.correct_with_log(text)
        if logs:
            print(f"  {text} → {result} (修正: {', '.join(logs)})")
        else:
            print(f"  {text} → {result} (無修正)")
