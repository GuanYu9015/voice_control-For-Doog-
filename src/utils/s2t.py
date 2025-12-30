"""
簡體中文轉繁體中文模組

提供簡轉繁功能，用於處理 ASR 輸出的簡體中文文字。

實作方式建議：
1. OpenCC（推薦）：成熟的簡繁轉換庫，支援詞彙級轉換
2. hanziconv：輕量級，純 Python
3. 內建字典：自建簡繁對照表

注意：繁體中文有台灣、香港、澳門等變體，本模組預設使用台灣繁體（tw）
"""

# 嘗試導入 OpenCC
try:
    from opencc import OpenCC
    OPENCC_AVAILABLE = True
except ImportError:
    OPENCC_AVAILABLE = False

# 備用：嘗試 hanziconv
try:
    import hanziconv
    HANZICONV_AVAILABLE = True
except ImportError:
    HANZICONV_AVAILABLE = False


class SimplifiedToTraditional:
    """
    簡體中文轉繁體中文
    
    優先使用 OpenCC，若不可用則使用 hanziconv，
    最後使用內建基本字典。
    """
    
    def __init__(self, variant: str = "tw"):
        """
        初始化轉換器
        
        Args:
            variant: 繁體變體
                - "tw": 台灣繁體（預設）
                - "hk": 香港繁體
                - "t": 標準繁體
        """
        self.variant = variant
        self._converter = None
        self._backend = "none"
        
        self._init_converter()
    
    def _init_converter(self) -> None:
        """初始化轉換後端"""
        if OPENCC_AVAILABLE:
            try:
                # OpenCC 設定：s2tw = 簡體到台灣繁體
                config_map = {
                    "tw": "s2tw",      # 簡體到台灣繁體
                    "twp": "s2twp",    # 簡體到台灣繁體（含慣用詞）
                    "hk": "s2hk",      # 簡體到香港繁體
                    "t": "s2t",        # 簡體到標準繁體
                }
                config = config_map.get(self.variant, "s2tw")
                self._converter = OpenCC(config)
                self._backend = "opencc"
                print(f"S2T using OpenCC ({config})")
                return
            except Exception as e:
                print(f"OpenCC init failed: {e}")
        
        if HANZICONV_AVAILABLE:
            self._backend = "hanziconv"
            print("S2T using hanziconv")
            return
        
        # 使用內建基本字典
        self._backend = "builtin"
        self._init_builtin_dict()
        print("S2T using builtin dictionary")
    
    def _init_builtin_dict(self) -> None:
        """初始化內建簡繁對照字典"""
        # 常用簡繁對照（僅包含常用字）
        self._builtin_dict = {
            # 護理機器人相關
            "护理": "護理", "机器人": "機器人", "智护车": "智護車",
            "护": "護", "机": "機", "车": "車",
            # 病房相關
            "号": "號", "病房": "病房", "护理站": "護理站",
            # 動作指令
            "向前": "向前", "后退": "後退", "左转": "左轉", "右转": "右轉",
            "前进": "前進", "后": "後", "转": "轉", "进": "進",
            "停止": "停止", "继续": "繼續", "跟随": "跟隨",
            # 常用字（補充）
            "从": "從", "块": "塊", "跳": "跳", "个": "個",
            "请": "請", "谢谢": "謝謝", "对": "對", "说": "說",
            "听": "聽", "这": "這", "那": "那", "里": "裡",
            "时间": "時間", "电": "電", "话": "話",
            "开": "開", "关": "關", "门": "門",
            "东": "東", "西": "西", "南": "南", "北": "北",
            "医": "醫", "药": "藥", "病": "病",
            "帮": "幫", "给": "給", "拿": "拿",
            "发": "發", "认": "認", "识": "識",
            "长": "長", "头": "頭", "脸": "臉", "见": "見",
            "几": "幾", "点": "點", "�的": "點",
            "国": "國", "语": "語", "学": "學",
            "为": "為", "什": "什", "么": "麼",
            "没": "沒", "有": "有", "过": "過",
            # 數字相關
            "一": "一", "二": "二", "三": "三", "四": "四", "五": "五",
            "六": "六", "七": "七", "八": "八", "九": "九", "十": "十",
            "零": "零", "两": "兩",
        }
    
    def convert(self, text: str) -> str:
        """
        簡體轉繁體
        
        Args:
            text: 簡體中文文字
        
        Returns:
            繁體中文文字
        """
        if not text:
            return text
        
        if self._backend == "opencc":
            return self._converter.convert(text)
        
        elif self._backend == "hanziconv":
            return hanziconv.HanziConv.toTraditional(text)
        
        else:
            # 使用內建字典
            result = text
            for simplified, traditional in self._builtin_dict.items():
                result = result.replace(simplified, traditional)
            return result
    
    @property
    def backend(self) -> str:
        """當前使用的後端"""
        return self._backend
    
    @property
    def is_available(self) -> bool:
        """轉換器是否可用"""
        return self._backend != "none"


# 全域轉換器實例
_converter: SimplifiedToTraditional = None


def s2t(text: str, variant: str = "tw") -> str:
    """
    簡體轉繁體（便捷函式）
    
    Args:
        text: 簡體中文文字
        variant: 繁體變體 (tw/hk/t)
    
    Returns:
        繁體中文文字
    """
    global _converter
    
    if _converter is None:
        _converter = SimplifiedToTraditional(variant)
    
    return _converter.convert(text)


def create_converter(variant: str = "tw") -> SimplifiedToTraditional:
    """建立轉換器實例"""
    return SimplifiedToTraditional(variant)


# 測試
if __name__ == "__main__":
    print("=== 簡轉繁測試 ===")
    
    converter = create_converter("tw")
    print(f"Backend: {converter.backend}")
    
    test_texts = [
        "护理车请过来",
        "向前走",
        "左转",
        "停止",
        "请帮我开门",
        "智护车你好"
    ]
    
    print("\n轉換結果:")
    for text in test_texts:
        result = converter.convert(text)
        print(f"  {text} → {result}")
