"""
NanoLLM 推理介面

封裝 NanoLLM 的推理功能，支援：
1. 模型載入與推理
2. 對話歷史管理
3. 串流輸出（配合 TTS 即時回應）

設計說明：
NanoLLM 需要在 jetson-containers 容器內執行，
此模組提供與容器內 NanoLLM 互動的介面。

實際部署時有兩種方式：
1. 在容器內執行整個應用
2. 透過 HTTP/gRPC 與容器內的 NanoLLM 服務通訊
"""
import json
from dataclasses import dataclass, field
from typing import Generator, List, Optional

# 嘗試導入 NanoLLM（僅在容器內可用）
try:
    from nano_llm import NanoLLM as NanoLLMCore
    NANOLLM_AVAILABLE = True
except ImportError:
    NANOLLM_AVAILABLE = False


@dataclass
class Message:
    """對話訊息"""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class ConversationHistory:
    """
    對話歷史管理
    
    維護與 LLM 的對話上下文，
    支援最大歷史長度限制以控制記憶體使用。
    """
    messages: List[Message] = field(default_factory=list)
    max_history: int = 10
    system_prompt: str = ""
    
    def add_message(self, role: str, content: str) -> None:
        """新增訊息"""
        self.messages.append(Message(role=role, content=content))
        
        # 保留最新的 N 輪對話
        if len(self.messages) > self.max_history * 2:
            self.messages = self.messages[-(self.max_history * 2):]
    
    def add_user_message(self, content: str) -> None:
        """新增用戶訊息"""
        self.add_message("user", content)
    
    def add_assistant_message(self, content: str) -> None:
        """新增助手回應"""
        self.add_message("assistant", content)
    
    def get_messages_for_chat(self) -> List[dict]:
        """取得對話歷史（字典格式）"""
        result = []
        
        if self.system_prompt:
            result.append({"role": "system", "content": self.system_prompt})
        
        for msg in self.messages:
            result.append({"role": msg.role, "content": msg.content})
        
        return result
    
    def clear(self) -> None:
        """清除對話歷史"""
        self.messages.clear()


class NanoLLMInterface:
    """
    NanoLLM 推理介面
    
    封裝 NanoLLM 的模型載入與推理功能。
    需要在 jetson-containers 環境內執行。
    """
    
    # 支援的模型列表
    SUPPORTED_MODELS = {
        "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
        "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
        "llama-3.2-tw-3b": "yentinglin/Llama-3.2-TW-3B",
        "taide-llama-3-8b": "taide/TAIDE-Llama-3-8B",
        "gemma-3-taide-12b": "taide/Gemma-3-TAIDE-12b-Chat"
    }
    
    def __init__(
        self,
        model_name: str = "qwen2.5-3b",
        quantization: str = "q4f16_ft",
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system_prompt: str = ""
    ):
        """
        初始化 NanoLLM 介面
        
        Args:
            model_name: 模型名稱（短名或完整路徑）
            quantization: 量化格式
            max_tokens: 最大生成 tokens
            temperature: 溫度參數
            top_p: Top-p 取樣
            system_prompt: 系統提示詞
        """
        # 解析模型路徑
        if model_name in self.SUPPORTED_MODELS:
            self.model_path = self.SUPPORTED_MODELS[model_name]
        else:
            self.model_path = model_name
        
        self.quantization = quantization
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        self._model = None
        self._is_initialized = False
        
        # 對話歷史
        self.history = ConversationHistory(system_prompt=system_prompt)
    
    def load_model(self) -> bool:
        """
        載入模型
        
        Returns:
            是否成功載入
        """
        if not NANOLLM_AVAILABLE:
            print("NanoLLM not available (not in jetson-containers)")
            print("Please run this in the NanoLLM container")
            return False
        
        try:
            self._model = NanoLLMCore.from_pretrained(
                self.model_path,
                quantization=self.quantization,
                api="mlc"
            )
            self._is_initialized = True
            print(f"Model loaded: {self.model_path}")
            return True
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def generate(self, prompt: str, stream: bool = False) -> str:
        """
        生成回應
        
        Args:
            prompt: 用戶輸入
            stream: 是否使用串流模式
        
        Returns:
            LLM 回應文字
        """
        if not self._is_initialized or self._model is None:
            return "錯誤：模型尚未載入"
        
        # 加入用戶訊息
        self.history.add_user_message(prompt)
        
        try:
            # 取得完整對話歷史
            messages = self.history.get_messages_for_chat()
            
            # 生成回應
            response = self._model.generate(
                messages,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                streaming=stream
            )
            
            if stream:
                # 串流模式回傳生成器
                return self._stream_response(response)
            else:
                # 非串流模式
                result = response if isinstance(response, str) else str(response)
                self.history.add_assistant_message(result)
                return result
                
        except Exception as e:
            error_msg = f"生成失敗: {e}"
            print(error_msg)
            return error_msg
    
    def _stream_response(self, response_iter) -> Generator[str, None, None]:
        """處理串流回應"""
        full_response = ""
        
        for token in response_iter:
            full_response += token
            yield token
        
        # 完成後加入歷史
        self.history.add_assistant_message(full_response)
    
    def chat(self, user_input: str) -> str:
        """
        對話介面（非串流）
        
        Args:
            user_input: 用戶輸入
        
        Returns:
            助手回應
        """
        return self.generate(user_input, stream=False)
    
    def stream_chat(self, user_input: str) -> Generator[str, None, None]:
        """
        串流對話介面
        
        Args:
            user_input: 用戶輸入
        
        Yields:
            回應 tokens
        """
        return self.generate(user_input, stream=True)
    
    def reset_history(self) -> None:
        """重置對話歷史"""
        self.history.clear()
    
    @property
    def is_available(self) -> bool:
        """模型是否可用"""
        return self._is_initialized


class MockLLM:
    """
    模擬 LLM（測試用）
    
    當 NanoLLM 不可用時用於測試流程。
    """
    
    def __init__(self, system_prompt: str = ""):
        self.history = ConversationHistory(system_prompt=system_prompt)
    
    def generate(self, prompt: str, stream: bool = False) -> str:
        """模擬生成"""
        self.history.add_user_message(prompt)
        
        # 簡單的關鍵字回應
        response = self._mock_response(prompt)
        self.history.add_assistant_message(response)
        
        return response
    
    def _mock_response(self, prompt: str) -> str:
        """根據關鍵字產生模擬回應"""
        prompt_lower = prompt.lower()
        
        if "前進" in prompt:
            return '{"command": "forward", "duration": 2.0}'
        elif "後退" in prompt:
            return '{"command": "backward", "duration": 2.0}'
        elif "左轉" in prompt:
            return '{"command": "left", "duration": 1.0}'
        elif "右轉" in prompt:
            return '{"command": "right", "duration": 1.0}'
        elif "停" in prompt:
            return '{"command": "stop"}'
        else:
            return '{"command": "unknown", "message": "我不太明白您的指令，請再說一次"}'
    
    def chat(self, user_input: str) -> str:
        return self.generate(user_input)
    
    def reset_history(self) -> None:
        self.history.clear()
    
    @property
    def is_available(self) -> bool:
        return True


def create_llm(
    model_name: str = "qwen2.5-3b",
    use_mock: bool = False,
    **kwargs
) -> NanoLLMInterface:
    """
    建立 LLM 實例的工廠函式
    
    Args:
        model_name: 模型名稱
        use_mock: 是否使用模擬 LLM
        **kwargs: 其他參數
    
    Returns:
        LLM 實例
    """
    if use_mock or not NANOLLM_AVAILABLE:
        print("Using MockLLM for testing")
        return MockLLM(**kwargs)
    
    llm = NanoLLMInterface(model_name=model_name, **kwargs)
    llm.load_model()
    return llm


# 測試用主程式
if __name__ == "__main__":
    print("=== NanoLLM 模組測試 ===")
    print(f"NanoLLM available: {NANOLLM_AVAILABLE}")
    
    # 使用 MockLLM 測試
    llm = create_llm(use_mock=True)
    
    test_inputs = [
        "請向前走",
        "左轉一下",
        "停止",
        "今天天氣如何"
    ]
    
    for user_input in test_inputs:
        response = llm.chat(user_input)
        print(f"User: {user_input}")
        print(f"Assistant: {response}")
        print()
