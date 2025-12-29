"""
多後端 LLM 介面

整合多種 LLM 推理後端：
1. llama.cpp - 支援 GGUF 格式，適合新模型（Qwen2.5、Llama3.2 等）
2. NanoLLM/MLC - 最快速，但模型支援有限
3. Mock - 測試用

使用工廠函式 create_llm() 自動選擇合適的後端。
"""
from pathlib import Path
from typing import Optional, Union

# 導入各後端
from src.llm.llama_cpp import LlamaCppInterface, create_llama_cpp, LLAMA_CPP_AVAILABLE

try:
    from src.llm.nano_llm import NanoLLMInterface, MockLLM, NANOLLM_AVAILABLE
except ImportError:
    NANOLLM_AVAILABLE = False
    NanoLLMInterface = None
    MockLLM = None


class UnifiedLLM:
    """
    統一 LLM 介面
    
    封裝不同後端，提供一致的 API。
    """
    
    def __init__(self, backend: str = "auto", **kwargs):
        """
        初始化統一 LLM 介面
        
        Args:
            backend: 後端選擇 ("auto", "llama_cpp", "nanollm", "mock")
            **kwargs: 傳遞給後端的參數
        """
        self.backend_name = backend
        self._backend = None
        self._init_backend(backend, **kwargs)
    
    def _init_backend(self, backend: str, **kwargs) -> None:
        """初始化後端"""
        if backend == "auto":
            # 自動選擇：優先 llama.cpp，然後 NanoLLM
            if LLAMA_CPP_AVAILABLE and kwargs.get('model_path'):
                backend = "llama_cpp"
            elif NANOLLM_AVAILABLE:
                backend = "nanollm"
            else:
                backend = "mock"
        
        self.backend_name = backend
        
        if backend == "llama_cpp":
            model_path = kwargs.get('model_path')
            system_prompt = kwargs.get('system_prompt', '')
            
            if model_path and Path(model_path).exists():
                self._backend = create_llama_cpp(
                    model_path=model_path,
                    system_prompt=system_prompt,
                    n_ctx=kwargs.get('n_ctx', 4096),
                    n_gpu_layers=kwargs.get('n_gpu_layers', -1)
                )
            else:
                print(f"llama.cpp model not found: {model_path}")
                self._backend = MockLLM() if MockLLM else None
        
        elif backend == "nanollm":
            if NanoLLMInterface:
                model_name = kwargs.get('model_name', 'meta-llama/Llama-2-7b-chat-hf')
                self._backend = NanoLLMInterface(
                    model_name=model_name,
                    system_prompt=kwargs.get('system_prompt', '')
                )
                self._backend.load_model()
            else:
                print("NanoLLM not available")
                self._backend = None
        
        else:  # mock
            if MockLLM:
                self._backend = MockLLM(system_prompt=kwargs.get('system_prompt', ''))
            else:
                self._backend = None
    
    def chat(self, user_input: str) -> str:
        """對話"""
        if self._backend is None:
            return "錯誤：LLM 後端未初始化"
        
        if hasattr(self._backend, 'chat'):
            return self._backend.chat(user_input)
        elif hasattr(self._backend, 'generate'):
            return self._backend.generate(user_input)
        else:
            return "錯誤：後端不支援對話"
    
    def reset_history(self) -> None:
        """重置對話歷史"""
        if self._backend and hasattr(self._backend, 'reset_history'):
            self._backend.reset_history()
    
    @property
    def is_available(self) -> bool:
        """是否可用"""
        if self._backend is None:
            return False
        return getattr(self._backend, 'is_available', False)


def create_llm(
    backend: str = "auto",
    model_path: Optional[str] = None,
    model_name: Optional[str] = None,
    system_prompt: str = "",
    use_mock: bool = False,
    **kwargs
) -> UnifiedLLM:
    """
    建立 LLM 實例的工廠函式
    
    Args:
        backend: 後端選擇
            - "auto": 自動選擇
            - "llama_cpp": 使用 llama.cpp (GGUF 模型)
            - "nanollm": 使用 NanoLLM/MLC
            - "mock": 測試用
        model_path: GGUF 模型路徑 (llama.cpp 用)
        model_name: HuggingFace 模型名稱 (NanoLLM 用)
        system_prompt: 系統提示詞
        use_mock: 強制使用 Mock
        **kwargs: 其他參數
    
    Returns:
        UnifiedLLM 實例
    """
    if use_mock:
        backend = "mock"
    
    return UnifiedLLM(
        backend=backend,
        model_path=model_path,
        model_name=model_name,
        system_prompt=system_prompt,
        **kwargs
    )


# 測試用主程式
if __name__ == "__main__":
    print("=== 多後端 LLM 測試 ===")
    print(f"llama.cpp available: {LLAMA_CPP_AVAILABLE}")
    print(f"NanoLLM available: {NANOLLM_AVAILABLE}")
    
    # 使用 Mock 測試
    llm = create_llm(use_mock=True, system_prompt="你是語音控制助手")
    
    if llm.is_available:
        response = llm.chat("向前走")
        print(f"Response: {response}")
