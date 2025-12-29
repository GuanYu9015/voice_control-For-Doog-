"""
llama.cpp 推理介面

使用 llama-cpp-python 進行 LLM 推理。
支援 GGUF 格式的模型，在 Jetson 上使用 CUDA 加速。

設計說明：
1. llama.cpp 支援最新的模型（Qwen2.5、Llama3.2 等）
2. 使用 GGUF 量化格式（Q4_K_M 等）
3. 支援 GPU 加速（CUDA）
"""
from pathlib import Path
from typing import Generator, List, Optional

# 嘗試導入 llama-cpp-python
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("Warning: llama-cpp-python not installed")
    print("Install with: CMAKE_ARGS='-DLLAMA_CUDA=on' pip install llama-cpp-python")


class LlamaCppInterface:
    """
    llama.cpp 推理介面
    
    使用 llama-cpp-python 載入 GGUF 模型進行推理。
    """
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,  # -1 = 全部放 GPU
        n_threads: int = 4,
        verbose: bool = False
    ):
        """
        初始化 llama.cpp 介面
        
        Args:
            model_path: GGUF 模型路徑
            n_ctx: 上下文長度
            n_gpu_layers: GPU 層數（-1 = 全部）
            n_threads: CPU 執行緒數
            verbose: 是否顯示詳細日誌
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.verbose = verbose
        
        self._model = None
        self._is_initialized = False
        
        # 對話歷史
        self._chat_history: List[dict] = []
        self._system_prompt = ""
        
        self._init_model()
    
    def _init_model(self) -> None:
        """初始化模型"""
        if not LLAMA_CPP_AVAILABLE:
            print("llama-cpp-python not available")
            return
        
        if not Path(self.model_path).exists():
            print(f"Model not found: {self.model_path}")
            return
        
        try:
            print(f"Loading model: {self.model_path}")
            
            self._model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                n_threads=self.n_threads,
                verbose=self.verbose
            )
            
            self._is_initialized = True
            print("Model loaded successfully")
            
        except Exception as e:
            print(f"Failed to load model: {e}")
    
    def set_system_prompt(self, prompt: str) -> None:
        """設定系統提示詞"""
        self._system_prompt = prompt
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.3,  # 低 temperature 確保穩定輸出
        top_p: float = 0.9,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        生成回應（無狀態模式，每次獨立生成）
        
        Args:
            prompt: 用戶輸入（包含完整的 Prompt）
            max_tokens: 最大生成 tokens
            temperature: 溫度
            top_p: Top-p
            stop: 停止詞
        
        Returns:
            生成的文字
        """
        if not self._is_initialized or self._model is None:
            return "錯誤：模型尚未載入"
        
        try:
            # 無狀態模式：不保留對話歷史，每次獨立生成
            # prompt 本身已經包含完整的指令（由 build_nursing_cart_prompt 生成）
            messages = []
            
            # 只加入當前輸入（prompt 本身就是完整的指令）
            messages.append({"role": "user", "content": prompt})
            
            # 使用 chat completion
            response = self._model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop
            )
            
            # 取得回應
            result = response["choices"][0]["message"]["content"]
            
            return result
            
        except Exception as e:
            return f"生成失敗: {e}"
    
    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> Generator[str, None, None]:
        """串流生成"""
        if not self._is_initialized or self._model is None:
            yield "錯誤：模型尚未載入"
            return
        
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.extend(self._chat_history)
        messages.append({"role": "user", "content": prompt})
        
        try:
            full_response = ""
            
            for chunk in self._model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            ):
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta:
                    content = delta["content"]
                    full_response += content
                    yield content
            
            # 更新歷史
            self._chat_history.append({"role": "user", "content": prompt})
            self._chat_history.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            yield f"串流生成失敗: {e}"
    
    def chat(self, user_input: str) -> str:
        """對話介面"""
        return self.generate(user_input)
    
    def reset_history(self) -> None:
        """重置對話歷史"""
        self._chat_history.clear()
    
    @property
    def is_available(self) -> bool:
        """模型是否可用"""
        return self._is_initialized


def create_llama_cpp(
    model_path: str,
    system_prompt: str = "",
    **kwargs
) -> Optional[LlamaCppInterface]:
    """
    建立 llama.cpp 介面的工廠函式
    
    Args:
        model_path: GGUF 模型路徑
        system_prompt: 系統提示詞
        **kwargs: 其他參數
    
    Returns:
        LlamaCppInterface 實例
    """
    if not LLAMA_CPP_AVAILABLE:
        print("llama-cpp-python not installed")
        return None
    
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        return None
    
    llm = LlamaCppInterface(model_path, **kwargs)
    
    if system_prompt:
        llm.set_system_prompt(system_prompt)
    
    return llm


# 測試用主程式
if __name__ == "__main__":
    print("=== llama.cpp 模組測試 ===")
    print(f"llama-cpp-python available: {LLAMA_CPP_AVAILABLE}")
    
    if LLAMA_CPP_AVAILABLE:
        # 測試需要實際模型
        print("\n使用方式:")
        print("  llm = create_llama_cpp('path/to/model.gguf')")
        print("  response = llm.chat('你好')")
