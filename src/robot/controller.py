"""
機器人控制模組

提供機器人控制介面，解析 LLM 輸出的指令並控制輪型機器人。

設計說明：
1. 採用抽象介面設計，方便之後替換實際控制實作
2. 目前為 Mock 實作，僅輸出指令不實際控制
3. 之後可替換為 Serial/ROS 等實際控制介面
"""
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class RobotCommand:
    """機器人控制指令"""
    command: str  # forward, backward, left, right, stop
    duration: float = 0.0  # 執行時間（秒），0 表示持續
    speed: float = 0.5  # 速度 (0.0 ~ 1.0)
    message: str = ""  # 語音回應訊息


class RobotController(ABC):
    """
    機器人控制器抽象介面
    
    定義機器人控制的標準介面，
    具體實作可以是 Serial、ROS、HTTP 等。
    """
    
    @abstractmethod
    def forward(self, duration: float = 0, speed: float = 0.5) -> bool:
        """前進"""
        pass
    
    @abstractmethod
    def backward(self, duration: float = 0, speed: float = 0.5) -> bool:
        """後退"""
        pass
    
    @abstractmethod
    def turn_left(self, duration: float = 0, speed: float = 0.5) -> bool:
        """左轉"""
        pass
    
    @abstractmethod
    def turn_right(self, duration: float = 0, speed: float = 0.5) -> bool:
        """右轉"""
        pass
    
    @abstractmethod
    def stop(self) -> bool:
        """停止"""
        pass
    
    def execute(self, command: RobotCommand) -> bool:
        """
        執行指令
        
        Args:
            command: 機器人指令
        
        Returns:
            是否執行成功
        """
        cmd_map = {
            'forward': lambda: self.forward(command.duration, command.speed),
            'backward': lambda: self.backward(command.duration, command.speed),
            'left': lambda: self.turn_left(command.duration, command.speed),
            'right': lambda: self.turn_right(command.duration, command.speed),
            'stop': lambda: self.stop(),
        }
        
        if command.command in cmd_map:
            return cmd_map[command.command]()
        else:
            print(f"[Robot] Unknown command: {command.command}")
            return False


class MockRobotController(RobotController):
    """
    模擬機器人控制器（測試用）
    
    不實際控制機器人，僅輸出指令。
    """
    
    def __init__(self):
        self._is_moving = False
        self._current_command = None
    
    def forward(self, duration: float = 0, speed: float = 0.5) -> bool:
        print(f"[Robot] FORWARD (duration={duration}s, speed={speed})")
        self._simulate_movement('forward', duration)
        return True
    
    def backward(self, duration: float = 0, speed: float = 0.5) -> bool:
        print(f"[Robot] BACKWARD (duration={duration}s, speed={speed})")
        self._simulate_movement('backward', duration)
        return True
    
    def turn_left(self, duration: float = 0, speed: float = 0.5) -> bool:
        print(f"[Robot] TURN LEFT (duration={duration}s, speed={speed})")
        self._simulate_movement('left', duration)
        return True
    
    def turn_right(self, duration: float = 0, speed: float = 0.5) -> bool:
        print(f"[Robot] TURN RIGHT (duration={duration}s, speed={speed})")
        self._simulate_movement('right', duration)
        return True
    
    def stop(self) -> bool:
        print("[Robot] STOP")
        self._is_moving = False
        self._current_command = None
        return True
    
    def _simulate_movement(self, command: str, duration: float) -> None:
        """模擬移動（非阻塞）"""
        self._is_moving = True
        self._current_command = command
        
        if duration > 0:
            # 模擬執行時間後停止
            import threading
            def delayed_stop():
                time.sleep(duration)
                if self._current_command == command:
                    self.stop()
            
            threading.Thread(target=delayed_stop, daemon=True).start()


class CommandParser:
    """
    指令解析器
    
    解析 LLM 輸出的 JSON 指令或自然語言指令。
    """
    
    # 關鍵字對應
    KEYWORD_MAP = {
        '前進': 'forward',
        '向前': 'forward',
        '往前': 'forward',
        '後退': 'backward',
        '往後': 'backward',
        '倒退': 'backward',
        '左轉': 'left',
        '向左': 'left',
        '右轉': 'right',
        '向右': 'right',
        '停止': 'stop',
        '停': 'stop',
        '別動': 'stop',
    }
    
    def parse(self, text: str) -> Optional[RobotCommand]:
        """
        解析指令文字
        
        Args:
            text: LLM 輸出或語音辨識結果
        
        Returns:
            機器人指令或 None
        """
        # 嘗試解析 JSON
        try:
            data = json.loads(text)
            return RobotCommand(
                command=data.get('command', 'unknown'),
                duration=float(data.get('duration', 0)),
                speed=float(data.get('speed', 0.5)),
                message=data.get('message', '')
            )
        except (json.JSONDecodeError, TypeError):
            pass
        
        # 嘗試關鍵字比對
        for keyword, command in self.KEYWORD_MAP.items():
            if keyword in text:
                return RobotCommand(
                    command=command,
                    duration=2.0,  # 預設執行 2 秒
                    message=f"好的，{keyword}"
                )
        
        return None


def create_robot_controller(
    controller_type: str = "mock",
    **kwargs
) -> RobotController:
    """
    建立機器人控制器的工廠函式
    
    Args:
        controller_type: 控制器類型 ("mock", "serial", "ros")
        **kwargs: 其他參數
    
    Returns:
        機器人控制器實例
    """
    if controller_type == "mock":
        return MockRobotController()
    else:
        # 未來可擴展其他控制器
        print(f"Unknown controller type: {controller_type}, using mock")
        return MockRobotController()


# 測試用主程式
if __name__ == "__main__":
    print("=== 機器人控制模組測試 ===")
    
    # 建立控制器與解析器
    robot = create_robot_controller("mock")
    parser = CommandParser()
    
    # 測試指令
    test_inputs = [
        '{"command": "forward", "duration": 2.0}',
        '{"command": "left", "duration": 1.0}',
        '請向前走',
        '左轉一下',
        '停止',
        '今天天氣如何'
    ]
    
    for text in test_inputs:
        print(f"\n輸入: {text}")
        cmd = parser.parse(text)
        
        if cmd:
            print(f"解析: {cmd}")
            robot.execute(cmd)
        else:
            print("無法解析為機器人指令")
