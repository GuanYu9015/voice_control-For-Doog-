"""
機器人控制指令定義

定義 LLM 可能輸出的 JSON 指令格式。
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class CommandType(Enum):
    """指令類型"""
    # 移動控制
    FORWARD = "forward"      # 前進
    BACKWARD = "backward"    # 後退
    LEFT = "left"            # 左轉
    RIGHT = "right"          # 右轉
    STOP = "stop"            # 停止
    
    # 速度控制
    SPEED_UP = "speed_up"    # 加速
    SPEED_DOWN = "speed_down"  # 減速
    
    # 狀態查詢
    STATUS = "status"        # 查詢狀態
    BATTERY = "battery"      # 查詢電量
    
    # 特殊指令
    PATROL = "patrol"        # 巡邏
    RETURN = "return"        # 返回
    FOLLOW = "follow"        # 跟隨
    
    # 未知/對話
    UNKNOWN = "unknown"      # 未識別的指令


@dataclass
class RobotCommand:
    """
    機器人控制指令
    
    LLM 輸出的 JSON 格式範例：
    
    # 基本移動
    {"command": "forward", "duration": 2.0}
    {"command": "backward", "duration": 1.5}
    {"command": "left", "duration": 0.5}
    {"command": "right", "duration": 0.5}
    {"command": "stop"}
    
    # 帶速度
    {"command": "forward", "duration": 2.0, "speed": 0.8}
    
    # 帶回應訊息
    {"command": "forward", "duration": 2.0, "message": "好的，向前走兩秒"}
    
    # 僅回應（非控制指令）
    {"command": "unknown", "message": "今天天氣晴朗"}
    
    # 狀態查詢
    {"command": "status"}
    {"command": "battery"}
    """
    command: str               # 指令類型
    duration: float = 0.0      # 執行時間（秒），0 = 持續
    speed: float = 0.5         # 速度 (0.0 ~ 1.0)
    message: str = ""          # 語音回應訊息
    
    def is_movement(self) -> bool:
        """是否為移動指令"""
        return self.command in ['forward', 'backward', 'left', 'right', 'stop']
    
    def is_query(self) -> bool:
        """是否為查詢指令"""
        return self.command in ['status', 'battery']


# LLM System Prompt 建議
LLM_SYSTEM_PROMPT = """你是護理機器人「智護車」的語音控制助手。

## 角色
你負責理解用戶的語音指令，並產生機器人控制命令。

## 輸出格式
永遠使用 JSON 格式回應，格式如下：

### 移動控制
- 前進：{"command": "forward", "duration": 2.0, "message": "好的，向前走"}
- 後退：{"command": "backward", "duration": 2.0, "message": "好的，後退"}
- 左轉：{"command": "left", "duration": 1.0, "message": "好的，左轉"}
- 右轉：{"command": "right", "duration": 1.0, "message": "好的，右轉"}
- 停止：{"command": "stop", "message": "已停止"}

### 速度控制
- 加速：{"command": "speed_up", "message": "好的，加速"}
- 減速：{"command": "speed_down", "message": "好的，減速"}

### 狀態查詢
- 電量：{"command": "battery", "message": "電量剩餘80%"}
- 狀態：{"command": "status", "message": "目前狀態正常"}

### 一般對話（非控制指令）
- {"command": "unknown", "message": "您的回應"}

## 參數說明
- command: 指令類型
- duration: 執行時間（秒），省略或 0 表示持續執行直到下個指令
- speed: 速度 (0.1 ~ 1.0)，省略使用預設值 0.5
- message: 語音回應文字

## 範例對話
用戶：向前走三秒
回應：{"command": "forward", "duration": 3.0, "message": "好的，向前走三秒"}

用戶：慢一點
回應：{"command": "speed_down", "message": "好的，已減速"}

用戶：今天幾號
回應：{"command": "unknown", "message": "今天是12月28日"}
"""


def parse_command(text: str) -> Optional[RobotCommand]:
    """
    解析 JSON 文字為 RobotCommand
    
    Args:
        text: JSON 字串或 LLM 回應
    
    Returns:
        RobotCommand 或 None
    """
    import json
    import re
    
    # 嘗試直接解析
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
    
    # 嘗試從文字中提取 JSON
    json_match = re.search(r'\{[^}]+\}', text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return RobotCommand(
                command=data.get('command', 'unknown'),
                duration=float(data.get('duration', 0)),
                speed=float(data.get('speed', 0.5)),
                message=data.get('message', '')
            )
        except (json.JSONDecodeError, TypeError):
            pass
    
    return None


# 測試
if __name__ == "__main__":
    print("=== 指令解析測試 ===\n")
    
    test_cases = [
        '{"command": "forward", "duration": 2.0}',
        '{"command": "left", "duration": 1.0, "message": "好的，左轉"}',
        '{"command": "stop"}',
        '{"command": "unknown", "message": "今天天氣晴朗"}',
        '好的，我來幫你 {"command": "forward", "duration": 3.0} 前進三秒',
    ]
    
    for text in test_cases:
        cmd = parse_command(text)
        if cmd:
            print(f"輸入: {text[:50]}...")
            print(f"解析: {cmd}")
            print(f"是移動指令: {cmd.is_movement()}\n")
