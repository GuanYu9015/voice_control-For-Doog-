"""
護理輔助車 LLM 指令定義

根據用戶原有 Prompt 格式設計：
- type: COMMAND / FOLLOW / WAKE / ERROR
- route: 起點代號 + 終點代號 (如 "OD")
- target: 終點代號
- action: START / STOP (跟隨模式)
- text: 回應文字
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import json
import re


class CommandType(Enum):
    """指令類型"""
    COMMAND = "COMMAND"     # 導航指令
    FOLLOW = "FOLLOW"       # 跟隨模式
    WAKE = "WAKE"           # 喚醒詞
    ERROR = "ERROR"         # 錯誤/無法識別


@dataclass
class NursingCartAction:
    """
    護理輔助車動作
    
    LLM 輸出的 JSON 格式：
    {"type": "COMMAND", "route": "OD", "target": "D", "text": "從護理站到4號病房"}
    {"type": "FOLLOW", "action": "START", "text": "已啟動跟隨模式"}
    {"type": "WAKE", "text": "我在，請問需要什麼服務"}
    {"type": "ERROR", "text": "無法辨識指令"}
    """
    type: str               # 指令類型: COMMAND / FOLLOW / WAKE / ERROR
    route: str = ""         # 路徑代碼 (如 "OD")
    target: str = ""        # 目標代號 (如 "D")
    action: str = ""        # 動作: START / STOP (僅 FOLLOW 使用)
    text: str = ""          # 回應文字
    
    def to_dict(self) -> dict:
        """轉換為字典"""
        result = {"type": self.type, "text": self.text}
        if self.route:
            result["route"] = self.route
        if self.target:
            result["target"] = self.target
        if self.action:
            result["action"] = self.action
        return result
    
    def is_navigation(self) -> bool:
        """是否為導航指令"""
        return self.type == "COMMAND"
    
    def is_follow(self) -> bool:
        """是否為跟隨指令"""
        return self.type == "FOLLOW"
    
    def is_wake(self) -> bool:
        """是否為喚醒"""
        return self.type == "WAKE"
    
    def is_error(self) -> bool:
        """是否為錯誤"""
        return self.type == "ERROR"
    
    @property
    def start_code(self) -> str:
        """起點代號"""
        return self.route[0] if len(self.route) >= 2 else ""
    
    @property
    def end_code(self) -> str:
        """終點代號"""
        return self.route[1] if len(self.route) >= 2 else self.target


def parse_nursing_command(text: str) -> Optional[NursingCartAction]:
    """
    解析 LLM 回應為護理車動作
    
    Args:
        text: LLM 回應（JSON 或文字）
    
    Returns:
        NursingCartAction 或 None
    """
    # 嘗試直接解析 JSON
    try:
        data = json.loads(text.strip())
        return _parse_json_command(data)
    except (json.JSONDecodeError, TypeError):
        pass
    
    # 嘗試從文字中提取 JSON
    json_match = re.search(r'\{[^{}]+\}', text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return _parse_json_command(data)
        except (json.JSONDecodeError, TypeError):
            pass
    
    return None


def _parse_json_command(data: dict) -> NursingCartAction:
    """解析 JSON 為 NursingCartAction"""
    return NursingCartAction(
        type=data.get('type', 'ERROR'),
        route=data.get('route', ''),
        target=data.get('target', ''),
        action=data.get('action', ''),
        text=data.get('text', '')
    )


def build_nursing_cart_prompt(current_location: str, current_name: str, user_input: str) -> str:
    """
    建立護理車 LLM Prompt
    
    Args:
        current_location: 當前位置代號 (如 "O")
        current_name: 當前位置名稱 (如 "護理站")
        user_input: 用戶輸入文字
    
    Returns:
        完整的 Prompt 字串
    """
    prompt = f"""You are a hospital nursing cart voice assistant.
You MUST respond with ONLY ONE valid JSON object. No explanations, no extra text.

ROOM CODE MAPPING (Strictly follow this table, DO NOT guess):
1號 -> A    2號 -> B    3號 -> C    4號 -> D
5號 -> E    6號 -> F    7號 -> G    8號 -> H
9號 -> I    10號 -> J   11號 -> K   12號 -> L
13號 -> M   14號 -> N   15號 -> P   16號 -> Q
17號 -> R   護理站 -> O

DO NOT calculate letter positions (e.g. 5 is NOT F). Look up the table above.

CURRENT LOCATION: {current_name} (code: {current_location})

OUTPUT FORMAT:
- Navigation: {{"reasoning":"Step-by-step logic","type":"COMMAND","route":"XY","target":"Y","text":"description"}}
  Where X=start code, Y=target code. Example: "OD" means from O to D.
- Follow start: {{"reasoning":"User wants to follow","type":"FOLLOW","action":"START","text":"已啟動跟隨模式"}}
- Follow stop: {{"reasoning":"User wants to stop","type":"FOLLOW","action":"STOP","text":"已停止跟隨"}}
- Error: {{"type":"ERROR","text":"無法辨識指令"}}

CRITICAL EXAMPLES:
User: 去5號病房 → {{"reasoning":"Rule: 5號=E. Target is E.","type":"COMMAND","route":"{current_location}E","target":"E","text":"前往5號病房"}}
User: 到10號病房 → {{"reasoning":"Rule: J=10號. Target is J.","type":"COMMAND","route":"{current_location}J","target":"J","text":"前往10號病房"}}
User: 去16號病房 → {{"reasoning":"Rule: Q=16號. Target is Q.","type":"COMMAND","route":"{current_location}Q","target":"Q","text":"前往16號病房"}}
User: 到17號病房 → {{"reasoning":"Rule: R=17號. Target is R.","type":"COMMAND","route":"{current_location}R","target":"R","text":"前往17號病房"}}
User: 開始跟隨 → {{"reasoning":"User said '開始跟隨'","type":"FOLLOW","action":"START","text":"已啟動跟隨模式"}}

User input: {user_input}
JSON output:"""
    return prompt


# 注意：LLM Prompt 統一使用 build_nursing_cart_prompt() 動態生成
# 站點映射統一使用 station_mapper.py（資料來源：config/robot_config.yaml）


# 測試
if __name__ == "__main__":
    print("=== 護理車指令解析測試 ===\n")
    
    test_cases = [
        '{"type": "COMMAND", "route": "OD", "target": "D", "text": "從護理站到4號病房"}',
        '{"type": "FOLLOW", "action": "START", "text": "已啟動跟隨模式"}',
        '{"type": "FOLLOW", "action": "STOP", "text": "已停止跟隨"}',
        '{"type": "WAKE", "text": "我在，請問需要什麼服務"}',
        '{"type": "ERROR", "text": "無法辨識指令"}',
    ]
    
    for text in test_cases:
        action = parse_nursing_command(text)
        if action:
            print(f"輸入: {text[:60]}...")
            print(f"解析: type={action.type}, route={action.route}, action={action.action}")
            print(f"導航: {action.is_navigation()}, 跟隨: {action.is_follow()}\n")
    
    # 測試 Prompt 生成
    prompt = build_nursing_cart_prompt("O", "護理站", "去4號病房")
    print("=== Prompt 範例 ===")
    print(prompt[:500] + "...")
