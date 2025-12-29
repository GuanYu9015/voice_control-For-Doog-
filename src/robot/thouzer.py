"""
Thouzer 載具介面

整合 StationMapper 和 MQTTController，
提供高階護理車控制 API。

使用方式：
    thouzer = ThouzerController()
    thouzer.connect()
    thouzer.navigate_to("3201")
"""
import logging
from typing import Callable, Optional, Tuple

from .station_mapper import StationMapper, get_station_mapper
from .mqtt_controller import MQTTController, create_mqtt_controller
from .nursing_commands import NursingCartAction, parse_nursing_command

logger = logging.getLogger(__name__)


class ThouzerController:
    """
    Thouzer 載具控制器
    
    整合站點映射和 MQTT 控制。
    """
    
    def __init__(
        self,
        on_status_change: Optional[Callable[[str, str], None]] = None,
        on_position_change: Optional[Callable[[str, str], None]] = None
    ):
        """
        初始化控制器
        
        Args:
            on_status_change: 狀態變更回調 (status, message)
            on_position_change: 位置變更回調 (code, name)
        """
        self.on_status_change = on_status_change
        self.on_position_change = on_position_change
        
        # 初始化元件
        self.station_mapper = get_station_mapper()
        self.mqtt = create_mqtt_controller(on_status=self._on_mqtt_status)
        
        # 狀態
        self._current_position = 'O'  # 預設在護理站
        self._target_position = None
        self._is_moving = False
        self._is_following = False
        self._is_paused = False
        
    def _on_mqtt_status(self, status: dict):
        """MQTT 狀態回調"""
        logger.debug(f"MQTT status: {status}")
        # 可根據需要解析載具狀態
    
    def connect(self) -> bool:
        """連接 MQTT"""
        if self.mqtt:
            result = self.mqtt.connect()
            if result:
                self._notify_status("connected", "已連接到載具")
            return result
        return False
    
    def disconnect(self) -> None:
        """斷開連接"""
        if self.mqtt:
            self.mqtt.disconnect()
            self._notify_status("disconnected", "已斷開連接")
    
    def navigate_to(self, destination: str) -> bool:
        """
        導航到目標位置
        
        Args:
            destination: 目標可以是：
                - 代號 (如 "A", "C")
                - 病房號碼 (如 "3201")  
                - 簡單數字 (如 "3" = 3號病房 = C)
        
        Returns:
            是否成功發送指令
        """
        # 確定目標代號
        dest_code = None
        destination = str(destination).strip()
        
        # 1. 單字母代號（A-R, O）
        if len(destination) == 1 and destination.upper() in self.station_mapper.STATIONS:
            dest_code = destination.upper()
        
        # 2. 使用 station_mapper 轉換（支援數字病房號碼 1-17 或完整病房號碼 3201）
        else:
            dest_code = self.station_mapper.room_to_code(destination)
        
        if not dest_code:
            logger.error(f"Unknown destination: {destination}")
            self._notify_status("error", f"未知目的地：{destination}")
            return False
        
        # 取得軌跡
        route = self.station_mapper.get_route_for_destination(
            self._current_position, 
            self.station_mapper.code_to_room(dest_code) or dest_code
        )
        
        if not route:
            logger.error(f"No route from {self._current_position} to {dest_code}")
            self._notify_status("error", f"找不到從目前位置到{destination}的路徑")
            return False
        
        route_code, route_number = route
        
        # 發送指令
        if self.mqtt and self.mqtt.execute_route(route_number):
            self._target_position = dest_code
            self._is_moving = True
            
            # 導航指令發送成功後，立即更新當前位置
            # （實際的載具會在抵達後更新，這裡先假設成功）
            self._current_position = dest_code
            
            dest_name = self.station_mapper.code_to_name(dest_code)
            self._notify_status("navigating", f"正在前往{dest_name}")
            logger.info(f"Navigation started: {route_code} -> B{route_number}")
            
            # 觸發位置更新回調
            if self.on_position_change:
                self.on_position_change(dest_code, dest_name)
            
            return True
        
        return False
    
    def return_to_station(self) -> bool:
        """返回護理站"""
        return self.navigate_to('O')
    
    def start_follow(self) -> bool:
        """啟動跟隨模式"""
        if self.mqtt and self.mqtt.start_follow_mode():
            self._is_following = True
            self._notify_status("following", "跟隨模式已啟動")
            return True
        return False
    
    def stop_follow(self) -> bool:
        """停止跟隨模式"""
        if self.mqtt and self.mqtt.stop_follow_mode():
            self._is_following = False
            self._notify_status("stopped", "跟隨模式已停止")
            return True
        return False
    
    def pause(self) -> bool:
        """暫停"""
        if self.mqtt and self.mqtt.pause():
            self._is_paused = True
            self._notify_status("paused", "已暫停")
            return True
        return False
    
    def resume(self) -> bool:
        """繼續"""
        if self.mqtt and self.mqtt.resume():
            self._is_paused = False
            self._notify_status("moving", "已繼續移動")
            return True
        return False
    
    def stop(self) -> bool:
        """停止（取消跟隨/導航）"""
        if self.mqtt:
            self.mqtt.stop_follow_mode()
            self._is_moving = False
            self._is_following = False
            self._notify_status("stopped", "已停止")
            return True
        return False
    
    def update_position(self, position: str) -> None:
        """
        更新當前位置
        
        Args:
            position: 位置代號或病房號碼
        """
        # 轉換為代號
        if len(position) == 1 and position.upper() in self.station_mapper.STATIONS:
            code = position.upper()
        else:
            code = self.station_mapper.room_to_code(position) or position
        
        self._current_position = code
        name = self.station_mapper.code_to_name(code)
        
        if self.on_position_change:
            self.on_position_change(code, name)
        
        logger.info(f"Position updated: {code} ({name})")
    
    def execute_action(self, action: NursingCartAction) -> bool:
        """
        執行護理車動作
        
        Args:
            action: 解析後的動作
        
        Returns:
            是否成功
        """
        command = action.command
        
        if command == 'navigate':
            return self.navigate_to(action.destination)
        
        elif command == 'return':
            return self.return_to_station()
        
        elif command == 'follow_start':
            return self.start_follow()
        
        elif command == 'follow_stop':
            return self.stop_follow()
        
        elif command == 'pause':
            return self.pause()
        
        elif command == 'resume':
            return self.resume()
        
        elif command == 'stop':
            return self.stop()
        
        elif command in ['status', 'battery', 'position', 'chat']:
            # 查詢類指令不需執行動作
            return True
        
        else:
            logger.warning(f"Unknown command: {command}")
            return False
    
    def _notify_status(self, status: str, message: str) -> None:
        """通知狀態變更"""
        if self.on_status_change:
            self.on_status_change(status, message)
    
    @property
    def current_position(self) -> str:
        """當前位置代號"""
        return self._current_position
    
    @property
    def current_position_name(self) -> str:
        """當前位置名稱"""
        return self.station_mapper.code_to_name(self._current_position)
    
    @property
    def is_connected(self) -> bool:
        """是否已連接"""
        return self.mqtt.is_connected if self.mqtt else False
    
    @property
    def is_moving(self) -> bool:
        """是否移動中"""
        return self._is_moving
    
    @property
    def is_following(self) -> bool:
        """是否跟隨中"""
        return self._is_following


def create_thouzer_controller(**kwargs) -> ThouzerController:
    """建立 Thouzer 控制器"""
    return ThouzerController(**kwargs)


# 測試
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    def on_status(status, msg):
        print(f"[Status] {status}: {msg}")
    
    def on_position(code, name):
        print(f"[Position] {code}: {name}")
    
    print("=== Thouzer 控制器測試 ===")
    
    controller = create_thouzer_controller(
        on_status_change=on_status,
        on_position_change=on_position
    )
    
    print(f"Current position: {controller.current_position_name}")
    print(f"MQTT available: {controller.mqtt is not None}")
    
    # 測試路徑查詢
    room = "3205"
    route = controller.station_mapper.get_route_for_destination('O', room)
    if route:
        print(f"Route from O to {room}: {route[0]} -> B{route[1]}")
