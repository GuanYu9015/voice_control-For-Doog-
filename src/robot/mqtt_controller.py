"""
MQTT 載具控制器

負責與 Thouzer 載具通訊，透過 MQTT 發送控制指令。

設計說明：
- 從 .env 讀取連線設定
- 支援記憶軌跡執行、跟隨模式、暫停/繼續等指令
"""
import json
import logging
import os
import time
from pathlib import Path
from typing import Callable, Optional

try:
    from paho.mqtt import client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("Warning: paho-mqtt not installed, run: pip install paho-mqtt")

# 嘗試載入 dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


class MQTTController:
    """
    MQTT 載具控制器
    
    負責與 Thouzer 載具通訊。
    """
    
    def __init__(
        self,
        broker_host: str = None,
        broker_port: int = None,
        username: str = None,
        password: str = None,
        thouzer_id: str = None,
        on_status: Optional[Callable[[dict], None]] = None,
        config_path: str = "config/robot_config.yaml"
    ):
        """
        初始化 MQTT 控制器
        
        Args:
            broker_host: MQTT Broker 主機
            broker_port: MQTT Broker 埠
            username: 使用者名稱
            password: 密碼
            thouzer_id: 載具識別碼
            on_status: 狀態回調函式
            config_path: 設定檔路徑
        """
        # 從設定檔讀取
        config = self._load_config(config_path)
        mqtt_cfg = config.get('mqtt', {})
        broker_cfg = mqtt_cfg.get('broker', {})
        auth_cfg = mqtt_cfg.get('auth', {})
        robot_cfg = config.get('robot', {}).get('thouzer', {})
        
        # 優先使用參數，其次設定檔，最後環境變數
        self.broker_host = broker_host or broker_cfg.get('host') or os.getenv('MQTT_BROKER_HOST', '192.168.212.1')
        self.broker_port = broker_port or broker_cfg.get('port') or int(os.getenv('MQTT_BROKER_PORT', 1883))
        self.username = username or auth_cfg.get('username') or os.getenv('MQTT_USERNAME', 'mqtt')
        self.password = password or auth_cfg.get('password') or os.getenv('MQTT_PASSWORD', '')
        self.thouzer_id = thouzer_id or robot_cfg.get('device_id') or os.getenv('THOUZER_ID', 'RMS-10C2-AAY67')
        
        self.on_status = on_status
        
        # MQTT 主題
        self.topic_cmd = f"0/THOUZER_HW/{self.thouzer_id}/exec/cmd"
        self.topic_status = f"0/WHISPERER/{self.thouzer_id}/app_status"
        
        # 狀態
        self._client = None
        self._connected = False
        self._last_status = {}
        
        self._init_client()
    
    def _load_config(self, config_path: str) -> dict:
        """從 YAML 設定檔讀取設定"""
        import yaml
        try:
            path = Path(config_path)
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
        return {}
    
    def _init_client(self) -> None:
        """初始化 MQTT 客戶端"""
        if not MQTT_AVAILABLE:
            logger.error("paho-mqtt not available")
            return
        
        try:
            client_id = f"voice_control_{int(time.time())}"
            
            # 支援 paho-mqtt 2.0 API
            try:
                # paho-mqtt >= 2.0
                self._client = mqtt.Client(
                    client_id=client_id,
                    callback_api_version=mqtt.CallbackAPIVersion.VERSION1
                )
            except (TypeError, AttributeError):
                # paho-mqtt < 2.0
                self._client = mqtt.Client(client_id)
            self._client.username_pw_set(self.username, self.password)
            
            # 設定回調
            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect
            self._client.on_message = self._on_message
            self._client.on_publish = self._on_publish
            
            logger.info(f"MQTT client initialized (broker: {self.broker_host}:{self.broker_port})")
            
        except Exception as e:
            logger.error(f"Failed to initialize MQTT client: {e}")
    
    def _on_connect(self, client, userdata, flags, rc):
        """連接回調"""
        if rc == 0:
            self._connected = True
            logger.info(f"✓ Connected to MQTT Broker ({self.broker_host})")
            
            # 訂閱狀態主題
            client.subscribe(self.topic_status)
        else:
            self._connected = False
            logger.error(f"✗ Connection failed, rc: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """斷線回調"""
        self._connected = False
        logger.warning(f"Disconnected from MQTT Broker, rc: {rc}")
    
    def _on_message(self, client, userdata, msg):
        """訊息回調"""
        try:
            payload = json.loads(msg.payload.decode())
            self._last_status = payload
            
            if self.on_status:
                self.on_status(payload)
                
        except Exception as e:
            logger.error(f"Message parse error: {e}")
    
    def _on_publish(self, client, userdata, mid):
        """發布回調"""
        logger.debug(f"Message published, mid: {mid}")
    
    def connect(self) -> bool:
        """建立連接"""
        if not MQTT_AVAILABLE or self._client is None:
            return False
        
        try:
            self._client.connect(self.broker_host, self.broker_port, 60)
            self._client.loop_start()
            
            # 等待連接
            for _ in range(30):  # 最多等 3 秒
                if self._connected:
                    return True
                time.sleep(0.1)
            
            return False
            
        except Exception as e:
            logger.error(f"MQTT connection error: {e}")
            return False
    
    def disconnect(self) -> None:
        """斷開連接"""
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            self._connected = False
    
    def publish(self, message: dict) -> bool:
        """
        發布訊息到指令主題
        
        Args:
            message: 訊息字典
        
        Returns:
            是否發送成功
        """
        if not self._connected:
            logger.warning("Not connected to MQTT")
            return False
        
        try:
            payload = json.dumps(message)
            result = self._client.publish(self.topic_cmd, payload)
            logger.info(f"→ Published: {payload}")
            return result.rc == mqtt.MQTT_ERR_SUCCESS
            
        except Exception as e:
            logger.error(f"Publish error: {e}")
            return False
    
    # ========================================
    # 載具控制指令
    # ========================================
    
    def execute_route(self, route_number: int) -> bool:
        """
        執行記憶軌跡
        
        Args:
            route_number: 軌跡編號 (如 350)
        
        Returns:
            是否發送成功
        """
        map_name = f"B{route_number}"
        message = {
            "app": "app-memorytrace",
            "params": f"--map {map_name}"
        }
        return self.publish(message)
    
    def start_follow_mode(self) -> bool:
        """啟動跟隨模式"""
        return self.publish({"app": "app-karugamo"})
    
    def stop_follow_mode(self) -> bool:
        """停止跟隨模式"""
        return self.publish({"app": ""})
    
    def pause(self) -> bool:
        """暫停"""
        return self.publish({"app": "#pause"})
    
    def resume(self) -> bool:
        """繼續"""
        return self.publish({"app": "#run"})
    
    def set_hold(self) -> bool:
        """設定 Hold 點"""
        return self.publish({"app": "# set_suspend"})
    
    def memory_save(self) -> bool:
        """記憶存檔"""
        return self.publish({"app": "memory-save"})
    
    def start_memory_mode(self, route_number: int) -> bool:
        """
        啟動記憶建置模式
        
        Args:
            route_number: 軌跡編號
        """
        map_name = f"B{route_number}"
        message = {
            "app": "memory-start-cancel",
            "params": f"--nk --map {map_name}"
        }
        return self.publish(message)
    
    @property
    def is_connected(self) -> bool:
        """是否已連接"""
        return self._connected
    
    @property
    def last_status(self) -> dict:
        """最後狀態"""
        return self._last_status


def create_mqtt_controller(**kwargs) -> Optional[MQTTController]:
    """建立 MQTT 控制器工廠函式"""
    if not MQTT_AVAILABLE:
        logger.warning("paho-mqtt not installed")
        return None
    
    return MQTTController(**kwargs)


# 測試
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== MQTT 控制器測試 ===")
    print(f"paho-mqtt available: {MQTT_AVAILABLE}")
    
    controller = create_mqtt_controller()
    if controller:
        print(f"Broker: {controller.broker_host}:{controller.broker_port}")
        print(f"Thouzer ID: {controller.thouzer_id}")
        print(f"Command topic: {controller.topic_cmd}")
        
        # 測試連接
        if controller.connect():
            print("✓ Connected")
            controller.disconnect()
        else:
            print("✗ Connection failed")
