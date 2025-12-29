"""
病房站點與軌跡映射模組

負責：
1. 病房號碼 ↔ 英文代號轉換
2. 起終點組合 → 軌跡編號轉換
3. 從設定檔載入站點資訊

【重要】站點設定的唯一來源：config/robot_config.yaml
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from pathlib import Path
import yaml


@dataclass
class Station:
    """站點資訊"""
    code: str           # 英文代號 (A, B, C, ...)
    name: str           # 中文名稱 (1號病房, 護理站)
    room_number: str    # 病房號碼 (3201, O)
    number: int         # 病房數字編號 (1-17, 0=護理站)
    image: str          # 圖片檔名
    position: Tuple[float, float]  # 相對位置 (x, y)


class StationMapper:
    """
    站點映射器
    
    提供病房號碼、英文代號、軌跡編號之間的轉換。
    設定從 config/robot_config.yaml 載入。
    """
    
    # 軌跡編號映射：起終點代號 → 軌跡編號
    ROUTE_MAPPING: Dict[str, int] = {
        # O 開頭（從各病房出發到護理站）
        'AO': 350, 'BO': 351, 'CO': 352, 'DO': 353, 'EO': 354, 'FO': 355, 'GO': 356, 'HO': 357,
        'IO': 358, 'JO': 359, 'KO': 360, 'LO': 361, 'MO': 362, 'NO': 363, 'PO': 364, 'QO': 365, 'RO': 366,
        # A 結尾（從各處到病房3201）
        'BA': 367, 'CA': 368, 'DA': 369, 'EA': 370, 'FA': 371, 'GA': 372, 'HA': 373, 'IA': 374,
        'JA': 375, 'KA': 376, 'LA': 377, 'MA': 378, 'NA': 379, 'OA': 380, 'PA': 381, 'QA': 382, 'RA': 383,
        # B 結尾
        'AB': 384, 'CB': 385, 'DB': 386, 'EB': 387, 'FB': 388, 'GB': 389, 'HB': 390, 'IB': 391,
        'JB': 392, 'KB': 393, 'LB': 394, 'MB': 395, 'NB': 396, 'OB': 397, 'PB': 398, 'QB': 399, 'RB': 400,
        # C 結尾
        'AC': 401, 'BC': 402, 'DC': 403, 'EC': 404, 'FC': 405, 'GC': 406, 'HC': 407, 'IC': 408,
        'JC': 409, 'KC': 410, 'LC': 411, 'MC': 412, 'NC': 413, 'OC': 414, 'PC': 415, 'QC': 416, 'RC': 417,
        # D 結尾
        'AD': 418, 'BD': 419, 'CD': 420, 'ED': 421, 'FD': 422, 'GD': 423, 'HD': 424, 'ID': 425,
        'JD': 426, 'KD': 427, 'LD': 428, 'MD': 429, 'ND': 430, 'OD': 431, 'PD': 432, 'QD': 433, 'RD': 434,
        # E 結尾
        'AE': 435, 'BE': 436, 'CE': 437, 'DE': 438, 'FE': 439, 'GE': 440, 'HE': 441, 'IE': 442,
        'JE': 443, 'KE': 444, 'LE': 445, 'ME': 446, 'NE': 447, 'OE': 448, 'PE': 449, 'QE': 450, 'RE': 451,
        # F 結尾
        'AF': 452, 'BF': 453, 'CF': 454, 'DF': 455, 'EF': 456, 'GF': 457, 'HF': 458, 'IF': 459,
        'JF': 460, 'KF': 461, 'LF': 462, 'MF': 463, 'NF': 464, 'OF': 465, 'PF': 466, 'QF': 467, 'RF': 468,
        # G 結尾
        'AG': 469, 'BG': 470, 'CG': 471, 'DG': 472, 'EG': 473, 'FG': 474, 'HG': 475, 'IG': 476,
        'JG': 477, 'KG': 478, 'LG': 479, 'MG': 480, 'NG': 481, 'OG': 482, 'PG': 483, 'QG': 484, 'RG': 485,
        # H 結尾
        'AH': 486, 'BH': 487, 'CH': 488, 'DH': 489, 'EH': 490, 'FH': 491, 'GH': 492, 'IH': 493,
        'JH': 494, 'KH': 495, 'LH': 496, 'MH': 497, 'NH': 498, 'OH': 499, 'PH': 500, 'QH': 501, 'RH': 502,
        # I 結尾
        'AI': 503, 'BI': 504, 'CI': 505, 'DI': 506, 'EI': 507, 'FI': 508, 'GI': 509, 'HI': 510,
        'JI': 511, 'KI': 512, 'LI': 513, 'MI': 514, 'NI': 515, 'OI': 516, 'PI': 517, 'QI': 518, 'RI': 519,
        # J 結尾
        'AJ': 520, 'BJ': 521, 'CJ': 522, 'DJ': 523, 'EJ': 524, 'FJ': 525, 'GJ': 526, 'HJ': 527,
        'IJ': 528, 'KJ': 529, 'LJ': 530, 'MJ': 531, 'NJ': 532, 'OJ': 533, 'PJ': 534, 'QJ': 535, 'RJ': 536,
        # K 結尾
        'AK': 537, 'BK': 538, 'CK': 539, 'DK': 540, 'EK': 541, 'FK': 542, 'GK': 543, 'HK': 544,
        'IK': 545, 'JK': 546, 'LK': 547, 'MK': 548, 'NK': 549, 'OK': 550, 'PK': 551, 'QK': 552, 'RK': 553,
        # L 結尾
        'AL': 554, 'BL': 555, 'CL': 556, 'DL': 557, 'EL': 558, 'FL': 559, 'GL': 560, 'HL': 561,
        'IL': 562, 'JL': 563, 'KL': 564, 'ML': 565, 'NL': 566, 'OL': 567, 'PL': 568, 'QL': 569, 'RL': 570,
        # M 結尾
        'AM': 571, 'BM': 572, 'CM': 573, 'DM': 574, 'EM': 575, 'FM': 576, 'GM': 577, 'HM': 578,
        'IM': 579, 'JM': 580, 'KM': 581, 'LM': 582, 'NM': 583, 'OM': 584, 'PM': 585, 'QM': 586, 'RM': 587,
        # N 結尾
        'AN': 588, 'BN': 589, 'CN': 590, 'DN': 591, 'EN': 592, 'FN': 593, 'GN': 594, 'HN': 595,
        'IN': 596, 'JN': 597, 'KN': 598, 'LN': 599, 'MN': 600, 'ON': 601, 'PN': 602, 'QN': 603, 'RN': 604,
        # P 結尾
        'AP': 605, 'BP': 606, 'CP': 607, 'DP': 608, 'EP': 609, 'FP': 610, 'GP': 611, 'HP': 612,
        'IP': 613, 'JP': 614, 'KP': 615, 'LP': 616, 'MP': 617, 'NP': 618, 'OP': 619, 'QP': 620, 'RP': 621,
        # Q 結尾
        'AQ': 622, 'BQ': 623, 'CQ': 624, 'DQ': 625, 'EQ': 626, 'FQ': 627, 'GQ': 628, 'HQ': 629,
        'IQ': 630, 'JQ': 631, 'KQ': 632, 'LQ': 633, 'MQ': 634, 'NQ': 635, 'OQ': 636, 'PQ': 637, 'RQ': 638,
        # R 結尾
        'AR': 639, 'BR': 640, 'CR': 641, 'DR': 642, 'ER': 643, 'FR': 644, 'GR': 645, 'HR': 646,
        'IR': 647, 'JR': 648, 'KR': 649, 'LR': 650, 'MR': 651, 'NR': 652, 'OR': 653, 'PR': 654, 'QR': 655,
    }
    
    def __init__(self, config_path: str = "config/robot_config.yaml"):
        """
        初始化映射器
        
        Args:
            config_path: 設定檔路徑
        """
        # 從設定檔載入站點
        self.STATIONS: Dict[str, Station] = {}
        self._room_to_code: Dict[str, str] = {}
        self._number_to_code: Dict[int, str] = {}
        
        self._load_config(config_path)
    
    def _load_config(self, config_path: str) -> None:
        """從設定檔載入站點"""
        try:
            path = Path(config_path)
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
                
                stations = config.get('stations', {})
                for code, data in stations.items():
                    station = Station(
                        code=code,
                        name=data.get('name', f'位置{code}'),
                        room_number=str(data.get('room', code)),
                        number=data.get('number', 0),
                        image=data.get('image', ''),
                        position=tuple(data.get('position', [0, 0]))
                    )
                    self.STATIONS[code] = station
                    self._room_to_code[station.room_number] = code
                    self._number_to_code[station.number] = code
                
                print(f"[StationMapper] 載入 {len(self.STATIONS)} 個站點")
        except Exception as e:
            print(f"[StationMapper] 載入設定失敗: {e}")
            # 使用預設值
            self._load_defaults()
    
    def _load_defaults(self) -> None:
        """載入預設站點（備用）"""
        default_stations = {
            'O': ('護理站', 'O', 0),
            'A': ('1號病房', '3201', 1), 'B': ('2號病房', '3202', 2),
            'C': ('3號病房', '3203', 3), 'D': ('4號病房', '3204', 4),
            'E': ('5號病房', '3205', 5), 'F': ('6號病房', '3206', 6),
            'G': ('7號病房', '3207', 7), 'H': ('8號病房', '3208', 8),
            'I': ('9號病房', '3209', 9), 'J': ('10號病房', '3210', 10),
            'K': ('11號病房', '3211', 11), 'L': ('12號病房', '3212', 12),
            'M': ('13號病房', '3213', 13), 'N': ('14號病房', '3214', 14),
            'P': ('15號病房', '3215', 15), 'Q': ('16號病房', '3216', 16),
            'R': ('17號病房', '3217', 17),
        }
        for code, (name, room, number) in default_stations.items():
            station = Station(code, name, room, number, '', (0, 0))
            self.STATIONS[code] = station
            self._room_to_code[room] = code
            self._number_to_code[number] = code
    
    def room_to_code(self, room_number: str) -> Optional[str]:
        """
        病房號碼轉英文代號
        
        Args:
            room_number: 病房號碼 (如 "3201") 或數字 (如 "1")
        
        Returns:
            英文代號 (如 "A") 或 None
        """
        # 處理護理站
        if room_number.upper() in ['O', '護理站', '护理站']:
            return 'O'
        
        # 直接查詢病房號碼
        if room_number in self._room_to_code:
            return self._room_to_code[room_number]
        
        # 嘗試數字查詢（1-17）
        try:
            num = int(room_number)
            if num in self._number_to_code:
                return self._number_to_code[num]
        except ValueError:
            pass
        
        return None
    
    def number_to_code(self, number: int) -> Optional[str]:
        """
        數字病房編號轉英文代號
        
        Args:
            number: 病房數字 (1-17, 0=護理站)
        
        Returns:
            英文代號
        """
        return self._number_to_code.get(number)
    
    def code_to_room(self, code: str) -> Optional[str]:
        """
        英文代號轉病房號碼
        
        Args:
            code: 英文代號 (如 "A")
        
        Returns:
            病房號碼 (如 "3201") 或 None
        """
        station = self.STATIONS.get(code.upper())
        return station.room_number if station else None
    
    def code_to_name(self, code: str) -> str:
        """
        英文代號轉站點名稱
        
        Args:
            code: 英文代號
        
        Returns:
            站點名稱 (如 "1號病房")
        """
        station = self.STATIONS.get(code.upper())
        return station.name if station else f"位置{code}"
    
    def get_route_number(self, start_code: str, end_code: str) -> Optional[int]:
        """
        取得軌跡編號
        
        Args:
            start_code: 起點英文代號
            end_code: 終點英文代號
        
        Returns:
            軌跡編號 或 None
        """
        route_key = f"{start_code.upper()}{end_code.upper()}"
        return self.ROUTE_MAPPING.get(route_key)
    
    def get_route_for_destination(
        self, 
        current_position: str, 
        destination_room: str
    ) -> Optional[Tuple[str, int]]:
        """
        根據當前位置和目標病房取得軌跡
        
        Args:
            current_position: 當前位置代號
            destination_room: 目標病房號碼
        
        Returns:
            (軌跡代碼, 軌跡編號) 或 None
        """
        dest_code = self.room_to_code(destination_room)
        if not dest_code:
            return None
        
        route_number = self.get_route_number(current_position, dest_code)
        if route_number is None:
            return None
        
        route_code = f"{current_position.upper()}{dest_code}"
        return (route_code, route_number)
    
    def get_station(self, code: str) -> Optional[Station]:
        """取得站點資訊"""
        return self.STATIONS.get(code.upper())
    
    @property
    def all_room_numbers(self) -> list:
        """取得所有病房號碼"""
        return [s.room_number for s in self.STATIONS.values() if s.room_number != 'O']


# 全域實例
_mapper: StationMapper = None


def get_station_mapper() -> StationMapper:
    """取得全域 StationMapper 實例"""
    global _mapper
    if _mapper is None:
        _mapper = StationMapper()
    return _mapper


# 測試
if __name__ == "__main__":
    mapper = get_station_mapper()
    
    print("=== 站點映射測試 ===")
    print(f"載入站點數: {len(mapper.STATIONS)}")
    print(f"病房號碼 3201 → 代號: {mapper.room_to_code('3201')}")
    print(f"數字 3 → 代號: {mapper.number_to_code(3)}")
    print(f"代號 A → 病房號碼: {mapper.code_to_room('A')}")
    print(f"代號 A → 名稱: {mapper.code_to_name('A')}")
    print(f"軌跡 O→A: {mapper.get_route_number('O', 'A')}")
    print(f"從護理站到3201: {mapper.get_route_for_destination('O', '3201')}")
