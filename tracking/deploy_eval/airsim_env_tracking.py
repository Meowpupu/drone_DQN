"""
AirSim 環境：YOLO 穿越兩個黃框 -> 觸碰粉球 (v3，依 Table I / Table II 規格)
==========================================================================
狀態 (30 維，已正規化):
  [0:2]   中心偏移 (dx_img, dy_img)
  [2:4]   追蹤比例 (ratio_current, ratio_initial)   — bbox 面積佔畫面比例，取 sqrt 後 clip [0,1]
  [4:7]   世界座標 NED (x, y, z)                       — 與 simGetObjectPose 同一座標系
  [7:10]  避障向量 (d_left, d_center, d_right)        — 由前視深度圖左/中/右三區取得
  [10:14] 姿態四元數 (w, x, y, z)
  [14:17] 線速度   [17:20] 角速度   [20:23] 線加速度
  [23:27] 馬達推力 (F1..F4)   [27:30] 機身力矩估計 (Mx, My, Mz)
動作 (Table II): 13 個離散動作 (0~11 為運動動作，12 為 Hover/Brake)，見 ACTION_TABLE。

獎勵核心：以模擬器真實座標計算的 potential-based shaping（只用於算獎勵，不進 state），
並以「穿越平面 + 孔徑半徑」判定真正穿過黃框，而非只靠像素置中。
"""

import math
import os
import time

import airsim
import cv2
import numpy as np
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
YOLO_WEIGHTS = os.path.join(HERE, "best.pt")

# ===================== UE5 場景物件 =====================
# AirSim API 使用的是 Actor「內部名稱」(StaticMeshActor_UAID_...)，不是 Outliner 的 Label。
# 兩個黃框的先後順序不看 Label，由程式依「離粉球距離」自動排序（最遠者先穿）。
COURSE_OBJECTS = {
    "yellow_a": "StaticMeshActor_UAID_D03C1F5659FE2EF902_2097138535",   # 編輯器 Label: YellowFrame1
    "yellow_b": "StaticMeshActor_UAID_D03C1F5659FE2FF902_1400802713",   # 編輯器 Label: YellowFrame2
    "ball":     "StaticMeshActor_UAID_D03C1F5659FE2FF902_1251224712",   # 編輯器 Label: PinkBall (scale 3)
}

# ===================== 需要依實際場景確認的幾何參數 =====================
# 黃框實測（Static Mesh 編輯器 Front 視圖）：外框 6.0 x 12.0 m、厚 0.8 m、邊框約 0.43 m
#   -> 洞淨空約 5.1 m 寬 x 11.1 m 高。穿越點需落在「洞半尺寸 - 機身半徑」之內（假設網格原點在框中心，待實測確認）。
GATE_HOLE_WIDTH = 5.1
GATE_HOLE_HEIGHT = 11.1
DRONE_HALF_SIZE = 0.3
GATE_HALF_WIDTH = GATE_HOLE_WIDTH / 2 - DRONE_HALF_SIZE     # 橫向容許偏移 (m)
GATE_HALF_HEIGHT = GATE_HOLE_HEIGHT / 2 - DRONE_HALF_SIZE   # 垂直容許偏移 (m)
VEHICLE_NAME = "Drone1"   # settings.json 的載具名稱；用來量測「載具座標系 → 世界座標系」位移
BALL_CONTACT_DIST = 2.0    # 球心距離上限 (m)：引擎 Sphere 半徑 0.5 m x scale 3 = 1.5 m，再加機身餘裕 0.5 m

# ===================== 控制 / 安全參數 =====================
STEP_DURATION = 0.5
BRAKE_GAIN = 0.045            # hover_brake：每 1 m/s 的水平速度，反向傾斜多少弧度（約 2.6°）
BRAKE_MAX_TILT = math.radians(10)
MAX_FORWARD_SPEED = 4.0       # m/s；超過後前進類動作的俯仰歸零（本版本沒有倒退動作，需限制速度才煞得住）
COMMAND_HOLD_SECONDS = 30.0   # 指令有效時間（遠大於一步；下一步指令會覆蓋）
HOVER_THROTTLE = 0.595   # 實測懸停油門約 0.593
# 實測此 AirSim 版本：pitch 為負 → 往後退、yaw_rate 為負 → NED yaw 增加（順時針），與 Table II 語意相反，故在 API 呼叫處反號
PITCH_SIGN = -1.0
YAW_SIGN = -1.0
MAX_TILT_RAD = math.radians(30)     # Table II 的百分比以此為 100%
MIN_SAFE_ALTITUDE = 1.0
ALT_RECOVERY_BELOW_TARGET = 12.0    # 低於「最低目標高度 - 此值」時強制壓平姿態並加油門
ALT_CEILING_ABOVE_TARGET = 15.0
CORRIDOR_RADIUS = 20.0              # 距賽道中心線的最大 3D 垂直距離 (m)
CORRIDOR_MARGIN_ALONG = 15.0        # 賽道前後端額外允許的長度 (m)
MAX_TARGET_LOST_STEPS = 20
STALL_STEPS = 40                    # 連續這麼多步沒有「更接近目標」就視為原地空轉
STALL_MIN_IMPROVE = 0.25            # 判定「有進步」所需的最小距離縮短量 (m)
DEBUG_SAVE_YOLO_FRAME = False

# ===================== YOLO =====================
YELLOW_FRAME_CLASS_ID = 0
PINK_BALL_CLASS_ID = 1

# ===================== 動作表 (Table II) =====================
# (名稱, roll 比例, pitch 比例, 油門增量, yaw rate deg/s)；比例以 MAX_TILT_RAD 為 100%
ACTION_TABLE = [
    ("fast_forward",     0.0, -0.15, 0.00,   0.0),   # 0
    ("slow_forward",     0.0, -0.05, 0.00,   0.0),   # 1
    ("roll_left",       -0.12, 0.0,  0.00,   0.0),   # 2
    ("roll_right",       0.12, 0.0,  0.00,   0.0),   # 3
    ("ascend",           0.0,  0.0,  0.25,   0.0),   # 4
    ("descend",          0.0,  0.0, -0.25,   0.0),   # 5
    ("yaw_left",         0.0,  0.0,  0.00, -20.0),   # 6
    ("yaw_right",        0.0,  0.0,  0.00,  20.0),   # 7
    ("forward_yaw_left", 0.0, -0.08, 0.00, -15.0),   # 8
    ("forward_yaw_right", 0.0, -0.08, 0.00, 15.0),   # 9
    ("forward_ascend",   0.0, -0.10, 0.15,   0.0),   # 10
    ("forward_descend",  0.0, -0.10, -0.15,  0.0),   # 11
    ("hover_brake",      0.0,  0.0,  0.00,   0.0),   # 12
]
ENABLE_HOVER_ACTION = True   # 介面固定 13 個輸出（Table II 全部動作，含 Hover/Brake）。要改 12 個設為 False，會移除 index 12
NUM_ACTIONS = len(ACTION_TABLE) if ENABLE_HOVER_ACTION else len(ACTION_TABLE) - 1
ACTION_NAMES = {i: ACTION_TABLE[i][0] for i in range(NUM_ACTIONS)}


def default_action_prior():
    """隨機探索時的動作分佈：含前進分量的動作權重較高，Hover 較低，降低初期亂飛墜地的機率。"""
    w = []
    for name, roll, pitch, thr, yaw in ACTION_TABLE[:NUM_ACTIONS]:
        if pitch < 0:
            w.append(3.0)
        elif name == "hover_brake":
            w.append(0.5)
        else:
            w.append(1.0)
    w = np.array(w, dtype=np.float64)
    return w / w.sum()


# ===================== 狀態正規化尺度 =====================
OBSTACLE_MAX_RANGE = 20.0
VEL_SCALE = 8.0
ANGVEL_SCALE = 3.5
ACC_SCALE = 20.0
THRUST_MAX = 1.0       # F1..F4 已是 [0,1] 的油門指令
MOMENT_SCALE = 1.0     # Mx, My, Mz 已是 [-1,1] 的正規化指令
MAX_YAW_RATE = math.radians(20.0)
POS_MARGIN = 30.0     # 世界座標正規化：賽道物件範圍外再加的邊界 (m)

# ===================== 獎勵參數 =====================
GAMMA_SHAPING = 0.99
ALIVE_PENALTY = -0.02
ALIGN_BONUS_MAX = 0.1
STAGE_PASS_BONUS = 30.0
SUCCESS_REWARD = 100.0
# --- 「接近球並懸停」任務（ball_mode == "hover"）---
BALL_RADIUS = 1.5               # 球半徑 (m)：引擎 Sphere 半徑 0.5 m x scale 3
HOVER_SURFACE_MIN, HOVER_SURFACE_MAX = 2.0, 4.0   # 離「球表面」的目標距離區間 (m)
HOVER_STANDOFF = BALL_RADIUS + (HOVER_SURFACE_MIN + HOVER_SURFACE_MAX) / 2   # 理想停留距離（到球心）= 4.5 m
HOVER_ZONE_MIN = BALL_RADIUS + HOVER_SURFACE_MIN   # = 3.5 m（到球心）
HOVER_ZONE_MAX = BALL_RADIUS + HOVER_SURFACE_MAX   # = 5.5 m（到球心）
HOVER_MAX_SPEED = 0.5           # m/s；低於此速度才算懸停
HOVER_SECONDS = 10.0            # 需連續懸停的模擬時間
HOVER_REQUIRED_STEPS = int(round(HOVER_SECONDS / 0.5))   # = HOVER_SECONDS / STEP_DURATION
HOVER_MAX_YAW_ERR = math.radians(15.0)   # 懸停時機頭需對準球：水平方位誤差上限，且球必須在畫面內被偵測到
HOVER_YAW_POTENTIAL_WEIGHT = 2.0         # 潛在函數中「方位誤差」的權重 (m / rad)：90° 約等於 3.1 m 的距離
HOVER_STEP_BONUS = 0.3          # 在停留區且夠慢時，每步的小額獎勵（稠密訊號）
COLLISION_PENALTY = -50.0
GROUND_CRASH_PENALTY = -50.0
OUT_OF_BOUNDS_PENALTY = -50.0
TARGET_LOST_PENALTY = -30.0
MISSED_GATE_PENALTY = -30.0
STALL_PENALTY = -20.0
TIMEOUT_PENALTY = -10.0
CONTINUOUS_REWARD_CLIP = 3.0

# ===================== 課程學習關卡 =====================
# spawn: ("before_ball", 最大距離) 出生在球前方；("before_gate1", 距離) 出生在第一個黃框前方。
# 賽道方向由「黃框1 -> 粉球」連線決定（直線賽道）。
MISSION_STAGES = {
    1: {"targets": ["ball"],                  "max_steps": 150, "spawn": ("before_ball", 8.0),
        "jitter": (2.0, 1.5, 10.0)},
    2: {"targets": ["gate1"],                 "max_steps": 250, "spawn": ("before_gate1", 15.0),
        "jitter": (1.0, 1.0, 5.0)},
    3: {"targets": ["gate1", "gate2", "ball"], "max_steps": 450, "spawn": ("before_gate1", 15.0),
        "jitter": (1.0, 1.0, 5.0), "ball_mode": "hover"},   # 第 3 關：不可碰球，接近並懸停 10 秒
}
MAX_SPAWN_BACK = max(c["spawn"][1] for c in MISSION_STAGES.values())
# jitter = (側向 m, 垂直 m, 偏航 deg)，定義在各關卡設定內


def _vec(p):
    return np.array([p.x_val, p.y_val, p.z_val], dtype=np.float64)


class VisionTrackingEnv:
    def __init__(self, curriculum_stage=3, yolo_path=YOLO_WEIGHTS):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.state_size = 30
        self.action_size = NUM_ACTIONS
        self.curriculum_stage = curriculum_stage

        self.current_step = 0
        self.max_steps = MISSION_STAGES[curriculum_stage]["max_steps"]
        self.target_lost_counter = 0
        self._collision_baseline = self.client.simGetCollisionInfo().time_stamp

        self._mission_targets = []
        self._stage_idx = 0
        self._ratio_initial = 0.0
        self._prev_potential = 0.0
        self._prev_pos = None
        self._world_pos = np.zeros(3)
        self._frame_offset = np.zeros(3)   # 世界座標 = 載具座標 + 此位移（起點在 settings.json 的 X/Y/Z）
        self._last_altitude = 0.0
        self._best_dist = float("inf")
        self._hover_steps = 0
        self._yaw = 0.0
        self.last_gate_cross = None  # (橫向偏差 m, 垂直偏差 m)：最近一次穿越黃框平面時
        self.last_detections = []   # [(class_id, conf, cx, cy, w, h), ...] 最近一次觀測 YOLO 看到的全部框（像素）
        self._steps_since_best = 0
        self.last_raw_state = None
        self._last_cmd = (0.0, 0.0, 0.0, HOVER_THROTTLE)
        self._warned_small_image = False

        self.client.simPause(False)   # 上一次程式若異常結束，模擬器可能還停在暫停狀態
        self._load_course()
        self._clock_ratio = self._measure_clock_speed()
        print(f"模擬時間倍率 (ClockSpeed 實測): {self._clock_ratio:.2f}x -> 每步放行 {STEP_DURATION / self._clock_ratio:.3f} s 實際時間")

        print("正在載入 YOLO 模型...")
        self.yolo_model = YOLO(yolo_path)
        self.cls_frame, self.cls_ball = YELLOW_FRAME_CLASS_ID, PINK_BALL_CLASS_ID
        names = getattr(self.yolo_model, "names", None)
        if isinstance(names, dict):
            for idx, nm in names.items():          # 以權重檔內建的類別名稱為準，不信任寫死的編號
                if "frame" in str(nm).lower():
                    self.cls_frame = int(idx)
                elif "ball" in str(nm).lower():
                    self.cls_ball = int(idx)
            print(f"YOLO 類別: {names} -> 黃框={self.cls_frame}, 粉球={self.cls_ball}")
        print("YOLO 模型載入完成！")

    # ------------------------------------------------------------------
    # 賽道幾何（一次性讀取，物件是靜態的，不必每步 RPC）
    # ------------------------------------------------------------------
    def _load_course(self):
        # AirSim 兩套座標：simGetObjectPose 是世界座標（原點=PlayerStart），simGetVehiclePose / simSetVehiclePose
        # 是載具座標（原點=載具出生點，即 PlayerStart + settings.json 的 X/Y/Z）。本環境內部一律使用世界座標，
        # 位移實測 = 載具 actor 的世界位置 - 載具座標位置。
        veh_world = _vec(self.client.simGetObjectPose(VEHICLE_NAME).position)
        if np.any(np.isnan(veh_world)):
            raise RuntimeError(f"simGetObjectPose('{VEHICLE_NAME}') 回傳 NaN，請確認 VEHICLE_NAME 與 settings.json 的載具名稱一致。")
        self._frame_offset = veh_world - _vec(self.client.simGetVehiclePose().position)
        print("座標系位移 (世界 = 載具 + offset): ", np.round(self._frame_offset, 2))

        raw, missing = {}, []
        for key, name in COURSE_OBJECTS.items():
            vec = _vec(self.client.simGetObjectPose(name).position)
            if np.any(np.isnan(vec)):
                missing.append(name)
            else:
                raw[key] = vec
        if missing:
            hints = [n for n in self.client.simListSceneObjects() if n.startswith("StaticMeshActor")]
            raise RuntimeError(
                f"場景中找不到物件 {missing}。AirSim 使用內部名稱，請用 find_course_objects.py 對照位置後修改 COURSE_OBJECTS。"
                f"\nStaticMeshActor 候選: {hints}")

        # 依離粉球的距離由遠到近排序：最遠的黃框是第一個要穿的 (gate1)
        order = sorted(("yellow_a", "yellow_b"), key=lambda k: -np.linalg.norm(raw[k] - raw["ball"]))
        self.course_names = {"gate1": COURSE_OBJECTS[order[0]], "gate2": COURSE_OBJECTS[order[1]],
                             "ball": COURSE_OBJECTS["ball"]}
        poses = {"gate1": raw[order[0]], "gate2": raw[order[1]], "ball": raw["ball"]}

        self.course_pos = poses
        span = poses["ball"] - poses["gate1"]
        self.course_length = float(np.linalg.norm(span))
        if self.course_length < 1e-3:
            raise RuntimeError("第一個黃框與粉球位置重合，無法決定賽道方向。")
        self.axis = span / self.course_length

        rel = poses["gate2"] - poses["gate1"]
        off_line = rel - np.dot(rel, self.axis) * self.axis
        if np.linalg.norm(off_line) > 3.0:
            print(f"⚠️ 第二個黃框偏離「第一框->粉球」連線 {np.linalg.norm(off_line):.1f} m，賽道可能不是直線，"
                  f"穿越判定假設黃框平面垂直於此連線。")
        if np.dot(rel, self.axis) <= 0:
            print("⚠️ 第二個黃框不在第一框與粉球之間，請檢查場景佈局。")

        pts = np.stack(list(poses.values()))
        self.pos_center = pts.mean(axis=0)
        self.pos_scale = np.maximum(np.abs(pts - self.pos_center).max(axis=0) + POS_MARGIN, POS_MARGIN)

        alts = -pts[:, 2]
        self._alt_recover = max(MIN_SAFE_ALTITUDE + 2.0, float(alts.min()) - ALT_RECOVERY_BELOW_TARGET)
        self._alt_ceiling = float(alts.max()) + ALT_CEILING_ABOVE_TARGET

        print("賽道資訊（飛行方向單位向量 %s）:" % np.round(self.axis, 2))
        for k in ("gate1", "gate2", "ball"):
            v = poses[k]
            print(f"  {k:5s}: NED = ({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})  [{self.course_names[k][-24:]}]")
        print(f"  賽道長度 {self.course_length:.1f} m | 高度回復線 {self._alt_recover:.1f} m | 高度上限 {self._alt_ceiling:.1f} m")

    def _measure_clock_speed(self):
        """simContinueForTime(x) 是「實際時間 x 秒」，模擬時間會被 ClockSpeed 放大；量測倍率，讓每步固定 STEP_DURATION 秒模擬時間。"""
        try:
            self.client.simPause(True)
            t0 = self.client.getMultirotorState().timestamp
            self.client.simContinueForTime(0.2)
            t1 = self.client.getMultirotorState().timestamp
            self.client.simPause(False)
            ratio = (t1 - t0) / 1e9 / 0.2
            if 0.2 <= ratio <= 50:
                return float(ratio)
        except Exception as e:
            print(f"⚠️ 量測 ClockSpeed 失敗 ({e})，假設為 1")
            self.client.simPause(False)
        return 1.0

    def set_curriculum_stage(self, stage: int):
        assert stage in MISSION_STAGES, f"未知課程階段: {stage}"
        self.curriculum_stage = stage

    def _make_target(self, key):
        return {
            "key": key,
            "name": self.course_names[key],
            "type": "ball" if key == "ball" else "gate",
            "pos": self.course_pos[key],
        }

    def _current_target(self):
        return self._mission_targets[self._stage_idx] if self._stage_idx < len(self._mission_targets) else None

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _sample_spawn(self, cfg):
        kind, dist = cfg["spawn"]
        if kind == "before_ball":
            gap = float(np.linalg.norm(self.course_pos["ball"] - self.course_pos["gate2"]))
            base = self.course_pos["ball"] - self.axis * min(dist, 0.9 * gap)
        else:
            base = self.course_pos["gate1"] - self.axis * dist

        lateral = np.array([-self.axis[1], self.axis[0], 0.0])
        lateral = lateral / (np.linalg.norm(lateral) + 1e-9)
        jit_lat, jit_vert, jit_yaw = cfg["jitter"]
        spawn = base + lateral * np.random.uniform(-jit_lat, jit_lat)
        spawn[2] += np.random.uniform(-jit_vert, jit_vert)

        yaw = math.atan2(self.axis[1], self.axis[0]) + math.radians(np.random.uniform(-jit_yaw, jit_yaw))
        return spawn, yaw

    def reset(self):
        self.client.simPause(False)
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()

        cfg = MISSION_STAGES[self.curriculum_stage]
        self._mission_targets = [self._make_target(k) for k in cfg["targets"]]
        self._stage_idx = 0
        self.max_steps = cfg["max_steps"]

        spawn, yaw = self._sample_spawn(cfg)
        pose = airsim.Pose(airsim.Vector3r(*(spawn - self._frame_offset)), airsim.to_quaternion(0.0, 0.0, yaw))
        for attempt in range(3):
            self.client.simSetVehiclePose(pose, True)
            # simSetVehiclePose 是異步生效（等遊戲執行緒下一個 tick）；ClockSpeed 高時指令等待太短，
            # 若不先等位置真的到位，後面讀到的是舊位置、傳送也可能被後續指令覆蓋。所以輪詢等待。
            if self._wait_until_at(spawn, tol=1.0, timeout_s=4.0):
                break
            print(f"⚠️ 傳送後位置未到位，重試 ({attempt + 1}/3)")
        self._settle_velocity()
        self.client.moveByRollPitchYawrateThrottleAsync(0.0, 0.0, 0.0, HOVER_THROTTLE, 0.3).join()
        self.client.simPause(True)

        self.current_step = 0
        self.last_gate_cross = None
        self.target_lost_counter = 0
        self._collision_baseline = self.client.simGetCollisionInfo().time_stamp
        self._ratio_initial = 0.0
        self._last_cmd = (0.0, 0.0, 0.0, HOVER_THROTTLE)

        state = self._get_state()
        self._prev_pos = self._world_pos.copy()
        self._reset_segment_progress()
        self._prev_potential = self._compute_potential()
        return state

    def _hover_mode(self):
        target = self._current_target()
        return (target is not None and target["type"] == "ball"
                and MISSION_STAGES[self.curriculum_stage].get("ball_mode", "touch") == "hover")

    def _yaw_error_to_target(self):
        """機頭方向與「無人機 -> 目標」水平方位的夾角（rad，已正規化到 [-pi, pi]）。"""
        t = self._current_target()
        if t is None:
            return 0.0
        rel = t["pos"] - self._world_pos
        err = math.atan2(rel[1], rel[0]) - self._yaw
        return (err + math.pi) % (2 * math.pi) - math.pi

    def _progress_distance(self):
        """潛在函數與空轉偵測使用的「距離目標還有多遠」；懸停任務以停留距離為目標，並加入機頭對準球的項。"""
        d = self._distance_to_current_target()
        if self._hover_mode():
            return abs(d - HOVER_STANDOFF) + HOVER_YAW_POTENTIAL_WEIGHT * abs(self._yaw_error_to_target())
        return d

    def _wait_until_at(self, target_world, tol=1.0, timeout_s=4.0):
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            here = _vec(self.client.simGetVehiclePose().position) + self._frame_offset
            if np.linalg.norm(here - target_world) < tol:
                return True
            time.sleep(0.05)
        return False

    def _settle_velocity(self, tol=0.1, timeout_s=4.0):
        """傳送不會清掉速度；油門是開環控制，殘留的垂直速度會讓高度一路漂移。用 hover 模式把速度壓到接近 0 再開始回合。"""
        # 不能用 hoverAsync：那是「定點保持」，會把剛傳送的無人機拉回舊位置。改用速度控制到 0（只壓速度，不拉位置）。
        self.client.moveByVelocityAsync(0.0, 0.0, 0.0, 3.0)
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            v = self.client.getMultirotorState().kinematics_estimated.linear_velocity
            if (v.x_val ** 2 + v.y_val ** 2 + v.z_val ** 2) ** 0.5 < tol:
                return
            time.sleep(0.05)

    def _reset_segment_progress(self):
        self._hover_steps = 0
        self._best_dist = self._progress_distance()
        self._steps_since_best = 0

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action, substeps=1, on_substep=None):
        """substeps > 1 時把這一步（STEP_DURATION 模擬秒）切成幾小段，每段結束呼叫 on_substep(k)；總模擬時間不變（錄影用）。"""
        self._execute_action(action, substeps=substeps, on_substep=on_substep)
        self.current_step += 1
        state = self._get_state()
        reward, done, info = self._compute_reward()
        self._prev_pos = self._world_pos.copy()
        return state, reward, done, info

    # ------------------------------------------------------------------
    # 感知
    # ------------------------------------------------------------------
    def grab_scene(self):
        """只擷取場景圖（BGR，畫面錄影用，比 _grab_images 快，不抓深度）。"""
        r = self.client.simGetImages([airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)])[0]
        buf = np.frombuffer(r.image_data_uint8, dtype=np.uint8)
        ch = buf.size // (r.width * r.height)
        return buf.reshape(r.height, r.width, ch)[:, :, :3].copy()

    def _grab_images(self):
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
                airsim.ImageRequest("front_center", airsim.ImageType.DepthPerspective, True, False),
            ])
        except Exception as e:
            print(f"⚠️ simGetImages 失敗: {e}")
            return None, None

        scene, depth = None, None
        if responses and len(responses) >= 1:
            r = responses[0]
            if r.width > 0 and r.height > 0:
                buf = np.frombuffer(r.image_data_uint8, dtype=np.uint8)
                ch = buf.size // (r.width * r.height)
                if ch >= 3:
                    scene = buf.reshape(r.height, r.width, ch)[:, :, :3]
        if responses and len(responses) >= 2:
            r = responses[1]
            if r.width > 0 and r.height > 0 and len(r.image_data_float) == r.width * r.height:
                depth = np.array(r.image_data_float, dtype=np.float32).reshape(r.height, r.width)
        return scene, depth

    def _vision_features(self, scene):
        """回傳 [dx, dy, ratio_current(面積佔比)]；找不到目標時為 [0, 0, 0] 並累計丟失計數。"""
        target_box, img_w, img_h = None, 640, 480
        if scene is not None:
            img_h, img_w = scene.shape[:2]
            if img_w < 480 and not self._warned_small_image:
                print(f"⚠️ 相機解析度只有 {img_w}x{img_h}，YOLO 很可能偵測不到遠處目標；請在 settings.json 設定 CaptureSettings (建議 640x360 以上)。")
                self._warned_small_image = True
            result = self.yolo_model(scene, verbose=False, conf=0.25)[0]
            self.last_detections = [
                (int(b.cls[0].item()), float(b.conf[0].item()), *[float(v) for v in b.xywh[0].cpu().numpy()])
                for b in result.boxes]
            if DEBUG_SAVE_YOLO_FRAME:
                cv2.imwrite(os.path.join(HERE, "debug_yolo.jpg"), result.plot())

            target = self._current_target()
            wanted_cls = self.cls_ball if (target and target["type"] == "ball") else self.cls_frame
            boxes = [b for b in result.boxes if int(b.cls[0].item()) == wanted_cls]
            if boxes:
                boxes.sort(key=lambda b: float(b.xywh[0][2] * b.xywh[0][3]), reverse=True)
                target_box = boxes[0]

        if target_box is None:
            self.target_lost_counter += 1
            return [0.0, 0.0, 0.0]

        self.target_lost_counter = 0
        xc, yc, w, h = target_box.xywh[0].cpu().numpy()
        dx = (xc - img_w / 2) / (img_w / 2)
        dy = (yc - img_h / 2) / (img_h / 2)
        area_frac = float(w * h) / float(img_w * img_h)
        return [float(dx), float(dy), area_frac]

    @staticmethod
    def _depth_obstacles(depth):
        R = OBSTACLE_MAX_RANGE
        if depth is None:
            return [R, R, R]
        h = depth.shape[0]
        band = depth[int(h * 0.35):int(h * 0.65)]
        out = []
        for part in np.array_split(band, 3, axis=1):
            vals = part[np.isfinite(part)]
            d = float(np.percentile(vals, 10)) if vals.size else R
            out.append(min(max(d, 0.0), R))
        return out  # [left, center, right]

    def _rotor_forces_and_moments(self):
        """馬達推力指令 F1..F4 與機身力矩指令 (Mx, My, Mz)。
        此 UE5 版 AirSim 的 getRotorStates() 回傳亂碼，故改用「上一個動作的指令」經 X 型混控器換算：
        F = 油門 + 俯仰/滾轉/偏航修正（前右、後左、前左、後右），力矩為正規化後的姿態/偏航率指令。"""
        roll, pitch, yaw_rate, throttle = self._last_cmd
        r = roll / MAX_TILT_RAD
        p = pitch / MAX_TILT_RAD
        y = yaw_rate / MAX_YAW_RATE
        k = 0.1
        #       前後(+1前)  左右(+1右)  旋向
        mix = [(+1, +1, +1),   # F1 前右
               (-1, -1, +1),   # F2 後左
               (+1, -1, -1),   # F3 前左
               (-1, +1, -1)]   # F4 後右
        thrusts = [float(np.clip(throttle + k * (p * fb - r * lr + y * cw), 0.0, 1.0)) for fb, lr, cw in mix]
        return thrusts, [r, p, y]

    def _get_raw_state(self):
        scene, depth = self._grab_images()
        dx, dy, ratio_cur = self._vision_features(scene)
        if self._ratio_initial == 0.0 and ratio_cur > 0.0:
            self._ratio_initial = ratio_cur

        kin = self.client.getMultirotorState().kinematics_estimated
        world = _vec(self.client.simGetVehiclePose().position) + self._frame_offset
        self._world_pos = world
        self._last_altitude = -world[2]

        thrusts, moments = self._rotor_forces_and_moments()
        o, v, w, a = kin.orientation, kin.linear_velocity, kin.angular_velocity, kin.linear_acceleration
        self._yaw = airsim.to_eularian_angles(o)[2]

        raw = np.array([
            dx, dy, ratio_cur, self._ratio_initial,
            world[0], world[1], world[2],
            *self._depth_obstacles(depth),
            o.w_val, o.x_val, o.y_val, o.z_val,
            v.x_val, v.y_val, v.z_val,
            w.x_val, w.y_val, w.z_val,
            a.x_val, a.y_val, a.z_val,
            *thrusts,
            *moments,
        ], dtype=np.float32)
        return raw

    def _normalize_state(self, raw):
        s = raw.copy()
        s[2] = math.sqrt(max(raw[2], 0.0))
        s[3] = math.sqrt(max(raw[3], 0.0))
        s[2:4] = np.clip(s[2:4], 0.0, 1.0)
        s[4:7] = np.clip((raw[4:7] - self.pos_center) / self.pos_scale, -1.0, 1.0)
        s[7:10] = np.clip((raw[7:10] / OBSTACLE_MAX_RANGE) * 2.0 - 1.0, -1.0, 1.0)
        s[10:14] = raw[10:14] / (np.linalg.norm(raw[10:14]) + 1e-8)
        s[14:17] = np.clip(raw[14:17] / VEL_SCALE, -1.0, 1.0)
        s[17:20] = np.clip(raw[17:20] / ANGVEL_SCALE, -1.0, 1.0)
        s[20:23] = np.clip(raw[20:23] / ACC_SCALE, -1.0, 1.0)
        s[23:27] = np.clip(raw[23:27] / THRUST_MAX, 0.0, 1.0)
        s[27:30] = np.clip(raw[27:30] / MOMENT_SCALE, -1.0, 1.0)
        return s.astype(np.float32)

    def _get_state(self):
        raw = self._get_raw_state()
        self.last_raw_state = raw
        return self._normalize_state(raw)

    # ------------------------------------------------------------------
    # 動作
    # ------------------------------------------------------------------
    def _execute_action(self, action, substeps=1, on_substep=None):
        _, roll_f, pitch_f, thr_delta, yaw_deg = ACTION_TABLE[action]
        roll = MAX_TILT_RAD * roll_f
        pitch = MAX_TILT_RAD * pitch_f
        yaw_rate = math.radians(yaw_deg)
        throttle = min(1.0, max(0.0, HOVER_THROTTLE + thr_delta))

        # 機體座標水平速度（前/右）。表中 pitch_f<0 為前進（機頭下壓）、roll_f>0 為右傾。
        kin = self.client.getMultirotorState().kinematics_estimated
        vel, yaw = kin.linear_velocity, airsim.to_eularian_angles(kin.orientation)[2]
        v_fwd = vel.x_val * math.cos(yaw) + vel.y_val * math.sin(yaw)
        v_right = -vel.x_val * math.sin(yaw) + vel.y_val * math.cos(yaw)
        if ACTION_NAMES[action] == "hover_brake":
            # 真正的煞車：反向傾斜抵銷水平速度（只用一般傾斜指令，不作弊）
            pitch = float(np.clip(BRAKE_GAIN * v_fwd, -BRAKE_MAX_TILT, BRAKE_MAX_TILT))
            roll = float(np.clip(-BRAKE_GAIN * v_right, -BRAKE_MAX_TILT, BRAKE_MAX_TILT))
        elif pitch < 0 and v_fwd > MAX_FORWARD_SPEED:
            pitch = 0.0

        throttle = min(1.0, throttle / (math.cos(pitch) * math.cos(roll) + 1e-6))

        if self._last_altitude <= self._alt_recover:
            roll, pitch = 0.0, 0.0
            throttle = min(1.0, HOVER_THROTTLE + 0.20)

        self._last_cmd = (roll, pitch, yaw_rate, throttle)
        # AirSim 簽名為 (roll, pitch, yaw_rate, throttle, duration)；舊版這裡把 roll/pitch 順序寫反了。
        # 模擬器平時暫停：下指令 -> 只讓模擬時間走 STEP_DURATION 秒 -> 再暫停做觀測。
        # 這樣每步的物理時間固定，不受 YOLO/RPC 延遲影響（否則指令到期後 SimpleFlight 會自動鎖點煞停，無人機根本飛不動）。
        # 注意：暫停狀態下 Async 指令的 future 永遠等不到完成（join 會卡死），所以不 join；
        # 指令給長 duration，下一步的新指令會取代它，由 simContinueForTime 決定實際走多久。
        self.client.moveByRollPitchYawrateThrottleAsync(
            roll, PITCH_SIGN * pitch, YAW_SIGN * yaw_rate, throttle, COMMAND_HOLD_SECONDS
        )
        for k in range(1, substeps + 1):
            self.client.simContinueForTime(STEP_DURATION / self._clock_ratio / substeps)
            if on_substep is not None:
                on_substep(k)

    # ------------------------------------------------------------------
    # 獎勵
    # ------------------------------------------------------------------
    def _distance_to_current_target(self):
        t = self._current_target()
        return 0.0 if t is None else float(np.linalg.norm(self._world_pos - t["pos"]))

    def _compute_potential(self):
        return -self._progress_distance()

    def _off_course(self):
        rel = self._world_pos - self.course_pos["gate1"]
        s = float(np.dot(rel, self.axis))
        radial = float(np.linalg.norm(rel - s * self.axis))
        return radial > CORRIDOR_RADIUS or s < -(MAX_SPAWN_BACK + CORRIDOR_MARGIN_ALONG) \
            or s > self.course_length + CORRIDOR_MARGIN_ALONG

    def _compute_reward(self):
        pos = self._world_pos
        target = self._current_target()
        dist = self._distance_to_current_target()
        info = {"result": "running", "stage_idx": self._stage_idx, "dist": dist}

        # 1. 碰撞
        col = self.client.simGetCollisionInfo()
        if col.has_collided and col.time_stamp != self._collision_baseline:
            hit_ball = target is not None and target["type"] == "ball" and (
                target["name"] in col.object_name or dist <= BALL_CONTACT_DIST)
            if hit_ball and self._hover_mode():
                return COLLISION_PENALTY, True, {**info, "result": "touched_ball_failed"}
            if hit_ball:
                return SUCCESS_REWARD, True, {**info, "result": "success_target_reached (collision)"}
            return COLLISION_PENALTY, True, {**info, "result": f"collision_failed ({col.object_name})"}

        # 2. 空域 / 賽道邊界
        if self._last_altitude < MIN_SAFE_ALTITUDE:
            return GROUND_CRASH_PENALTY, True, {**info, "result": "crashed_to_ground"}
        if self._last_altitude > self._alt_ceiling:
            return OUT_OF_BOUNDS_PENALTY, True, {**info, "result": "out_of_bounds_too_high"}
        if self._off_course():
            return OUT_OF_BOUNDS_PENALTY, True, {**info, "result": "out_of_bounds_corridor"}

        # 3. 目標丟失 / 超時
        if self.target_lost_counter >= MAX_TARGET_LOST_STEPS:
            return TARGET_LOST_PENALTY, True, {**info, "result": "target_lost_timeout"}
        if self.current_step >= self.max_steps:
            return TIMEOUT_PENALTY, True, {**info, "result": "timeout", "truncated": True}

        # 4. 目標達成判定
        if target is not None:
            if target["type"] == "ball" and dist <= BALL_CONTACT_DIST:
                if self._hover_mode():
                    return COLLISION_PENALTY, True, {**info, "result": "touched_ball_failed"}
                return SUCCESS_REWARD, True, {**info, "result": "success_target_reached (proximity)"}
            if target["type"] == "gate":
                outcome = self._check_gate_crossing(target, pos)
                if outcome == "passed":
                    return self._advance_stage_or_finish(info)
                if outcome == "missed":
                    return MISSED_GATE_PENALTY, True, {**info, "result": "missed_gate"}

        # 4b. 接近球並懸停：在停留區內、夠慢、且機頭對準球並看得到球，連續累積滿 HOVER_REQUIRED_STEPS 步（= 10 秒模擬時間）才算成功
        hover_bonus = 0.0
        if self._hover_mode():
            speed = float(np.linalg.norm(pos - self._prev_pos)) / STEP_DURATION
            yaw_err = abs(self._yaw_error_to_target())
            facing = yaw_err <= HOVER_MAX_YAW_ERR and self.last_raw_state[2] > 0.0   # 機頭對準 + 畫面中看得到球
            info["yaw_err_deg"] = math.degrees(yaw_err)
            if HOVER_ZONE_MIN <= dist <= HOVER_ZONE_MAX and speed <= HOVER_MAX_SPEED and facing:
                self._hover_steps += 1
                hover_bonus = HOVER_STEP_BONUS
                self._steps_since_best = 0          # 正在懸停，不算空轉
                if self._hover_steps >= HOVER_REQUIRED_STEPS:
                    return SUCCESS_REWARD, True, {**info, "result": "success_hover", "hover_steps": self._hover_steps}
            else:
                self._hover_steps = 0
            info["hover_steps"] = self._hover_steps

        # 5. 原地空轉偵測
        progress = self._progress_distance()
        if progress < self._best_dist - STALL_MIN_IMPROVE:
            self._best_dist = progress
            self._steps_since_best = 0
        else:
            self._steps_since_best += 1
            if self._steps_since_best >= STALL_STEPS:
                return STALL_PENALTY, True, {**info, "result": "stalled_no_progress"}

        # 6. Potential-based shaping（真實 3D 距離進步量）+ 僅在有真實推進時才給的視覺置中獎勵
        potential = self._compute_potential()
        shaping = GAMMA_SHAPING * potential - self._prev_potential
        self._prev_potential = potential

        align_bonus = 0.0
        if self.last_raw_state[2] > 0.0 and shaping > 0.0:
            err = math.hypot(self.last_raw_state[0], self.last_raw_state[1])
            align_bonus = max(0.0, 1.0 - err) * ALIGN_BONUS_MAX

        reward = float(np.clip(shaping + align_bonus + hover_bonus + ALIVE_PENALTY,
                               -CONTINUOUS_REWARD_CLIP, CONTINUOUS_REWARD_CLIP))
        return reward, False, info

    def _check_gate_crossing(self, target, pos):
        """穿越黃框平面（平面法向 = 賽道方向）的瞬間，用線段內插求穿越點，檢查是否落在孔徑內。"""
        if self._prev_pos is None:
            return "none"
        gate = target["pos"]
        prev_s = float(np.dot(self._prev_pos - gate, self.axis))
        cur_s = float(np.dot(pos - gate, self.axis))
        if prev_s < 0.0 <= cur_s:
            frac = prev_s / (prev_s - cur_s)
            cross = self._prev_pos + frac * (pos - self._prev_pos)
            rel = cross - gate
            side = np.array([-self.axis[1], self.axis[0], 0.0])
            side = side / (np.linalg.norm(side) + 1e-9)
            lateral = abs(float(np.dot(rel, side)))     # 橫向偏移
            vertical = abs(float(rel[2]))               # 垂直偏移（賽道假設為水平）
            self.last_gate_cross = (lateral, vertical)  # 供評估腳本記錄穿框餘裕
            ok = lateral <= GATE_HALF_WIDTH and vertical <= GATE_HALF_HEIGHT
            return "passed" if ok else "missed"
        return "none"

    def _advance_stage_or_finish(self, info):
        self._stage_idx += 1
        self._ratio_initial = 0.0
        if self._stage_idx >= len(self._mission_targets):
            return SUCCESS_REWARD, True, {**info, "result": "success_target_reached (gate)"}
        self._prev_potential = self._compute_potential()
        self._reset_segment_progress()
        return STAGE_PASS_BONUS, False, {**info, "result": f"stage_{self._stage_idx}_passed",
                                         "stage_idx": self._stage_idx}

    def close(self):
        self.client.simPause(False)
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
