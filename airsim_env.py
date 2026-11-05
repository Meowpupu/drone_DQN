"""
AirSim 四旋翼視覺追蹤 + 避障導航環境（Gym 風格介面）
======================================================
動作空間：13 個離散低階飛控指令（Pitch/Roll/Throttle/YawRate 百分比）
狀態空間：31 維向量，涵蓋視覺追蹤、位置、障礙物、姿態、動力學、致動器輸出

【重要】以下數值為可調參數，需依實際載具動態特性校正：
  - MAX_TILT_DEG    : 100% Pitch/Roll 對應的最大傾角
  - HOVER_THROTTLE  : 懸停基準油門（依載具重量/推力比而異）
  - THROTTLE_RANGE  : 油門百分比對應的實際油門變化範圍
  - INERTIA_DIAG    : 機身慣性矩（用於力矩近似估算，非精確物理量）
"""

import airsim
import numpy as np
import math
import time


# ============ 可調參數區 ============
MAX_TILT_DEG = 20.0          # 100% Pitch/Roll 對應的最大傾角（度）
HOVER_THROTTLE = 0.593       # 懸停基準油門（AirSim SimpleFlight 預設約略值，請依實測校正）
THROTTLE_RANGE = 0.35        # +-100% throttle 指令對應的油門變化幅度
MAX_ROTOR_THRUST = 4.5       # 單馬達最大推力（牛頓），用於正規化 F1-F4
INERTIA_DIAG = np.array([0.0219, 0.0219, 0.0366])  # 機身慣性矩近似值 (Ixx, Iyy, Izz)
OBSTACLE_MAX_RANGE = 40.0    # 障礙物感測器最大量測距離（公尺），用於正規化


# ============ 13 個離散動作定義表 ============
ACTION_TABLE = [
    dict(name="Fast Forward",        pitch=-15, roll=0,   throttle=0,   yaw=0),    # 0
    dict(name="Slow Forward",        pitch=-5,  roll=0,   throttle=0,   yaw=0),    # 1
    dict(name="Roll Left",           pitch=0,   roll=-12, throttle=0,   yaw=0),    # 2
    dict(name="Roll Right",          pitch=0,   roll=12,  throttle=0,   yaw=0),    # 3
    dict(name="Ascend",              pitch=0,   roll=0,   throttle=25,  yaw=0),    # 4
    dict(name="Descend",             pitch=0,   roll=0,   throttle=-25, yaw=0),    # 5
    dict(name="Yaw Left",            pitch=0,   roll=0,   throttle=0,   yaw=-20),  # 6
    dict(name="Yaw Right",           pitch=0,   roll=0,   throttle=0,   yaw=20),   # 7
    dict(name="Forward + Yaw L",     pitch=-8,  roll=0,   throttle=0,   yaw=-15),  # 8
    dict(name="Forward + Yaw R",     pitch=-8,  roll=0,   throttle=0,   yaw=15),   # 9
    dict(name="Forward + Ascend",    pitch=-10, roll=0,   throttle=15,  yaw=0),    # 10
    dict(name="Forward + Descend",   pitch=-10, roll=0,   throttle=-15, yaw=0),    # 11
    dict(name="Hover / Brake",       pitch=0,   roll=0,   throttle=0,   yaw=0),    # 12
]

STATE_LAYOUT = {
    "vision":       5,   # dx_img, dy_img, ratio_current, ratio_initial, target_visible
    "position":     3,   # x, y, z (NED)
    "obstacle":     3,   # d_left, d_center, d_right
    "quaternion":   4,   # w, x, y, z
    "lin_velocity": 3,   # vx, vy, vz
    "ang_velocity": 3,   # p, q, r
    "lin_accel":    3,   # ax, ay, az
    "actuation":    4,   # F1, F2, F3, F4
    "torque":       3,   # Mx, My, Mz（近似值）
}
STATE_SIZE = sum(STATE_LAYOUT.values())  # 31
ACTION_SIZE = len(ACTION_TABLE)          # 13


class AirSimDroneEnv:
    def __init__(
        self,
        max_steps=300,
        step_duration=0.3,
        goal_ratio_threshold=0.6,
        min_obstacle_dist=1.5,
        distance_sensor_names=("distance_left", "distance_center", "distance_right"),
        camera_name="0",
        img_width=256,
        img_height=144,
    ):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.max_steps = max_steps
        self.step_duration = step_duration
        self.goal_ratio_threshold = goal_ratio_threshold
        self.min_obstacle_dist = min_obstacle_dist
        self.distance_sensor_names = distance_sensor_names
        self.camera_name = camera_name
        self.img_width = img_width
        self.img_height = img_height

        self.state_size = STATE_SIZE
        self.action_size = ACTION_SIZE

        self.current_step = 0
        self.initial_ratio = None
        self.prev_ang_velocity = np.zeros(3)
        self.prev_time = None

    # ---------------- 核心 Gym 介面 ----------------

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.client.takeoffAsync().join()
        self.client.moveToZAsync(-5, 2).join()

        self.current_step = 0
        self.initial_ratio = None
        self.prev_ang_velocity = np.zeros(3)
        self.prev_time = time.time()

        return self._get_state(detection=None)

    def step(self, action, detection=None):
        """
        action:    0~12 的整數，對應 ACTION_TABLE
        detection: 外部視覺模組（如 YOLO）傳入的偵測結果，格式:
                   {"bbox": (x1, y1, x2, y2), "img_w": int, "img_h": int}
                   若本幀未偵測到目標，傳入 None
        """
        self._apply_action(action)
        self.current_step += 1

        state = self._get_state(detection)
        reward, done, info = self._compute_reward(detection)

        return state, reward, done, info

    # ---------------- 動作執行 ----------------

    def _apply_action(self, action_idx):
        a = ACTION_TABLE[action_idx]

        pitch_rad = math.radians(a["pitch"] / 100.0 * MAX_TILT_DEG)
        roll_rad = math.radians(a["roll"] / 100.0 * MAX_TILT_DEG)
        yaw_rate_rad = math.radians(a["yaw"])
        throttle = float(np.clip(
            HOVER_THROTTLE + (a["throttle"] / 100.0) * THROTTLE_RANGE, 0.0, 1.0
        ))

        self.client.moveByRollPitchYawrateThrottleAsync(
            roll_rad, pitch_rad, yaw_rate_rad, throttle, self.step_duration
        ).join()

    # ---------------- 狀態向量組裝 ----------------

    def _get_state(self, detection):
        vision = self._get_vision_features(detection)
        position = self._get_position()
        obstacle = self._get_obstacle_distances()
        quat = self._get_quaternion()
        lin_vel = self._get_linear_velocity()
        ang_vel = self._get_angular_velocity()
        lin_acc = self._get_linear_acceleration()
        actuation = self._get_rotor_thrusts()
        torque = self._estimate_torque(ang_vel)

        state = np.concatenate([
            vision, position, obstacle, quat,
            lin_vel, ang_vel, lin_acc, actuation, torque
        ]).astype(np.float32)

        return state

    def _get_vision_features(self, detection):
        if detection is None:
            ratio_current = 0.0
            dx_img, dy_img = 0.0, 0.0
            visible = 0.0
        else:
            x1, y1, x2, y2 = detection["bbox"]
            img_w = detection.get("img_w", self.img_width)
            img_h = detection.get("img_h", self.img_height)

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            dx_img = (cx - img_w / 2.0) / (img_w / 2.0)
            dy_img = (cy - img_h / 2.0) / (img_h / 2.0)

            box_area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
            ratio_current = box_area / float(img_w * img_h)
            visible = 1.0

            if self.initial_ratio is None:
                self.initial_ratio = ratio_current

        ratio_initial = self.initial_ratio if self.initial_ratio is not None else 0.0

        return np.array([dx_img, dy_img, ratio_current, ratio_initial, visible], dtype=np.float32)

    def _get_position(self):
        pos = self.client.getMultirotorState().kinematics_estimated.position
        return np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32) / 100.0

    def _get_obstacle_distances(self):
        dists = []
        for name in self.distance_sensor_names:
            try:
                d = self.client.getDistanceSensorData(name).distance
            except Exception:
                d = OBSTACLE_MAX_RANGE
            dists.append(min(d, OBSTACLE_MAX_RANGE) / OBSTACLE_MAX_RANGE)
        return np.array(dists, dtype=np.float32)

    def _get_quaternion(self):
        q = self.client.getMultirotorState().kinematics_estimated.orientation
        return np.array([q.w_val, q.x_val, q.y_val, q.z_val], dtype=np.float32)

    def _get_linear_velocity(self):
        v = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        return np.array([v.x_val, v.y_val, v.z_val], dtype=np.float32) / 10.0

    def _get_angular_velocity(self):
        av = self.client.getMultirotorState().kinematics_estimated.angular_velocity
        return np.array([av.x_val, av.y_val, av.z_val], dtype=np.float32)

    def _get_linear_acceleration(self):
        acc = self.client.getMultirotorState().kinematics_estimated.linear_acceleration
        return np.array([acc.x_val, acc.y_val, acc.z_val], dtype=np.float32) / 20.0

    def _get_rotor_thrusts(self):
        try:
            rotor_states = self.client.getRotorStates().rotors
            thrusts = np.array([r["thrust"] for r in rotor_states], dtype=np.float32)
        except Exception:
            thrusts = np.zeros(4, dtype=np.float32)
        return thrusts / MAX_ROTOR_THRUST

    def _estimate_torque(self, ang_vel_raw):
        """力矩近似估算: M ~= I * domega/dt（非直接量測值，僅供 RL 特徵參考，非精確物理量）"""
        now = time.time()
        dt = max(now - (self.prev_time or now), 1e-3)
        d_omega = ang_vel_raw - self.prev_ang_velocity
        torque = INERTIA_DIAG * (d_omega / dt)

        self.prev_ang_velocity = ang_vel_raw
        self.prev_time = now

        return np.clip(torque, -5.0, 5.0).astype(np.float32)

    # ---------------- 獎勵計算 ----------------

    def _compute_reward(self, detection):
        if self.client.simGetCollisionInfo().has_collided:
            return -100.0, True, {"result": "collision"}

        obstacle_raw = self._get_obstacle_distances() * OBSTACLE_MAX_RANGE
        if np.min(obstacle_raw) < self.min_obstacle_dist:
            return -50.0, True, {"result": "obstacle_too_close"}

        if self.current_step >= self.max_steps:
            return -10.0, True, {"result": "timeout"}

        if detection is None:
            return -1.0, False, {"result": "target_lost"}

        x1, y1, x2, y2 = detection["bbox"]
        img_w = detection.get("img_w", self.img_width)
        img_h = detection.get("img_h", self.img_height)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dx_norm = abs((cx - img_w / 2.0) / (img_w / 2.0))
        dy_norm = abs((cy - img_h / 2.0) / (img_h / 2.0))
        ratio_current = ((x2 - x1) * (y2 - y1)) / float(img_w * img_h)

        if ratio_current > self.goal_ratio_threshold:
            return 100.0, True, {"result": "target_reached"}

        centering_reward = 1.0 - (dx_norm + dy_norm) / 2.0
        reward = centering_reward - 0.01

        return reward, False, {"result": "tracking"}

    def close(self):
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
