"""
進階 AirSim 環境：視覺追蹤與姿態控制多維度任務
=====================================================
狀態（30 維）: 視覺偏移、追蹤佔比、絕對座標、避障射線、四元數、線速度、角速度、線加速度、推力、力矩
動作（13 個）: 基於 Roll, Pitch, Yaw Rate, Throttle 的百分比控制
"""

import airsim
import numpy as np
import math
import time

# ============ 可調參數 ============
STEP_DURATION = 0.5
MIN_SAFE_ALTITUDE = 2.0
MAX_STEPS = 500

class VisionTrackingEnv:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # 狀態與動作維度
        self.state_size = 30
        self.action_size = 13

        self.current_step = 0
        self.initial_bbox_area = None
        self._collision_baseline = self.client.simGetCollisionInfo().time_stamp

        # 如果有載入 YOLO 模型，可在此初始化
        # from ultralytics import YOLO
        # self.yolo_model = YOLO('best.pt')

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()

        self.current_step = 0
        self.initial_bbox_area = None
        self._collision_baseline = self.client.simGetCollisionInfo().time_stamp

        return self._get_state()

    def step(self, action):
        self._execute_action(action)
        self.current_step += 1

        state = self._get_state()
        reward, done, info = self._compute_reward(state)

        return state, reward, done, info

    # ---------------- 視覺與狀態提取 ----------------

    def _get_vision_features(self):
        """擷取影像並使用 YOLO 提取 Bounding Box 特徵"""
        responses = self.client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
        ])
        
        if not responses or len(responses) == 0:
            return [0.0, 0.0, 0.0, 0.0]

        response = responses[0]
        # img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        # img_rgb = img1d.reshape(response.height, response.width, 3)
        
        # TODO: 替換為您的 YOLO 推論程式碼
        # results = self.yolo_model(img_rgb)
        # 假設取得 bbox = [x_center, y_center, width, height]
        
        # --- 以下為模擬資料，實作時請替換 ---
        img_w, img_h = response.width, response.height
        x_center, y_center = img_w / 2, img_h / 2  # 假位置
        width, height = 50, 50                     # 假尺寸

        # 1. Center Offset (正規化到 -1 ~ 1)
        dx_img = (x_center - img_w / 2) / (img_w / 2)
        dy_img = (y_center - img_h / 2) / (img_h / 2)
        
        # 2. Tracking Ratio
        current_area = width * height
        if self.initial_bbox_area is None:
            self.initial_bbox_area = current_area + 1e-6
        
        ratio_current = current_area / self.initial_bbox_area
        ratio_initial = 1.0 # 作為基準參考

        return [dx_img, dy_img, ratio_current, ratio_initial]

    def _get_obstacle_vector(self):
        """獲取避障射線距離"""
        # TODO: 透過 Unreal Engine 的 Distance Sensor 或 Depth 影像計算
        return [10.0, 10.0, 10.0] # [d_left, d_center, d_right] 模擬資料

    def _get_state(self):
        """組合 30 維狀態空間"""
        vision_features = self._get_vision_features() # 4維
        
        kinematics = self.client.getMultirotorState().kinematics_estimated
        pos = kinematics.position
        ori = kinematics.orientation
        lin_v = kinematics.linear_velocity
        ang_v = kinematics.angular_velocity
        lin_a = kinematics.linear_acceleration
        
        # 馬達推力指令與旋轉力矩 (若無客製化 API，暫以 0 填充或記錄上一次指令)
        rotor_states = [0.0, 0.0, 0.0, 0.0] # F1, F2, F3, F4 (4維)
        forces = [0.0, 0.0, 0.0]            # Mx, My, Mz (3維)

        state = np.array([
            *vision_features,                                  # [0:4]
            pos.x_val, pos.y_val, pos.z_val,                   # [4:7]
            *self._get_obstacle_vector(),                      # [7:10]
            ori.w, ori.x_val, ori.y_val, ori.z_val,            # [10:14]
            lin_v.x_val, lin_v.y_val, lin_v.z_val,             # [14:17]
            ang_v.x_val, ang_v.y_val, ang_v.z_val,             # [17:20]
            lin_a.x_val, lin_a.y_val, lin_a.z_val,             # [20:23]
            *rotor_states,                                     # [23:27]
            *forces                                            # [27:30]
        ], dtype=np.float32)

        return state

    # ---------------- 動作與獎勵 ----------------

    def _execute_action(self, action):
        """將 13 個離散動作映射到姿態控制指令"""
        MAX_ROLL_PITCH_RAD = math.radians(30)
        HOVER_THROTTLE = 0.59  # 需根據您的無人機重量微調
        
        roll, pitch, yaw_rate = 0.0, 0.0, 0.0
        throttle = HOVER_THROTTLE

        if action == 0:   # 強力前進
            pitch = MAX_ROLL_PITCH_RAD * -0.15
        elif action == 1: # 精準巡航
            pitch = MAX_ROLL_PITCH_RAD * -0.05
        elif action == 2: # 左平移
            roll = MAX_ROLL_PITCH_RAD * -0.12
        elif action == 3: # 右平移
            roll = MAX_ROLL_PITCH_RAD * 0.12
        elif action == 4: # 爬升
            throttle = min(1.0, HOVER_THROTTLE + 0.25)
        elif action == 5: # 降落
            throttle = max(0.0, HOVER_THROTTLE - 0.25)
        elif action == 6: # 左旋轉
            yaw_rate = math.radians(-20)
        elif action == 7: # 右旋轉
            yaw_rate = math.radians(20)
        elif action == 8: # 前進並左轉
            pitch = MAX_ROLL_PITCH_RAD * -0.08
            yaw_rate = math.radians(-15)
        elif action == 9: # 前進並右轉
            pitch = MAX_ROLL_PITCH_RAD * -0.08
            yaw_rate = math.radians(15)
        elif action == 10: # 前進並爬升
            pitch = MAX_ROLL_PITCH_RAD * -0.10
            throttle = min(1.0, HOVER_THROTTLE + 0.15)
        elif action == 11: # 前進並下降
            pitch = MAX_ROLL_PITCH_RAD * -0.10
            throttle = max(0.0, HOVER_THROTTLE - 0.15)
        elif action == 12: # 定點懸停
            pass 

        # 保護機制：避免撞地
        altitude = -self._get_state()[6] # z_val 為負代表高度
        if altitude <= MIN_SAFE_ALTITUDE and throttle < HOVER_THROTTLE:
            throttle = HOVER_THROTTLE

        if action == 12:
            self.client.hoverAsync().join()
        else:
            self.client.moveByRollPitchYawrateThrottleAsync(
                pitch, roll, yaw_rate, throttle, STEP_DURATION
            ).join()

    def _compute_reward(self, state):
        """以視覺追蹤為主的獎勵函數"""
        dx_img, dy_img = state[0], state[1]
        altitude = -state[6]

        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided and collision_info.time_stamp != self._collision_baseline:
            return -100.0, True, {"result": "collision"}

        if altitude < -1.0:
            return -100.0, True, {"result": "crashed"}

        if self.current_step >= MAX_STEPS:
            return 10.0, True, {"result": "timeout_success"} # 成功存活

        # 追蹤獎勵：目標越靠近畫面中心分數越高
        tracking_error = math.sqrt(dx_img**2 + dy_img**2)
        reward = 1.0 - tracking_error # 誤差越大扣越多
        
        # 若目標完全離開畫面 (假設誤差超過 1.0)，給予懲罰並結束回合
        if tracking_error > 1.0:
            return -50.0, True, {"result": "target_lost"}

        return float(np.clip(reward, -5.0, 5.0)), False, {"result": "running"}

    def close(self):
        self.client.armDisarm(False)
        self.client.enableApiControl(False)