"""
進階 AirSim 環境：YOLO 黃框穿越與粉球追蹤任務 (重構版 v2)
=====================================================
狀態（30 維，全部正規化至約 [-1, 1]）:
  [0:4]   視覺特徵 (dx, dy, ratio_current, target_type)
  [4:7]   NED 相對座標 (x, y, z)             — 相對 reset() 時的起飛點
  [7:10]  避障距離 (d_left, d_center, d_right)
  [10:14] 機身姿態四元數 (w, x, y, z)
  [14:17] 三軸線速度 (vx, vy, vz)
  [17:20] 三軸角速度 (p, q, r)
  [20:23] 三軸線加速度 (ax, ay, az)
  [23:27] 馬達推力指令 (F1, F2, F3, F4)
  [27:30] 機身旋轉力矩估計 (Mx, My, Mz)

任務結構：依序穿越黃框 1 -> 黃框 2 -> 觸碰粉球。
獎勵核心是「以模擬器真實世界座標算出的 Potential-based Reward Shaping」，
而非僅依賴 YOLO 像素對準，藉此杜絕「原地盯著目標」的局部最優解。
"""

import math
import time

import airsim
import cv2
import numpy as np
from ultralytics import YOLO

# ============ 任務 / 安全參數 ============
STEP_DURATION = 0.5
MIN_SAFE_ALTITUDE = 1.0
MAX_SAFE_ALTITUDE = 45.0
HOVER_THROTTLE = 0.62
MAX_TARGET_LOST_STEPS = 20
DEBUG_SAVE_YOLO_FRAME = False   # True 會每步存一張偵測結果圖，訓練時務必關閉（I/O 會拖垮 step 速度）

# YOLO / UE5 設定
YELLOW_FRAME_CLASS_ID = 0
PINK_BALL_CLASS_ID = 1

# ============ 狀態正規化尺度 ============
# 以已知的物理/任務量級做「固定尺度」正規化（非跑統計的 running mean/std）。
# 理由：起始 epsilon=1.0 時樣本分布極不穩定，用 running normalizer 在冷啟動階段
# 反而會讓尺度本身劇烈漂移；固定物理上界更適合這種先驗已知邊界的機器人任務。
POS_SCALE = 50.0        # 相對起飛點的座標範圍 (m)
OBSTACLE_MAX_RANGE = 20.0   # 避障測距感測器最大量程 (m)
VEL_SCALE = 8.0          # 線速度上界 (m/s)
ANGVEL_SCALE = 3.5       # 角速度上界 (rad/s, ~200 deg/s)
ACC_SCALE = 20.0         # 線加速度上界 (m/s^2)
THRUST_MAX = 4.5         # 單馬達最大推力上界 (N, AirSim 預設四軸量級)
MOMENT_SCALE = 2.0       # 機身力矩量級上界 (N*m, 概略值)
RATIO_TANH_DIV = 4.0     # bbox 面積比例的 tanh 壓縮除數

# ============ 任務潛能 / 獎勵參數 ============
GAMMA_SHAPING = 0.99          # 與 agent 的 gamma 對齊
SHAPING_COEF = 1.0             # 每公尺真實接近距離對應的獎勵權重
ALIVE_PENALTY = -0.02          # 每步存活成本，逼迫模型持續推進而非空轉刷分
ALIGN_BONUS_MAX = 0.1          # 視覺置中的輔助獎勵上限（僅在有真實推進時才發放）
STAGE_PASS_BONUS = 30.0        # 每通過一個黃框的一次性獎勵
SUCCESS_REWARD = 100.0
COLLISION_PENALTY = -50.0
GROUND_CRASH_PENALTY = -50.0
OUT_OF_BOUNDS_PENALTY = -50.0
TARGET_LOST_PENALTY = -30.0
TIMEOUT_PENALTY = -10.0
CONTINUOUS_REWARD_CLIP = 3.0   # 只夾住「連續型」獎勵，終局獎勵維持原值不裁切


# ============ 課程學習關卡設定 ============
# name: UE5 場景中對應的 Actor 名稱（用 client.simListSceneObjects() 自行核對後填入）。
# spawn_offset_from_first_target: 相對於第一個目標「世界座標」的 NED 出生點偏移，
#   用 simSetVehiclePose 直接瞬移到位，避免每回合都飛行過去、浪費牆鐘時間。
MISSION_STAGES = {
    # Stage 1：近距離粉球追蹤 — 只練「鎖定 + 精準碰撞」
    1: {
        "targets": [
            {"name": "PinkBall", "type": "ball", "pass_radius": 1.5},
        ],
        "spawn_offset_from_first_target": (-8.0, 0.0, 0.0),
        "max_steps": 150,
    },
    # Stage 2：單一黃框穿越 — 只練「直線穿框」
    2: {
        "targets": [
            {"name": "YellowFrame1", "type": "gate", "pass_radius": 2.0},
        ],
        "spawn_offset_from_first_target": (-15.0, 0.0, 0.0),
        "max_steps": 300,
    },
    # Stage 3：雙框 + 粉球，完整任務
    3: {
        "targets": [
            {"name": "YellowFrame1", "type": "gate", "pass_radius": 2.0},
            {"name": "YellowFrame2", "type": "gate", "pass_radius": 2.0},
            {"name": "PinkBall", "type": "ball", "pass_radius": 1.5},
        ],
        "spawn_offset_from_first_target": (-15.0, 0.0, 0.0),
        "max_steps": 500,
    },
}
DEFAULT_TARGET_WORLD_POS = airsim.Vector3r(30.0, 0.0, -30.0)  # 找不到 UE5 物件時的保底座標，請依場景調整


class VisionTrackingEnv:
    def __init__(self, curriculum_stage=3):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.state_size = 30
        self.action_size = 13

        self.curriculum_stage = curriculum_stage
        self.current_step = 0
        self.target_lost_counter = 0
        self.initial_bbox_area = None
        self._collision_baseline = self.client.simGetCollisionInfo().time_stamp

        self._start_pos = None          # reset() 時的世界座標，用於相對座標正規化
        self._mission_targets = []      # 本回合任務目標序列
        self._stage_idx = 0             # 目前追蹤到序列中的第幾個目標
        self._prev_potential = 0.0
        self.last_raw_state = None      # 未正規化的原始狀態，供人類判讀 / 除錯用

        print("正在載入 YOLO 模型...")
        self.yolo_model = YOLO('best.pt')
        print("YOLO 模型載入完成！")

    def set_curriculum_stage(self, stage: int):
        assert stage in MISSION_STAGES, f"未知課程階段: {stage}"
        self.curriculum_stage = stage

    # ---------------- Reset ----------------

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()

        stage_cfg = MISSION_STAGES[self.curriculum_stage]
        self._mission_targets = stage_cfg["targets"]
        self._stage_idx = 0
        self.max_steps = stage_cfg["max_steps"]

        first_target_pos = self._get_target_world_position(self._mission_targets[0]["name"])
        offset = stage_cfg["spawn_offset_from_first_target"]
        spawn_pos = airsim.Vector3r(
            first_target_pos.x_val + offset[0],
            first_target_pos.y_val + offset[1],
            first_target_pos.z_val + offset[2],
        )
        yaw = math.atan2(
            first_target_pos.y_val - spawn_pos.y_val,
            first_target_pos.x_val - spawn_pos.x_val,
        )
        spawn_pose = airsim.Pose(spawn_pos, airsim.to_quaternion(0.0, 0.0, yaw))
        self.client.simSetVehiclePose(spawn_pose, ignore_collision=True)
        time.sleep(0.2)  # 讓物理引擎穩定一瞬間，避免瞬移後速度殘留

        self._start_pos = self.client.getMultirotorState().kinematics_estimated.position

        self.current_step = 0
        self.target_lost_counter = 0
        self.initial_bbox_area = None
        self._collision_baseline = self.client.simGetCollisionInfo().time_stamp

        state = self._get_state()
        self._prev_potential = self._compute_potential()
        return state

    # ---------------- Step ----------------

    def step(self, action):
        self._execute_action(action)
        self.current_step += 1

        state = self._get_state()
        reward, done, info = self._compute_reward(state)

        return state, reward, done, info

    # ---------------- 任務目標 / 潛能函數（使用模擬器特權真實座標，僅用於算獎勵，不進 state） ----------------

    def _get_target_world_position(self, object_name):
        try:
            pose = self.client.simGetObjectPose(object_name)
            if pose is not None and not math.isnan(pose.position.x_val):
                return pose.position
        except Exception:
            pass
        return DEFAULT_TARGET_WORLD_POS

    def _current_stage_target(self):
        if self._stage_idx >= len(self._mission_targets):
            return None
        return self._mission_targets[self._stage_idx]

    def _distance_to_current_target(self):
        target = self._current_stage_target()
        if target is None:
            return 0.0
        target_pos = self._get_target_world_position(target["name"])
        drone_pos = self.client.getMultirotorState().kinematics_estimated.position
        dx = target_pos.x_val - drone_pos.x_val
        dy = target_pos.y_val - drone_pos.y_val
        dz = target_pos.z_val - drone_pos.z_val
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _compute_potential(self):
        """Φ(s) = -distance_to_current_target。距離越近，潛能越高。"""
        return -self._distance_to_current_target()

    # ---------------- 視覺與狀態提取 ----------------

    def _get_vision_features(self):
        responses = self.client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
        ])

        img_w, img_h = 640, 480
        result = None
        if responses and len(responses) > 0:
            response = responses[0]
            img_w, img_h = response.width, response.height
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(img_h, img_w, 3)

            results = self.yolo_model(img_rgb, verbose=False, conf=0.25)
            result = results[0]

            if DEBUG_SAVE_YOLO_FRAME:
                cv2.imwrite("debug_yolo.jpg", result.plot())
        else:
            print("⚠️ 警告：沒有從 AirSim 獲取到任何影像！")

        target_box = None
        target_type = -1.0

        if result is not None:
            pink_balls = []
            yellow_frames = []
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id == PINK_BALL_CLASS_ID:
                    pink_balls.append(box)
                elif cls_id == YELLOW_FRAME_CLASS_ID:
                    yellow_frames.append(box)

            # 嚴格只鎖定「目前任務階段需要的類型」，不做跨類型 fallback。
            # 場景中黃框與粉球會同時存在、同時入鏡，若允許 fallback 到錯誤類型，
            # 視覺特徵（dx, dy, target_type）會跟獎勵邏輯實際追蹤的真實目標對不上
            # （例如 Stage 1 只該看粉球，卻在粉球未偵測到的那幾幀改回報黃框的座標），
            # 讓 align_bonus 有機會獎勵到「對準了錯的東西」。抓不到就視為丟失目標即可。
            wanted_type = self._current_stage_target()
            wanted_type = wanted_type["type"] if wanted_type is not None else None

            def pick_largest(boxes):
                boxes.sort(key=lambda b: b.xywh[0][2] * b.xywh[0][3], reverse=True)
                return boxes[0]

            if wanted_type == "ball" and len(pink_balls) > 0:
                target_box, target_type = pick_largest(pink_balls), 1.0
            elif wanted_type == "gate" and len(yellow_frames) > 0:
                target_box, target_type = pick_largest(yellow_frames), 0.0

        if target_box is None:
            self.target_lost_counter += 1
            dx_img, dy_img = 0.0, 0.0
            ratio_current = 0.0
        else:
            self.target_lost_counter = 0
            x_center, y_center, width, height = target_box.xywh[0].cpu().numpy()

            dx_img = (x_center - img_w / 2) / (img_w / 2)
            dy_img = (y_center - img_h / 2) / (img_h / 2)

            current_area = width * height
            if self.initial_bbox_area is None:
                self.initial_bbox_area = current_area + 1e-6
            ratio_current = current_area / self.initial_bbox_area

        return [dx_img, dy_img, ratio_current, target_type]

    def _get_obstacle_vector(self):
        """回傳 [左, 中, 右] 避障距離 (公尺)。若場景未掛載對應的 distance sensor，
        則退回量程上界（視為「無障礙」），並非真正的避障，請在 settings.json
        設定名為 distance_left / distance_center / distance_right 的 DistanceSensor。"""
        names = ["distance_left", "distance_center", "distance_right"]
        distances = []
        for name in names:
            try:
                data = self.client.getDistanceSensorData(name)
                d = data.distance if data.distance > 0 else OBSTACLE_MAX_RANGE
            except Exception:
                d = OBSTACLE_MAX_RANGE
            distances.append(min(d, OBSTACLE_MAX_RANGE))
        return distances

    def _get_rotor_forces_and_moments(self):
        """讀取真實馬達推力；力矩為概略估計（依四軸 X 型配置的槓桿關係換算），
        僅作為特徵輸入，非精確剛體動力學計算。"""
        try:
            rotor_states = self.client.getRotorStates().rotors
            thrusts = [r["thrust"] for r in rotor_states] if isinstance(rotor_states[0], dict) \
                else [r.thrust for r in rotor_states]
            torques = [r["torque_scalar"] for r in rotor_states] if isinstance(rotor_states[0], dict) \
                else [r.torque_scalar for r in rotor_states]
        except Exception:
            thrusts = [0.0, 0.0, 0.0, 0.0]
            torques = [0.0, 0.0, 0.0]

        if len(thrusts) < 4:
            thrusts = (thrusts + [0.0, 0.0, 0.0, 0.0])[:4]

        # AirSim 預設四軸命名順序約為 [前右, 後左, 前左, 後右]（CW/CCW 交錯）。
        # Mx (roll力矩) ~ 左右推力差, My (pitch力矩) ~ 前後推力差, Mz (yaw力矩) ~ 反扭矩總和。
        f1, f2, f3, f4 = thrusts[0], thrusts[1], thrusts[2], thrusts[3]
        mx = (f3 + f2) - (f1 + f4)
        my = (f1 + f3) - (f2 + f4)
        mz = sum(torques) if torques else 0.0

        return thrusts, [mx, my, mz]

    def _get_raw_state(self):
        vision_features = self._get_vision_features()
        kinematics = self.client.getMultirotorState().kinematics_estimated
        pos = kinematics.position
        ori = kinematics.orientation
        lin_v = kinematics.linear_velocity
        ang_v = kinematics.angular_velocity
        lin_a = kinematics.linear_acceleration

        thrusts, moments = self._get_rotor_forces_and_moments()
        obstacle = self._get_obstacle_vector()

        rel_x = pos.x_val - (self._start_pos.x_val if self._start_pos else 0.0)
        rel_y = pos.y_val - (self._start_pos.y_val if self._start_pos else 0.0)
        rel_z = pos.z_val - (self._start_pos.z_val if self._start_pos else 0.0)

        raw = np.array([
            *vision_features,
            rel_x, rel_y, rel_z,
            *obstacle,
            ori.w_val, ori.x_val, ori.y_val, ori.z_val,
            lin_v.x_val, lin_v.y_val, lin_v.z_val,
            ang_v.x_val, ang_v.y_val, ang_v.z_val,
            lin_a.x_val, lin_a.y_val, lin_a.z_val,
            *thrusts,
            *moments,
        ], dtype=np.float32)
        return raw, pos.z_val  # 額外回傳絕對高度 (NED z，負值=高)

    def _normalize_state(self, raw, abs_z):
        s = raw.copy()

        # [0] dx, [1] dy 已在 [-1,1]
        # [2] ratio_current: 用 tanh 壓縮，中心點=1.0(面積無變化)
        s[2] = math.tanh((raw[2] - 1.0) / RATIO_TANH_DIV)
        # [3] target_type 本身已是 {-1, 0, 1}

        # [4:7] 相對座標
        s[4:7] = np.clip(raw[4:7] / POS_SCALE, -1.0, 1.0)

        # [7:10] 避障距離 -> 映射到 [-1, 1]，越近越負
        s[7:10] = np.clip((raw[7:10] / OBSTACLE_MAX_RANGE) * 2.0 - 1.0, -1.0, 1.0)

        # [10:14] 四元數，重新單位化以避免數值漂移
        quat = raw[10:14]
        norm = np.linalg.norm(quat) + 1e-8
        s[10:14] = quat / norm

        # [14:17] 線速度
        s[14:17] = np.clip(raw[14:17] / VEL_SCALE, -1.0, 1.0)

        # [17:20] 角速度
        s[17:20] = np.clip(raw[17:20] / ANGVEL_SCALE, -1.0, 1.0)

        # [20:23] 線加速度
        s[20:23] = np.clip(raw[20:23] / ACC_SCALE, -1.0, 1.0)

        # [23:27] 推力 (單向，[0,1] 即可，不必到 [-1,1])
        s[23:27] = np.clip(raw[23:27] / THRUST_MAX, 0.0, 1.0)

        # [27:30] 力矩
        s[27:30] = np.clip(raw[27:30] / MOMENT_SCALE, -1.0, 1.0)

        return s.astype(np.float32)

    def _get_state(self):
        raw, abs_z = self._get_raw_state()
        self.last_raw_state = raw
        self._last_abs_altitude = -abs_z  # NED z 為負值時代表在地面以上
        return self._normalize_state(raw, abs_z)

    # ---------------- 動作與控制 ----------------

    def _execute_action(self, action):
        MAX_ROLL_PITCH_RAD = math.radians(30)
        roll, pitch, yaw_rate = 0.0, 0.0, 0.0
        throttle = HOVER_THROTTLE

        if action == 0: pitch = MAX_ROLL_PITCH_RAD * -0.15
        elif action == 1: pitch = MAX_ROLL_PITCH_RAD * -0.05
        elif action == 2: roll = MAX_ROLL_PITCH_RAD * -0.12
        elif action == 3: roll = MAX_ROLL_PITCH_RAD * 0.12
        elif action == 4: throttle = min(1.0, HOVER_THROTTLE + 0.25)
        elif action == 5: throttle = max(0.0, HOVER_THROTTLE - 0.25)
        elif action == 6: yaw_rate = math.radians(-20)
        elif action == 7: yaw_rate = math.radians(20)
        elif action == 8:
            pitch = MAX_ROLL_PITCH_RAD * -0.08
            yaw_rate = math.radians(-15)
        elif action == 9:
            pitch = MAX_ROLL_PITCH_RAD * -0.08
            yaw_rate = math.radians(15)
        elif action == 10:
            pitch = MAX_ROLL_PITCH_RAD * -0.10
            throttle = min(1.0, HOVER_THROTTLE + 0.15)
        elif action == 11:
            pitch = MAX_ROLL_PITCH_RAD * -0.10
            throttle = max(0.0, HOVER_THROTTLE - 0.15)
        elif action == 12: pass

        tilt_factor = 1.0 / (math.cos(pitch) * math.cos(roll) + 1e-6)
        throttle = min(1.0, throttle * tilt_factor)

        # 主動防墜保護：直接讀取當前運動學高度，不再重覆呼叫 _get_state()
        # （舊版在這裡呼叫 _get_state() 等於每個 step 多跑一次 YOLO 推論 + 多拉一張影像，
        #   白白浪費一半的算力，是本次重構修掉的效能 bug）。
        altitude = -self.client.getMultirotorState().kinematics_estimated.position.z_val
        if altitude <= 15.0:
            pitch = 0.0
            roll = 0.0
            throttle = min(1.0, HOVER_THROTTLE + 0.20)

        if action == 12:
            self.client.hoverAsync().join()
        else:
            self.client.moveByRollPitchYawrateThrottleAsync(
                pitch, roll, yaw_rate, throttle, STEP_DURATION
            ).join()

    # ---------------- 獎勵計算 ----------------

    def _compute_reward(self, state):
        dx_img, dy_img, target_type = state[0], state[1], state[3]
        altitude = self._last_abs_altitude

        # 1. 碰撞判定：撞到「目前任務階段」要求的目標物才算通過/成功
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided and collision_info.time_stamp != self._collision_baseline:
            current_target = self._current_stage_target()
            hit_name = collision_info.object_name
            if current_target is not None and current_target["name"] in hit_name:
                return self._advance_stage_or_finish(is_collision=True)
            else:
                return COLLISION_PENALTY, True, {"result": f"collision_failed ({hit_name})"}

        # 2. 空域安全判定
        if altitude < MIN_SAFE_ALTITUDE:
            return GROUND_CRASH_PENALTY, True, {"result": "crashed_to_ground"}
        if altitude > MAX_SAFE_ALTITUDE:
            return OUT_OF_BOUNDS_PENALTY, True, {"result": "out_of_bounds_too_high"}

        # 3. 目標丟失判定
        if self.target_lost_counter >= MAX_TARGET_LOST_STEPS:
            return TARGET_LOST_PENALTY, True, {"result": "target_lost_timeout"}

        # 4. 超時判定
        if self.current_step >= self.max_steps:
            return TIMEOUT_PENALTY, True, {"result": "timeout"}

        # 5. 以真實世界座標算距離是否已進入通過半徑（非碰撞體的框也能算通過）
        current_target = self._current_stage_target()
        if current_target is not None:
            dist = self._distance_to_current_target()
            if dist <= current_target["pass_radius"]:
                return self._advance_stage_or_finish(is_collision=False)

        # 6. Potential-based Reward Shaping：用真實 3D 距離的進步量做核心獎勵，
        #    這是杜絕「原地盯著看」的關鍵 —— 只要沒有真的靠近目標，這項就不會有正值累積。
        potential = self._compute_potential()
        shaping_reward = SHAPING_COEF * (GAMMA_SHAPING * potential - self._prev_potential)
        self._prev_potential = potential

        # 7. 視覺置中輔助獎勵：只在「這步確實有真實推進」時才發放，
        #    避免模型學會單靠像素置中拿分而不飛。
        align_bonus = 0.0
        if target_type != -1.0 and shaping_reward > 0.0:
            tracking_error = math.sqrt(dx_img ** 2 + dy_img ** 2)
            align_bonus = max(0.0, (1.0 - tracking_error)) * ALIGN_BONUS_MAX

        reward = shaping_reward + align_bonus + ALIVE_PENALTY
        reward = float(np.clip(reward, -CONTINUOUS_REWARD_CLIP, CONTINUOUS_REWARD_CLIP))

        return reward, False, {"result": "running"}

    def _advance_stage_or_finish(self, is_collision):
        self._stage_idx += 1
        if self._stage_idx >= len(self._mission_targets):
            tag = "collision" if is_collision else "proximity"
            return SUCCESS_REWARD, True, {"result": f"success_target_reached ({tag})"}
        else:
            # 進入下一階段目標，重置潛能基準線（新的子目標，新的 Φ 起點）
            self._prev_potential = self._compute_potential()
            self.initial_bbox_area = None
            return STAGE_PASS_BONUS, False, {"result": f"stage_{self._stage_idx}_passed"}

    def close(self):
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
