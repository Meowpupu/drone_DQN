"""
簡化版 AirSim 環境：定高懸停任務
====================================
目標：無人機從地面起飛，爬升到指定高度並穩定懸停。
只控制垂直速度（vz），不涉及水平移動、視覺、障礙物等，用來驗證 DQN 訓練管線。

狀態（3 維）: [目前高度(正規化), 垂直速度(正規化), 高度誤差(正規化)]
動作（3 個）: 0=上升 1=下降 2=懸停
"""

import airsim
import numpy as np


TARGET_ALTITUDE = 30.0     # 目標高度（公尺）
MAX_ALTITUDE_NORM = 50.0   # 正規化用的高度上限
VZ_SPEED = 2.0             # 上升/下降時的垂直速度指令 (m/s)
STEP_DURATION = 0.5        # 每個動作執行時間（秒）
SUCCESS_THRESHOLD = 1.5    # 高度誤差小於此值（公尺）視為「在目標範圍內」
SUCCESS_STEPS_REQUIRED = 10  # 需連續穩定在目標範圍內幾步才算任務成功
MAX_STEPS = 150            # 單回合最大步數


class AltitudeHoverEnv:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.state_size = 3
        self.action_size = 3

        self.current_step = 0
        self.success_counter = 0

    # ---------------- 核心 Gym 介面 ----------------

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()  # 起飛到預設的低高度（約 1~2 公尺）

        self.current_step = 0
        self.success_counter = 0

        return self._get_state()

    def step(self, action):
        vz = self._action_to_vz(action)

        # 只控制垂直速度，水平速度固定為 0，盡量減少水平飄移
        self.client.moveByVelocityAsync(
            0, 0, vz, STEP_DURATION,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(False, 0)
        ).join()

        self.current_step += 1

        state = self._get_state()
        reward, done, info = self._compute_reward()

        return state, reward, done, info

    # ---------------- 內部工具 ----------------

    def _action_to_vz(self, action):
        if action == 0:      # 上升
            return -VZ_SPEED  # NED 座標系，z 負值代表往上
        elif action == 1:    # 下降
            return VZ_SPEED
        else:                # 懸停
            return 0.0

    def _get_altitude(self):
        pos = self.client.getMultirotorState().kinematics_estimated.position
        return -pos.z_val  # 轉成「正值代表高度」比較直覺

    def _get_vertical_velocity(self):
        vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        return -vel.z_val  # 同樣轉成正值代表向上

    def _get_state(self):
        altitude = self._get_altitude()
        vz = self._get_vertical_velocity()
        error = TARGET_ALTITUDE - altitude

        state = np.array([
            altitude / MAX_ALTITUDE_NORM,
            vz / VZ_SPEED,
            error / MAX_ALTITUDE_NORM,
        ], dtype=np.float32)

        return state

    def _compute_reward(self):
        altitude = self._get_altitude()
        error = TARGET_ALTITUDE - altitude

        # 碰撞（例如撞到地面以外的東西）
        if self.client.simGetCollisionInfo().has_collided:
            return -100.0, True, {"result": "collision"}

        # 掉到地面以下（不太可能，但保險）
        if altitude < -1.0:
            return -100.0, True, {"result": "crashed"}

        # 超過最大步數
        if self.current_step >= MAX_STEPS:
            return -10.0, True, {"result": "timeout"}

        # 每步的密集獎勵：離目標越近，分數越高（負的絕對誤差）
        reward = -abs(error) * 0.05

        if abs(error) < SUCCESS_THRESHOLD:
            reward += 1.0
            self.success_counter += 1
        else:
            self.success_counter = 0

        # 連續穩定在目標範圍內夠久 -> 視為任務完成
        if self.success_counter >= SUCCESS_STEPS_REQUIRED:
            return 100.0, True, {"result": "success"}

        return reward, False, {"result": "running"}

    def close(self):
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
