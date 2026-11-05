"""
三維定點導航模型評估腳本（v4，加入延伸驗證階段）
======================================================
載入訓練好的 DQN 模型，用「純粹依賴學到的策略」（不隨機探索）飛行，
並在終端機即時印出每一步的座標、速度、動作、以及「分軸誤差」等資訊。

【延伸驗證】一旦達成 success（訓練時的 3 秒懸停判定），會繼續讓模型
再飛行額外的時間（不受訓練時的終止條件限制），實際觀察它是否能持續
穩定懸停在目標點，而不只是通過 3 秒門檻。

執行方式:
    python evaluate_position.py
    python evaluate_position.py checkpoints_position/dqn_position_ep500.keras
"""

import sys
import glob
import os
import time
import numpy as np

from tensorflow.keras import models
from airsim_env_position import PositionNavEnv, TARGET_X, TARGET_Y, TARGET_ALTITUDE, STEP_DURATION


ACTION_NAMES = {
    0: "北 ↑", 1: "南 ↓", 2: "東 →", 3: "西 ←",
    4: "東北 ↗", 5: "東南 ↘", 6: "西北 ↖", 7: "西南 ↙",
    8: "上升 ⤴", 9: "下降 ⤵", 10: "懸停 ─", 11: "直飛 ➹",
}

EXTENDED_VERIFY_SECONDS = 20  # 【新增】success 之後，額外多觀察幾秒鐘


def find_latest_checkpoint(model_dir="checkpoints_position"):
    files = glob.glob(os.path.join(model_dir, "*.keras"))
    if not files:
        return None
    files.sort(key=lambda f: int("".join(filter(str.isdigit, os.path.basename(f)))))
    return files[-1]


def run_extended_verification(env, model, step_start, extra_seconds):
    """
    success 之後，繞過環境內建的終止判斷，直接手動驅動無人機額外飛行一段時間，
    純粹用來眼見為憑：確認它是否能持續穩定懸停，而不只是剛好卡過 3 秒門檻。
    """
    print(f"\n   >> 進入延伸驗證階段，額外觀察 {extra_seconds} 秒，確認是否能持續穩定懸停...")

    extra_steps = int(extra_seconds / STEP_DURATION)
    step = step_start

    for i in range(extra_steps):
        state = env._get_state()
        q_values = model.predict(state.reshape(1, -1), verbose=0)[0]
        action = int(q_values.argmax())

        vx, vy, vz = env._action_to_velocity(action)
        if action == 10:
            env.client.hoverAsync().join()
        else:
            env.client.moveByVelocityAsync(
                vx, vy, vz, STEP_DURATION,
                drivetrain=__import__("airsim").DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=__import__("airsim").YawMode(False, 0)
            ).join()

        x, y, z = env._get_position_raw()
        altitude = -z
        distance, (dx, dy, dz) = env._get_distance_3d()
        cur_vx, cur_vy, cur_vz = env._get_velocity_raw()
        speed = np.sqrt(cur_vx**2 + cur_vy**2 + cur_vz**2)

        print(f"   延伸 Step {step:3d} | x:{x:6.2f} y:{y:6.2f} 高度:{altitude:6.2f} m | "
              f"3D誤差: {distance:5.1f} m | 速度: {speed:4.2f} m/s | 動作: {ACTION_NAMES[action]:6s}")

        step += 1
        time.sleep(0.05)

    print(f"   >> 延伸驗證結束。")


def main():
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = find_latest_checkpoint()
        if model_path is None:
            print("找不到任何 checkpoint，請確認 checkpoints_position 資料夾裡有 .keras 檔案，"
                  "或直接指定路徑：python evaluate_position.py <模型路徑>")
            return

    print(f"載入模型: {model_path}\n")
    model = models.load_model(model_path)

    env = PositionNavEnv()

    num_episodes = 3
    for ep in range(1, num_episodes + 1):
        print(f"\n{'='*80}")
        print(f"第 {ep}/{num_episodes} 回合開始"
              f"（目標: x={TARGET_X}m, y={TARGET_Y}m, 高度={TARGET_ALTITUDE}m）")
        print(f"{'='*80}")

        state = env.reset()
        done = False
        step = 0
        total_reward = 0.0

        while not done:
            q_values = model.predict(state.reshape(1, -1), verbose=0)[0]
            action = int(q_values.argmax())

            next_state, reward, done, info = env.step(action)

            x, y, z = env._get_position_raw()
            altitude = -z
            distance, (dx, dy, dz) = env._get_distance_3d()

            print(f"Step {step:3d} | x:{x:6.2f} y:{y:6.2f} 高度:{altitude:6.2f} m | "
                  f"誤差(x,y,z): ({dx:5.1f}, {dy:5.1f}, {-dz:5.1f}) | 3D誤差: {distance:5.1f} m | "
                  f"動作: {ACTION_NAMES[action]:6s} | 獎勵: {reward:6.2f}")

            state = next_state
            total_reward += reward
            step += 1
            time.sleep(0.05)

        result = info.get("result", "unknown")
        print(f"\n>> 回合結束: {result}  總步數: {step}  總獎勵: {total_reward:.2f}")

        if result == "success":
            run_extended_verification(env, model, step, EXTENDED_VERIFY_SECONDS)

    env.close()
    print("\n評估完成。")


if __name__ == "__main__":
    main()