"""
定高懸停模型評估腳本
======================
載入訓練好的 DQN 模型，用「純粹依賴學到的策略」（不隨機探索）飛行，
並在終端機即時印出每一步的座標、速度、動作等資訊。

執行方式:
    python evaluate_altitude.py                          # 自動載入最新的 checkpoint
    python evaluate_altitude.py checkpoints_altitude/dqn_altitude_ep200.keras   # 指定模型路徑
"""

import sys
import glob
import os
import time

from tensorflow.keras import models
from airsim_env_altitude import AltitudeHoverEnv, TARGET_ALTITUDE


ACTION_NAMES = {0: "上升 ↑", 1: "下降 ↓", 2: "懸停 ─"}


def find_latest_checkpoint(model_dir="checkpoints_altitude"):
    files = glob.glob(os.path.join(model_dir, "*.keras"))
    if not files:
        return None
    # 依檔名裡的回合數排序，取最新的
    files.sort(key=lambda f: int("".join(filter(str.isdigit, os.path.basename(f)))))
    return files[-1]


def main():
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = find_latest_checkpoint()
        if model_path is None:
            print("找不到任何 checkpoint，請確認 checkpoints_altitude 資料夾裡有 .keras 檔案，"
                  "或直接指定路徑：python evaluate_altitude.py <模型路徑>")
            return

    print(f"載入模型: {model_path}\n")
    model = models.load_model(model_path)

    env = AltitudeHoverEnv()

    num_episodes = 3  # 展示幾回合，可自行調整
    for ep in range(1, num_episodes + 1):
        print(f"\n{'='*60}")
        print(f"第 {ep}/{num_episodes} 回合開始（目標高度: {TARGET_ALTITUDE} 公尺）")
        print(f"{'='*60}")

        state = env.reset()
        done = False
        step = 0
        total_reward = 0.0

        while not done:
            # 純粹用模型預測，不做隨機探索
            q_values = model.predict(state.reshape(1, -1), verbose=0)[0]
            action = int(q_values.argmax())

            next_state, reward, done, info = env.step(action)

            altitude = env._get_altitude()
            vz = env._get_vertical_velocity()
            error = TARGET_ALTITUDE - altitude

            print(f"Step {step:3d} | 高度: {altitude:6.2f} m | 垂直速度: {vz:6.2f} m/s | "
                  f"誤差: {error:6.2f} m | 動作: {ACTION_NAMES[action]} | "
                  f"Q值: [{q_values[0]:.2f}, {q_values[1]:.2f}, {q_values[2]:.2f}] | "
                  f"獎勵: {reward:6.2f}")

            state = next_state
            total_reward += reward
            step += 1
            time.sleep(0.05)  # 讓輸出速度適合閱讀，可拿掉加快顯示

        result = info.get("result", "unknown")
        print(f"\n>> 回合結束: {result}  總步數: {step}  總獎勵: {total_reward:.2f}")

    env.close()
    print("\n評估完成。")


if __name__ == "__main__":
    main()
