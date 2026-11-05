"""
視覺追蹤模型評估腳本
======================================================
執行方式: python evaluate_tracking.py checkpoints_tracking/best_model.keras
"""

import sys
import glob
import os
import time
import numpy as np
from tensorflow.keras import models
from airsim_env_tracking import VisionTrackingEnv
from dqn_agent import DuelingCombine

ACTION_NAMES = {
    0: "強力前進", 1: "精準巡航", 2: "左平移", 3: "右平移",
    4: "爬升", 5: "降落", 6: "左旋轉", 7: "右旋轉",
    8: "前進左轉", 9: "前進右轉", 10: "前進爬升", 11: "前進下降", 12: "懸停"
}

def find_latest_checkpoint(model_dir="checkpoints_tracking"):
    files = glob.glob(os.path.join(model_dir, "*.keras"))
    if not files: return None
    files.sort(key=lambda f: os.path.getmtime(f))
    return files[-1]

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else find_latest_checkpoint()
    if model_path is None:
        print("找不到模型檔案！")
        return

    print(f"載入模型: {model_path}\n")
    model = models.load_model(model_path, custom_objects={"DuelingCombine": DuelingCombine})
    env = VisionTrackingEnv()

    for ep in range(1, 4):
        print(f"\n{'='*80}\n第 {ep} 回合測試開始\n{'='*80}")
        state = env.reset()
        done = False
        step = 0
        total_reward = 0.0

        while not done:
            q_values = model.predict(state.reshape(1, -1), verbose=0)[0]
            action = int(q_values.argmax())
            next_state, reward, done, info = env.step(action)
            
            raw = env.last_raw_state
            dx_img, dy_img, target_type = raw[0], raw[1], raw[3]
            alt = env._last_abs_altitude
            target_str = "粉球" if target_type == 1.0 else ("黃框" if target_type == 0.0 else "無")
            
            print(f"Step {step:3d} | 鎖定:{target_str} | 偏移({dx_img:5.2f}, {dy_img:5.2f}) | "
                  f"高度:{alt:5.2f}m | 動作: {ACTION_NAMES[action]:6s} | 獎勵: {reward:5.2f}")

            state = next_state
            total_reward += reward
            step += 1
            time.sleep(0.05)

        print(f"\n>> 回合結束: {info.get('result', 'unknown')}  總獎勵: {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()