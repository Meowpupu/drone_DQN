"""
DQN 訓練主程式
================
執行方式: python train.py

【YOLO 整合】
  detect_target() 目前是佔位函式，回傳 None（代表沒有偵測到目標）。
  請替換成你實際的 YOLO 推論邏輯，回傳格式須為:
      {"bbox": (x1, y1, x2, y2), "img_w": 寬, "img_h": 高}
  或在未偵測到目標時回傳 None。
"""

import os
import numpy as np
import airsim

from airsim_env import AirSimDroneEnv, STATE_SIZE, ACTION_SIZE
from dqn_agent import DQNAgent


# ============ 訓練參數 ============
NUM_EPISODES = 2000
SAVE_EVERY = 50
MODEL_DIR = "checkpoints"
LOG_FILE = "training_log.csv"


def get_front_camera_image(env):
    responses = env.client.simGetImages([
        airsim.ImageRequest(env.camera_name, airsim.ImageType.Scene, False, False)
    ])
    resp = responses[0]
    img1d = np.frombuffer(bytes(resp.image_data_uint8), dtype=np.uint8)
    img_rgb = img1d.reshape(resp.height, resp.width, 3)
    return img_rgb


def detect_target(image):
    """
    佔位函式：請替換成實際的 YOLO 推論。
    範例（實際使用時請取消註解並改用你的模型）:

        results = yolo_model(image)
        if len(results) == 0:
            return None
        x1, y1, x2, y2 = results[0].boxes.xyxy[0]
        return {"bbox": (x1, y1, x2, y2), "img_w": image.shape[1], "img_h": image.shape[0]}
    """
    return None


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    env = AirSimDroneEnv()
    agent = DQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)

    with open(LOG_FILE, "w") as f:
        f.write("episode,total_reward,steps,epsilon,result\n")

    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset()
        total_reward = 0.0
        done = False
        info = {}

        while not done:
            action = agent.act(state, training=True)

            image = get_front_camera_image(env)
            detection = detect_target(image)

            next_state, reward, done, info = env.step(action, detection=detection)

            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            state = next_state
            total_reward += reward

        result = info.get("result", "unknown")
        print(f"[Episode {episode}/{NUM_EPISODES}] "
              f"Reward={total_reward:.2f}  Steps={env.current_step}  "
              f"Epsilon={agent.epsilon:.3f}  Result={result}")

        with open(LOG_FILE, "a") as f:
            f.write(f"{episode},{total_reward:.3f},{env.current_step},{agent.epsilon:.4f},{result}\n")

        if episode % SAVE_EVERY == 0:
            ckpt_path = os.path.join(MODEL_DIR, f"dqn_ep{episode}.keras")
            agent.save(ckpt_path)
            print(f"  -> 模型已儲存: {ckpt_path}")

    env.close()
    print("訓練結束。")


if __name__ == "__main__":
    main()
