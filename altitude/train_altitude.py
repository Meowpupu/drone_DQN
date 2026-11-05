"""
定高懸停 DQN 訓練主程式（簡化版任務）
========================================
執行方式: python train_altitude.py

目標: 無人機從地面起飛，學會爬升到 30 公尺並穩定懸停。
這是驗證整套 DQN 訓練管線的第一步測試，狀態/動作都刻意簡化。
"""

import os
from airsim_env_altitude import AltitudeHoverEnv
from dqn_agent import DQNAgent


NUM_EPISODES = 300      # 這個簡化任務應該遠比完整版更快收斂
SAVE_EVERY = 25
MODEL_DIR = "checkpoints_altitude"
LOG_FILE = "training_log_altitude.csv"


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    env = AltitudeHoverEnv()
    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        epsilon_decay=0.98,   # 任務簡單，讓探索率下降快一點
    )

    with open(LOG_FILE, "w") as f:
        f.write("episode,total_reward,steps,epsilon,result\n")

    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset()
        total_reward = 0.0
        done = False
        info = {}

        while not done:
            action = agent.act(state, training=True)
            next_state, reward, done, info = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            state = next_state
            total_reward += reward
        
        agent.decay_epsilon()

        result = info.get("result", "unknown")
        print(f"[Episode {episode}/{NUM_EPISODES}] "
              f"Reward={total_reward:.2f}  Steps={env.current_step}  "
              f"Epsilon={agent.epsilon:.3f}  Result={result}")

        with open(LOG_FILE, "a") as f:
            f.write(f"{episode},{total_reward:.3f},{env.current_step},{agent.epsilon:.4f},{result}\n")

        if episode % SAVE_EVERY == 0:
            ckpt_path = os.path.join(MODEL_DIR, f"dqn_altitude_ep{episode}.keras")
            agent.save(ckpt_path)
            print(f"  -> 模型已儲存: {ckpt_path}")

    env.close()
    print("訓練結束。")


if __name__ == "__main__":
    main()
