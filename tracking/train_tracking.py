"""
多維度視覺追蹤 DQN 訓練主程式（課程學習版）
==================================================================================
執行方式: python train_tracking.py

課程學習流程：
  Stage 1 近距離粉球追蹤 -> Stage 2 單框穿越 -> Stage 3 雙框 + 粉球全任務
  以「近 ROLLING_WINDOW 回合成功率」作為晉級門檻，而非固定回合數硬切換，
  避免模型在還沒穩定掌握前一階段技能時就被推入更難的任務。
"""

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(">> 成功設定 TensorFlow 動態 GPU 記憶體配置")
    except RuntimeError as e:
        print(e)

import os
from collections import deque

from airsim_env_tracking import VisionTrackingEnv
from dqn_agent import DQNAgent

NUM_EPISODES = 2000
SAVE_EVERY = 25
MODEL_DIR = "checkpoints_tracking"
LOG_FILE = "training_log_tracking.csv"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")

ROLLING_WINDOW = 20
SUCCESS_RESULT_PREFIX = "success_target_reached"

# 每個階段晉級所需的「近 ROLLING_WINDOW 回合成功率」門檻
STAGE_PROMOTION_THRESHOLD = {1: 0.7, 2: 0.5}
MAX_STAGE = 3


def is_success(result: str) -> bool:
    return result.startswith(SUCCESS_RESULT_PREFIX)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    env = VisionTrackingEnv(curriculum_stage=1)
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)

    with open(LOG_FILE, "w") as f:
        f.write("episode,stage,total_reward,steps,epsilon,result\n")

    recent_rewards = deque(maxlen=ROLLING_WINDOW)
    recent_success = deque(maxlen=ROLLING_WINDOW)
    best_avg_reward = -float("inf")
    stage = 1

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
        recent_rewards.append(total_reward)
        result = info.get("result", "unknown")
        recent_success.append(1.0 if is_success(result) else 0.0)

        print(f"[Ep {episode}/{NUM_EPISODES}] Stage={stage} Reward={total_reward:.2f} | "
              f"Steps={env.current_step} | Epsilon={agent.epsilon:.3f} | Result={result}")

        with open(LOG_FILE, "a") as f:
            f.write(f"{episode},{stage},{total_reward:.3f},{env.current_step},{agent.epsilon:.4f},{result}\n")

        if len(recent_rewards) == ROLLING_WINDOW:
            avg_reward = sum(recent_rewards) / ROLLING_WINDOW
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save(BEST_MODEL_PATH)
                print(f"  -> 【新紀錄】近 {ROLLING_WINDOW} 回合平均 Reward = {avg_reward:.2f}，已存檔。")

            success_rate = sum(recent_success) / ROLLING_WINDOW
            threshold = STAGE_PROMOTION_THRESHOLD.get(stage)
            if threshold is not None and success_rate >= threshold and stage < MAX_STAGE:
                stage += 1
                env.set_curriculum_stage(stage)
                agent.reset_epsilon_for_new_stage(warm_restart_value=0.3)
                recent_rewards.clear()
                recent_success.clear()
                best_avg_reward = -float("inf")
                print(f"  -> 【課程晉級】近 {ROLLING_WINDOW} 回合成功率 {success_rate:.0%} 達標，"
                      f"進入 Stage {stage}，epsilon 重置為 {agent.epsilon:.2f}。")

        if episode % SAVE_EVERY == 0:
            agent.save(os.path.join(MODEL_DIR, f"dqn_tracking_ep{episode}.keras"))

    env.close()
    print("訓練結束。")


if __name__ == "__main__":
    main()
