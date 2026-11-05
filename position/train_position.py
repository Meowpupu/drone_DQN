"""
三維定點導航 DQN 訓練主程式（v4，配合修正版環境 + Double DQN + 早停機制）
==================================================================================
執行方式: python train_position.py

【新增：早停機制】
  觀察到即使有 Double DQN，訓練夠久之後 Q 網路還是可能逐漸漂移、
  把已經學好的策略覆寫壞掉（黃金期維持一段時間後又開始退化）。
  與其試圖讓訓練永遠不發散，更務實的做法是：一旦連續穩定達到高成功率，
  就自動停止，把當下的模型存起來，不要冒險繼續訓練。

【最佳模型自動保存機制】
  追蹤最近 20 回合的平均 reward，一旦創新高就另存一份 best_model.keras。
"""

import os
from collections import deque
from airsim_env_position import PositionNavEnv
from dqn_agent import DQNAgent


NUM_EPISODES = 400
SAVE_EVERY = 25
MODEL_DIR = "checkpoints_position"
LOG_FILE = "training_log_position.csv"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "early_stopped_model.keras")

ROLLING_WINDOW = 20
EARLY_STOP_WINDOW = 40         # 拉大視窗提升統計可信度
EARLY_STOP_SUCCESS_RATE = 0.9  # 成功率門檻
EARLY_STOP_MIN_STREAK = 15     # 額外要求：至少連續成功這麼多次，不能中間穿插失敗
EARLY_STOP_MIN_EPISODE = 60    # 至少訓練這麼多回合才開始檢查早停
EARLY_STOP_MAX_EPSILON = 0.3   # 【新增】只有 epsilon 降到這個值以下才開始檢查早停，
                                # 避免探索率還很高時，隨機動作把「本來會成功」的回合搞砸，
                                # 這種雜訊會讓早停條件被誤判為「還不夠好」而遲遲無法觸發


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    env = PositionNavEnv()
    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        epsilon_decay=0.992,
    )

    with open(LOG_FILE, "w") as f:
        f.write("episode,total_reward,steps,epsilon,result\n")

    recent_rewards = deque(maxlen=ROLLING_WINDOW)
    recent_results = deque(maxlen=EARLY_STOP_WINDOW)
    best_avg_reward = -float("inf")

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
        recent_results.append(1 if result == "success" else 0)

        print(f"[Episode {episode}/{NUM_EPISODES}] "
              f"Reward={total_reward:.2f}  Steps={env.current_step}  "
              f"Epsilon={agent.epsilon:.3f}  Result={result}")

        with open(LOG_FILE, "a") as f:
            f.write(f"{episode},{total_reward:.3f},{env.current_step},{agent.epsilon:.4f},{result}\n")

        if len(recent_rewards) == ROLLING_WINDOW:
            avg_reward = sum(recent_rewards) / ROLLING_WINDOW
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save(BEST_MODEL_PATH)
                print(f"  -> 【新紀錄】近 {ROLLING_WINDOW} 回合平均 Reward = {avg_reward:.2f}，"
                      f"已另存最佳模型: {BEST_MODEL_PATH}")

        if episode % SAVE_EVERY == 0:
            ckpt_path = os.path.join(MODEL_DIR, f"dqn_position_ep{episode}.keras")
            agent.save(ckpt_path)
            print(f"  -> 定期存檔: {ckpt_path}")

        # 早停檢查：同時滿足「成功率夠高」跟「連續一段時間不中斷」，
        # 且 epsilon 已降到夠低（避免高探索率時的隨機雜訊誤導判斷）
        if (episode >= EARLY_STOP_MIN_EPISODE
                and agent.epsilon <= EARLY_STOP_MAX_EPSILON
                and len(recent_results) == EARLY_STOP_WINDOW):
            success_rate = sum(recent_results) / EARLY_STOP_WINDOW

            # 計算目前連續成功的長度（從最後一筆往前數，中間不能有0）
            consecutive_streak = 0
            for r in reversed(recent_results):
                if r == 1:
                    consecutive_streak += 1
                else:
                    break

            if success_rate >= EARLY_STOP_SUCCESS_RATE and consecutive_streak >= EARLY_STOP_MIN_STREAK:
                agent.save(FINAL_MODEL_PATH)
                print(f"\n【早停觸發】最近 {EARLY_STOP_WINDOW} 回合成功率達 {success_rate:.0%}，"
                      f"且連續成功 {consecutive_streak} 次（epsilon={agent.epsilon:.3f}），"
                      f"已在第 {episode} 回合提前停止訓練。")
                print(f"早停模型位置: {FINAL_MODEL_PATH}")
                env.close()
                return

    env.close()
    print(f"\n訓練結束（跑完全部 {NUM_EPISODES} 回合，未觸發早停）。")
    print(f"最佳近 {ROLLING_WINDOW} 回合平均 Reward: {best_avg_reward:.2f}")
    print(f"最佳模型位置: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()