"""
DQN Agent (TensorFlow / Keras) — Dueling Double DQN 重構版
================================================================
相對舊版的關鍵改動：
1. Dueling 架構：拆成 V(s) 與 A(s,a) 兩條分支再合併，13 個離散動作中有不少
   語意上高度相似（例如「前進+左轉」vs「前進+右轉」），Dueling 能讓網路先學
   「這個狀態本身好不好」，不必每個動作各自重新學一次狀態價值，對動作維度
   較大、部分動作差異細微的場景特別有效，也能加速收斂。
2. Huber Loss 取代 MSE：30 維狀態正規化後仍可能因 YOLO 偵測跳動或碰撞瞬間
   產生離群的 TD-target，Huber 對離群值不敏感，避免單一異常樣本把梯度炸掉。
3. 顯式 train_step（tf.function + GradientTape）取代 model.fit()：
   避免 predict()/fit() 的 retracing 與額外開銷，對「單步呼叫 AirSim + 單步訓練」
   這種高頻小批次的迴圈有明顯提速。
4. Soft target update（Polyak averaging）取代每 N 步硬拷貝一次：
   硬拷貝會讓 bootstrap target 出現階梯式跳動，soft update 讓目標網路平滑跟隨，
   對本任務這種獎勵尺度已經被大幅壓縮、方差本來就不小的情況更穩定。
5. 梯度裁剪（global norm）避免早期高 epsilon、高方差樣本把權重一次推爆。
"""

import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


@tf.keras.utils.register_keras_serializable(package="dqn_agent")
class DuelingCombine(layers.Layer):
    """Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))"""

    def call(self, inputs):
        value, advantage = inputs
        advantage_mean = tf.reduce_mean(advantage, axis=1, keepdims=True)
        return value + (advantage - advantage_mean)


class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=3e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.997,
                 buffer_size=150000, batch_size=128, tau=0.005,
                 grad_clip_norm=10.0):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau  # soft target update 係數

        self.memory = deque(maxlen=buffer_size)
        self.train_step_count = 0

        self.optimizer = optimizers.Adam(learning_rate=learning_rate, clipnorm=grad_clip_norm)
        self.loss_fn = tf.keras.losses.Huber(delta=1.0)

        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.update_target_network(hard=True)

    def _build_model(self):
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(256, activation="relu")(inputs)
        x = layers.Dense(256, activation="relu")(x)
        shared = layers.Dense(128, activation="relu")(x)

        value = layers.Dense(64, activation="relu")(shared)
        value = layers.Dense(1, activation="linear", name="value")(value)

        advantage = layers.Dense(64, activation="relu")(shared)
        advantage = layers.Dense(self.action_size, activation="linear", name="advantage")(advantage)

        q_values = DuelingCombine()([value, advantage])
        return models.Model(inputs=inputs, outputs=q_values)

    def update_target_network(self, hard=False):
        if hard:
            self.target_network.set_weights(self.q_network.get_weights())
            return
        online_weights = self.q_network.get_weights()
        target_weights = self.target_network.get_weights()
        new_weights = [
            self.tau * ow + (1.0 - self.tau) * tw
            for ow, tw in zip(online_weights, target_weights)
        ]
        self.target_network.set_weights(new_weights)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        if training and np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        state = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        q_values = self.q_network(state, training=False)[0]
        return int(tf.argmax(q_values).numpy())

    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones):
        # Double DQN：用 online network 選動作，用 target network 評分
        next_q_online = self.q_network(next_states, training=False)
        next_actions = tf.argmax(next_q_online, axis=1, output_type=tf.int32)

        next_q_target = self.target_network(next_states, training=False)
        batch_indices = tf.range(tf.shape(next_actions)[0], dtype=tf.int32)
        next_action_indices = tf.stack([batch_indices, next_actions], axis=1)
        max_next_q = tf.gather_nd(next_q_target, next_action_indices)

        td_target = rewards + (1.0 - dones) * self.gamma * max_next_q
        td_target = tf.stop_gradient(td_target)

        action_indices = tf.stack([batch_indices, actions], axis=1)

        with tf.GradientTape() as tape:
            q_values = self.q_network(states, training=True)
            q_selected = tf.gather_nd(q_values, action_indices)
            loss = self.loss_fn(td_target, q_selected)

        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))
        return loss

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None
        batch = random.sample(self.memory, self.batch_size)
        states = tf.convert_to_tensor([b[0] for b in batch], dtype=tf.float32)
        actions = tf.convert_to_tensor([b[1] for b in batch], dtype=tf.int32)
        rewards = tf.convert_to_tensor([b[2] for b in batch], dtype=tf.float32)
        next_states = tf.convert_to_tensor([b[3] for b in batch], dtype=tf.float32)
        dones = tf.convert_to_tensor([b[4] for b in batch], dtype=tf.float32)

        loss = self._train_step(states, actions, rewards, next_states, dones)

        self.train_step_count += 1
        self.update_target_network(hard=False)  # 每個訓練步都做一次小幅 soft update
        return float(loss.numpy())

    def save(self, path):
        self.q_network.save(path)

    def load(self, path):
        self.q_network = models.load_model(
            path, custom_objects={"DuelingCombine": DuelingCombine}
        )
        self.update_target_network(hard=True)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def reset_epsilon_for_new_stage(self, warm_restart_value=0.3):
        """課程學習切換到新階段時，任務動態改變（例如從單框變雙框），
        用 warm restart 把 epsilon 拉回一個中間值，讓模型重新探索新加入的
        子任務，而不是在低 epsilon 下對新狀態分布過度保守。"""
        self.epsilon = max(self.epsilon, warm_restart_value)
