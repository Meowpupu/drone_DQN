"""
DQN Agent (TensorFlow / Keras)
================================
標準 DQN：Q-network + Target network + Experience Replay + epsilon-greedy
"""

import numpy as np
import random
from collections import deque

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        buffer_size=50000,
        batch_size=64,
        target_update_freq=500,   # 每幾個訓練步驟同步一次 target network
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.memory = deque(maxlen=buffer_size)
        self.train_step_count = 0

        self.q_network = self._build_model(learning_rate)
        self.target_network = self._build_model(learning_rate)
        self.update_target_network()

    def _build_model(self, learning_rate):
        model = models.Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(self.action_size, activation="linear"),
        ])
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss="mse",
        )
        return model

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        if training and np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)

        state = np.expand_dims(state, axis=0)
        q_values = self.q_network.predict(state, verbose=0)[0]
        return int(np.argmax(q_values))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states = np.array([b[0] for b in batch], dtype=np.float32)
        actions = np.array([b[1] for b in batch], dtype=np.int32)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch], dtype=np.float32)
        dones = np.array([b[4] for b in batch], dtype=np.float32)

        q_current = self.q_network.predict(states, verbose=0)
        q_next = self.target_network.predict(next_states, verbose=0)

        targets = q_current.copy()
        max_next_q = np.max(q_next, axis=1)
        for i in range(self.batch_size):
            targets[i, actions[i]] = rewards[i] + (1 - dones[i]) * self.gamma * max_next_q[i]

        history = self.q_network.fit(states, targets, epochs=1, verbose=0, batch_size=self.batch_size)

        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.update_target_network()

        return history.history["loss"][0]

    def save(self, path):
        self.q_network.save(path)

    def load(self, path):
        self.q_network = models.load_model(path)
        self.update_target_network()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay