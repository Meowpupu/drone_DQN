"""
Dueling Double DQN (TensorFlow / Keras)
=======================================
- Dueling：Q = V(s) + A(s,a) - mean(A)
- Double DQN：online 選動作、target 估值
- Huber loss + 梯度裁剪 + 圖內 soft target update（tau）
- 預先配置的 numpy ring buffer（取樣快、無 list 轉 tensor 開銷）
- 探索：epsilon-greedy，隨機分支可依 action_prior 取樣（偏向含前進分量的動作，降低初期亂飛墜地）
- 區分「真終止」與「時間截斷」：截斷（timeout）不應把下一狀態價值截成 0，由訓練迴圈傳入 terminal 旗標
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


@tf.keras.utils.register_keras_serializable(package="dqn_agent")
class DuelingCombine(layers.Layer):
    def call(self, inputs):
        value, advantage = inputs
        return value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))


class ReplayBuffer:
    def __init__(self, capacity, state_size):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminals = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0

    def add(self, s, a, r, s2, terminal):
        i = self.idx
        self.states[i], self.actions[i], self.rewards[i] = s, a, r
        self.next_states[i], self.terminals[i] = s2, float(terminal)
        self.idx = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        ix = np.random.randint(0, self.size, size=batch_size)
        return (self.states[ix], self.actions[ix], self.rewards[ix],
                self.next_states[ix], self.terminals[ix])

    def __len__(self):
        return self.size


class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=3e-4, gamma=0.99,
                 buffer_size=200000, batch_size=128, tau=0.005, grad_clip_norm=10.0,
                 action_prior=None):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.epsilon = 1.0

        if action_prior is not None:
            p = np.asarray(action_prior, dtype=np.float64)
            assert p.shape == (action_size,), "action_prior 長度必須等於動作數"
            self.action_prior = p / p.sum()
        else:
            self.action_prior = None

        self.memory = ReplayBuffer(buffer_size, state_size)
        self.train_step_count = 0

        self.optimizer = optimizers.Adam(learning_rate=learning_rate, clipnorm=grad_clip_norm)
        self.loss_fn = tf.keras.losses.Huber(delta=1.0)

        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.target_network.set_weights(self.q_network.get_weights())

    def _build_model(self):
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(256, activation="relu")(inputs)
        x = layers.Dense(256, activation="relu")(x)
        shared = layers.Dense(128, activation="relu")(x)

        v = layers.Dense(64, activation="relu")(shared)
        v = layers.Dense(1, name="value")(v)
        a = layers.Dense(64, activation="relu")(shared)
        a = layers.Dense(self.action_size, name="advantage")(a)

        return models.Model(inputs=inputs, outputs=DuelingCombine()([v, a]))

    def remember(self, state, action, reward, next_state, terminal):
        self.memory.add(state, action, reward, next_state, terminal)

    def act(self, state, training=True):
        if training and np.random.rand() < self.epsilon:
            return int(np.random.choice(self.action_size, p=self.action_prior))
        q = self.q_network(tf.convert_to_tensor(state[None, :], dtype=tf.float32), training=False)[0]
        return int(tf.argmax(q).numpy())

    @tf.function
    def _train_step(self, states, actions, rewards, next_states, terminals):
        next_actions = tf.argmax(self.q_network(next_states, training=False), axis=1, output_type=tf.int32)
        next_q_target = self.target_network(next_states, training=False)
        rows = tf.range(tf.shape(next_actions)[0], dtype=tf.int32)
        max_next_q = tf.gather_nd(next_q_target, tf.stack([rows, next_actions], axis=1))
        td_target = tf.stop_gradient(rewards + (1.0 - terminals) * self.gamma * max_next_q)

        with tf.GradientTape() as tape:
            q_all = self.q_network(states, training=True)
            q_sel = tf.gather_nd(q_all, tf.stack([rows, actions], axis=1))
            loss = self.loss_fn(td_target, q_sel)

        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

        for tv, ov in zip(self.target_network.variables, self.q_network.variables):
            tv.assign(self.tau * ov + (1.0 - self.tau) * tv)
        return loss

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None
        s, a, r, s2, t = self.memory.sample(self.batch_size)
        loss = self._train_step(tf.constant(s), tf.constant(a), tf.constant(r),
                                tf.constant(s2), tf.constant(t))
        self.train_step_count += 1
        return float(loss.numpy())

    def save(self, path):
        self.q_network.save(path)

    def load(self, path):
        self.q_network = models.load_model(path, custom_objects={"DuelingCombine": DuelingCombine})
        self.target_network.set_weights(self.q_network.get_weights())
