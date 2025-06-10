import time
import random
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from config import (
    REPLAY_MEMORY_SIZE, MIN_REPLAY_MEMORY_SIZE, MINIBATCH_SIZE, PREDICTION_BATCH_SIZE,
    TRAINING_BATCH_SIZE, UPDATE_TARGET_EVERY, MODEL_NAME, DISCOUNT, TAU, ALPHA
)

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        super().set_model(model)  # Call the parent class's set_model method

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step=self.step)
            self.writer.flush()

class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-DQN-{int(time.time())}")
        self.target_update_counter = 0

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def create_model(self):
        # Input: [speed, acceleration] (2 features)
        inputs = Input(shape=(2,))
        x = Dense(64, activation="relu")(inputs)
        x = Dense(64, activation="relu")(x)
        predictions = Dense(3, activation="linear")(x)
        model = Model(inputs=inputs, outputs=predictions)
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])
        return model

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)

        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)

        X = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step

        self.model.fit(np.array(X), np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)

        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        # state: [speed, acceleration]
        return self.model.predict(np.array(state).reshape(-1, 2))[0]

    def train_in_loop(self):
        X = np.random.uniform(size=(1, 2)).astype(np.float32)
        y = np.random.uniform(size=(1, 3)).astype(np.float32)
        self.model.fit(X, y, verbose=False, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)

class SACAgent:
    def __init__(self, state_dim=2, action_dim=1, action_bound=1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound

        self.gamma = DISCOUNT
        self.tau = TAU
        self.alpha = ALPHA  # Entropy coefficient

        self.actor = self.build_actor()
        self.critic_1 = self.build_critic()
        self.critic_2 = self.build_critic()
        self.target_critic_1 = self.build_critic()
        self.target_critic_2 = self.build_critic()

        self.target_critic_1.set_weights(self.critic_1.get_weights())
        self.target_critic_2.set_weights(self.critic_2.get_weights())

        self.actor_optimizer = Adam(learning_rate=0.0003)
        self.critic_optimizer = Adam(learning_rate=0.0003)

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-SAC-{int(time.time())}")
        self.training_initialized = False
        self.terminate = False

    def build_actor(self):
        state_input = Input(shape=(self.state_dim,))
        x = Dense(64, activation='relu')(state_input)
        x = Dense(64, activation='relu')(x)
        mu = Dense(self.action_dim, activation='sigmoid')(x)  # Output in [0,1]
        log_std = Dense(self.action_dim, activation='tanh')(x)  # log_std ∈ [-1,1]
        model = Model(state_input, [mu, log_std])
        return model

    def build_critic(self):
        state_input = Input(shape=(self.state_dim,))
        action_input = Input(shape=(self.action_dim,))
        concat = Concatenate()([state_input, action_input])
        x = Dense(64, activation='relu')(concat)
        x = Dense(64, activation='relu')(x)
        q = Dense(1, activation='linear')(x)
        model = Model([state_input, action_input], q)
        model.compile(optimizer=Adam(learning_rate=0.0003), loss="mse")  # <-- ADD THIS LINE
        return model

    def get_action(self, state, deterministic=False):
        state = np.array(state).reshape(1, -1)
        mu, log_std = self.actor(state)
        log_std = tf.clip_by_value(log_std, -20, 2)
        std = tf.exp(log_std)
        if deterministic:
            action = mu
        else:
            noise = tf.random.normal(shape=mu.shape)
            action = mu + std * noise
        action = tf.clip_by_value(action, 0, 1)
        return action.numpy()[0]

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        # For compatibility, return the actor's mean action (mu) for the given state
        state = np.array(state).reshape(1, -1)
        mu, _ = self.actor(state)
        return mu.numpy()[0]

    def train(self, batch_size=MINIBATCH_SIZE):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, batch_size)
        states = np.array([t[0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch]).reshape(-1, 1)
        rewards = np.array([t[2] for t in minibatch]).reshape(-1, 1)
        next_states = np.array([t[3] for t in minibatch])
        dones = np.array([t[4] for t in minibatch]).reshape(-1, 1)

        # Critic update
        next_mu, next_log_std = self.actor(next_states)
        next_log_std = tf.clip_by_value(next_log_std, -20, 2)
        next_std = tf.exp(next_log_std)
        next_noise = tf.random.normal(shape=next_mu.shape)
        next_action = tf.clip_by_value(next_mu + next_std * next_noise, 0, 1)
        next_log_prob = -0.5 * (((next_action - next_mu) / (next_std + 1e-6)) ** 2 + 2 * next_log_std + np.log(2 * np.pi))
        next_log_prob = tf.reduce_sum(next_log_prob, axis=1, keepdims=True)

        target_q1 = self.target_critic_1([next_states, next_action])
        target_q2 = self.target_critic_2([next_states, next_action])
        target_q = tf.minimum(target_q1, target_q2) - self.alpha * next_log_prob
        target = rewards + self.gamma * (1 - dones) * target_q

        with tf.GradientTape(persistent=True) as tape:
            current_q1 = self.critic_1([states, actions])
            current_q2 = self.critic_2([states, actions])
            critic_loss = tf.reduce_mean((current_q1 - target) ** 2 + (current_q2 - target) ** 2)

        critic_grads = tape.gradient(critic_loss, self.critic_1.trainable_variables + self.critic_2.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic_1.trainable_variables + self.critic_2.trainable_variables))

        # Actor update
        with tf.GradientTape() as tape:
            mu, log_std = self.actor(states)
            log_std = tf.clip_by_value(log_std, -20, 2)
            std = tf.exp(log_std)
            noise = tf.random.normal(shape=mu.shape)
            sampled_action = tf.clip_by_value(mu + std * noise, 0, 1)
            log_prob = -0.5 * (((sampled_action - mu) / (std + 1e-6)) ** 2 + 2 * log_std + np.log(2 * np.pi))
            log_prob = tf.reduce_sum(log_prob, axis=1, keepdims=True)
            q1 = self.critic_1([states, sampled_action])
            q2 = self.critic_2([states, sampled_action])
            q = tf.minimum(q1, q2)
            actor_loss = tf.reduce_mean(self.alpha * log_prob - q)

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # Soft update target critics
        for target_param, param in zip(self.target_critic_1.trainable_variables, self.critic_1.trainable_variables):
            target_param.assign(self.tau * param + (1 - self.tau) * target_param)
        for target_param, param in zip(self.target_critic_2.trainable_variables, self.critic_2.trainable_variables):
            target_param.assign(self.tau * param + (1 - self.tau) * target_param)

        # Logging
        self.tensorboard.update_stats(critic_loss=critic_loss.numpy(), actor_loss=actor_loss.numpy())

    def train_in_loop(self):
        # Dummy fit to initialize
        dummy_state = np.random.uniform(size=(1, self.state_dim)).astype(np.float32)
        dummy_action = np.random.uniform(size=(1, self.action_dim)).astype(np.float32)
        self.critic_1.fit([dummy_state, dummy_action], np.zeros((1, 1)), verbose=False)
        self.critic_2.fit([dummy_state, dummy_action], np.zeros((1, 1)), verbose=False)
        self.training_initialized = True
        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)