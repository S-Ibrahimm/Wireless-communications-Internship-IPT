import random
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# 1. ENVIRONMENT DEFINITION
# ==========================================


class WirelessEnv(gym.Env):
    """Custom Gymnasium Environment for Wireless Power Control using Pure dB Scale."""

    def __init__(self):
        super().__init__()

        # [UPDATED] Expanded transmit power choices to include negative dBm levels so close-range users can achieve target SNR
        self.power_lvls = [-10, -5, 0, 5, 10, 15, 20]
        self.action_space = spaces.Discrete(len(self.power_lvls))

        # Continuous State Space: [Distance (m), Channel Gain (dB), SNR (dB)]
        low = np.array([10.0, -150.0, -20.0], dtype=np.float32)
        high = np.array([500.0, -30.0, 50.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=low, high=high, shape=(3,), dtype=np.float32
        )

        # Constants & Hyperparameters
        self.target_snr = 20.0  # Target SNR in dB
        self.noise_floor = -90.0  # Thermal Noise Floor in dBm
        self.eta = 3.5  # Path Loss Exponent
        self.lam = 0.01  # [UPDATED] Lowered power penalty factor so it doesn't overwhelm SNR alignment
        self.max_steps = 50  # Steps per episode
        self.current_step = 0

    # Scale distance, channel gain, and SNR into a standard [0, 1] range to ensure stable neural network learning
    def _get_normalized_state(self):
        norm_dist = (self.distance - 10.0) / (500.0 - 10.0)
        norm_gain = (self.chan_gain_db - (-150.0)) / (-30.0 - (-150.0))
        norm_snr = (self.snr_db - (-20.0)) / (50.0 - (-20.0))
        return np.array([norm_dist, norm_gain, norm_snr], dtype=np.float32)

    def _compute_channel_gain_db(self, distance):
        path_loss_db = 10.0 * self.eta * np.log10(distance)
        fading_linear = np.random.exponential(1.0)
        fading_db = 10.0 * np.log10(max(fading_linear, 1e-4))
        return -path_loss_db + fading_db

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        self.distance = np.random.uniform(10.0, 500.0)

        # Compute block fading channel gain ONCE per episode base state
        self.chan_gain_db = self._compute_channel_gain_db(self.distance)

        initial_p_tx_dbm = self.power_lvls[0]
        p_rx_dbm = initial_p_tx_dbm + self.chan_gain_db
        self.snr_db = p_rx_dbm - self.noise_floor

        # Return the normalized state vector to prevent large inputs from biasing network learning
        return self._get_normalized_state(), {}

    def step(self, action):
        self.current_step += 1

        p_tx_dbm = self.power_lvls[action]

        # [NEW] Simulate small user movement / channel drift per step so state transitions are dynamic and meaningful
        self.distance = np.clip(
            self.distance + np.random.uniform(-1.0, 1.0), 10.0, 500.0
        )
        self.chan_gain_db = self._compute_channel_gain_db(self.distance)

        p_rx_dbm = p_tx_dbm + self.chan_gain_db
        self.snr_db = p_rx_dbm - self.noise_floor

        snr_error = abs(self.target_snr - self.snr_db)
        reward = -snr_error - (self.lam * max(0, p_tx_dbm))

        # Return normalized state vector for next_state
        next_state = self._get_normalized_state()

        terminated = False
        truncated = self.current_step >= self.max_steps
        return next_state, reward, terminated, truncated, {}


# ==========================================
# 2. DQN COMPONENTS (BUFFER & NETWORK)
# ==========================================


class ReplayBuffer:

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        return (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.int64),
            np.array(reward, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array(done, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.fc(x)


# ==========================================
# 3. DQN AGENT IMPLEMENTATION
# ==========================================


class DQNAgent:

    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.99,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.policy_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        # copy stuff of policy net to target network
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()

    def select_action(self, state, evaluate=False):
        # epsilon greedy policy
        if not evaluate and random.random() < self.epsilon:
            # means that we are training and not evaluatng(testing) and a random num <epsilon -> explore
            return random.randint(
                0, self.action_dim - 1
            )  # action_dim-1 bcz we want an index between 0-action dim and act dim starts from 0

        state_t = torch.FloatTensor(state).unsqueeze(0)
        # NNs expect data to be in batches (batch_size, num_features). unsqueeze(0) transforms a 1D state into a 2D batch
        with torch.no_grad():
            q_values = self.policy_net(
                state_t
            )  # orward pass through the policy Q-network
        return q_values.argmax(
            dim=1
        ).item()  # select largest action value(since dim=1) and convert it into int(using item)

    def train_step(self, batch_size=64):
        if (
            len(self.buffer) < batch_size
        ):  # Prevents training until the Replay Buffer contains at least batch_size transitions
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(
            batch_size
        )
        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.FloatTensor(dones).unsqueeze(1)

        # Current Q-values
        q_values = self.policy_net(states_t).gather(
            1, actions_t
        )  # do forward pass and get only the q value for specific action_t

        # Target Q-values using Target Network
        with torch.no_grad():
            # the line below does forw pass and gets q values in q table from which we get only the max q val tensor, then we convert
            # it into a matrix by adding a column at 1 , i.e now its like batch size*1
            max_next_q = self.target_net(next_states_t).max(1)[0].unsqueeze(1)
            target_q = (
                rewards_t + (1 - dones_t) * self.gamma * max_next_q
            )  # bellman eq

        loss = nn.MSELoss()(q_values, target_q)

        self.optimizer.zero_grad()  # reset all prev derivatives so tat we can now update em with new replay buffer gradients
        loss.backward()

        # Prevent exploding gradients by capping high weight gradients during backpropagation
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)

        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


# ==========================================
# 4. TRAINING LOOP & EVALUATION
# ==========================================

if __name__ == "__main__":  # boilerplate
    env = WirelessEnv()  # create an instance of wireless env
    state_dim = env.observation_space.shape[0]  # converts tuple to integers
    action_dim = env.action_space.n  # tells how many action choices agent has

    agent = DQNAgent(state_dim, action_dim)

    num_episodes = 1000
    batch_size = 64
    target_update_freq = 10

    # Track recent episode rewards to print a proper moving average
    recent_rewards = []

    print("=== TRAINING START ===")
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(
                action
            )  # '_' means info , just stuff for gym
            done = terminated or truncated

            agent.buffer.push(state, action, reward, next_state, done)
            agent.train_step(
                batch_size
            )  # Triggers a single gradient descent update step on policy_net by sampling a mini-batch from the replay buffer.

            state = next_state
            total_reward += reward

            if done:
                break

        agent.update_epsilon()

        # Append total episode reward
        recent_rewards.append(total_reward)

        if (
            episode % target_update_freq == 0
        ):  # Every  every 10 episodes,copy the weights of policy_net over to target_net
            agent.update_target_network()

        if episode % 50 == 0:
            # Corrected calculation to show average return over the last 50 episodes
            avg_reward_50_episodes = np.mean(recent_rewards[-50:])
            print(
                f"Episode {episode:3d}/{num_episodes} | Avg Total Reward (last 50 eps): {avg_reward_50_episodes:.2f} | Epsilon: {agent.epsilon:.2f}"
            )

    print("\n=== EVALUATION RUN ===")
    eval_state, _ = env.reset()  # resetting sets a random distance of Rx-Tx

    # Read distance from env.distance directly, since eval_state now contains normalized features
    print(f"User Initial Distance: {env.distance:.1f} meters")

    for step in range(1, 6):
        action = agent.select_action(eval_state, evaluate=True)
        chosen_power = env.power_lvls[action]
        eval_state, reward, _, _, _ = env.step(action)

        # Access real physical variables directly from environment instance for accurate print statements
        print(
            f"Step {step} | Distance: {env.distance:.1f}m | Power Chosen: {chosen_power:2d} dBm | Achieved SNR: {env.snr_db:.2f} dB | Reward: {reward:.2f}"
        )