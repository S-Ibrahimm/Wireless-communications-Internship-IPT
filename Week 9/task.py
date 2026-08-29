import random
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.special import erfc
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ==========================================
# 1. ENVIRONMENT DEFINITION
# ==========================================


class WirelessEnv(gym.Env):
    """Custom Gymnasium Environment for Wireless Power Control using Pure dB Scale."""

    def __init__(self):
        super().__init__()

        # [UPDATED] Expanded transmit power choices to include negative dBm levels so close-range users can achieve target SNR
        self.power_lvls = [-10, -5, 0, 5, 10, 15, 20]
        self.modulations = ['BPSK' , 'QPSK', '16-QAM']
# [NEW WEEK 8] RIS phase configuration choices (0: Unaligned/Random, 1: Perfect Alignment with N^2 gain)
        self.ris_phase_modes = [0, 1]
        
    # [NEW WEEK 8] Joint action space expanded to (Power, Modulation, RIS Phase Shift)
        self.action_list = [(p, m, r) for p in self.power_lvls for m in self.modulations for r in self.ris_phase_modes]
        self.action_space = spaces.Discrete(len(self.action_list))

# [NEW WEEK 9] Expanded observation range for NTN distances (10m to 3000m)
        # Continuous State Space: [Distance (m), Channel Gain (dB)]
        low = np.array([10.0, -150.0], dtype=np.float32)
        high = np.array([3000.0, -30.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=low, high=high, shape=(2,), dtype=np.float32
        )

        # Constants & Hyperparameters
        self.target_snr = 5.0  # Target SNR in dB
        self.noise_floor = -90.0  # Thermal Noise Floor in dBm
        self.eta = 3.0  # Path Loss Exponent
        self.lam = 0.01  # [UPDATED] Lowered power penalty factor so it doesn't overwhelm SNR alignment
        self.max_steps = 50  # Steps per episode
        self.current_step = 0

        # [NEW WEEK 8] RIS System Geometry & Physical Parameters
        self.num_ris_elements = 128  # Number of reflecting elements (N)-> less elements were causing issues
        self.eta_ris = 2.0
        self.d_bs_ris = 50.0        # Fixed distance from Base Station to RIS (meters)

# [NEW WEEK 9] NTN (Non-Terrestrial Network) Physical Parameters
        self.altitude = 1000.0
        self.carrier_freq_hz = 2.4e9 #2.4GHz
        self.rician_k_factor_db = (6.0)
    # Scale distance, channel gain, and SNR into a standard [0, 1] range to ensure stable neural network learning
    def _get_normalized_state(self):
        norm_dist = (self.distance - 10.0) / (3000.0 - 10.0)
        norm_gain = (self.chan_gain_db - (-150.0)) / (-30.0 - (-150.0))
       
    # Clip values to ensure neural network inputs remain bounded in [0, 1]
        state = np.array([norm_dist, norm_gain], dtype=np.float32)
        return np.clip(state, 0.0, 1.0)
    
    def _compute_channel_gain_db(self, distance, ris_mode=0):
        # [NEW WEEK 9] 1. Calculate actual 3D distance through the air
        d_3d_direct = np.sqrt(distance**2 + self.altitude**2)

        c = 3e8
        #FSPL formula in db
        fspl_direct_db = (
            20.0 * np.log10(d_3d_direct) + 20.0 * np.log10(self.carrier_freq_hz)
            + 20.0 * np.log10((4.0*np.pi)/c)
            )

        # [NEW WEEK 9] 3. Rician Fading for NTN (Strong direct path + some scattered paths)
        k_linear = 10.0 ** (self.rician_k_factor_db / 10.0)
        los_component = np.sqrt(k_linear / (k_linear + 1.0))
        # Generate random scattering (complex numbers for radio wave phases)
        scatter_component = np.sqrt(1.0 / (k_linear + 1.0)) * (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2.0)

        # Total fading power is the magnitude squared of the combined paths
        fading_linear = np.abs(los_component + scatter_component)**2
        # Final direct gain
        gain_direct_linear = (10.0 ** (-fspl_direct_db / 10.0)) * max(fading_linear, 1e-4)
        # [NEW WEEK 9] 4. Reflected Link Channel Gain (Drone -> RIS -> User)
        # The RIS is on the ground at 'd_bs_ris' meters horizontally from the drone
        d_3d_drone_to_ris = np.sqrt(self.d_bs_ris**2 + self.altitude**2)
        d_ris_to_user = max(abs(distance - self.d_bs_ris), 1.0)
        
        # Drone to RIS uses Free-Space Path Loss (it's up in the air)
        fspl_drone_to_ris_db = (
            20.0 * np.log10(d_3d_drone_to_ris) 
            + 20.0 * np.log10(self.carrier_freq_hz) 
            + 20.0 * np.log10((4.0 * np.pi) / c)
        )

        # RIS to User uses Terrestrial path loss (both are on the ground)
        pl_0_db = 38.5 # Reference path loss at 1m for ground
        path_loss_ris_to_user_db = pl_0_db + (10.0 * self.eta_ris * np.log10(d_ris_to_user))
        
        total_ris_path_loss_db = fspl_drone_to_ris_db + path_loss_ris_to_user_db

        array_gain = (self.num_ris_elements**2) if ris_mode == 1 else 1.0
        gain_ris_linear = (
            10.0 ** (-total_ris_path_loss_db / 10.0)
        ) * array_gain

        # Superposition of direct and reflected paths
        total_gain_linear = gain_direct_linear + gain_ris_linear
        return 10.0 * np.log10(total_gain_linear)
       

    def _compute_ber(self,snr_db,modulation):
        snr_linear = 10.0**(snr_db / 10.0)
        def q_func(x):
            return 0.5 * erfc(x / np.sqrt(2.0))
        if modulation == 'BPSK':
            return float(q_func(np.sqrt(2.0 * snr_linear)))
        elif modulation == 'QPSK':
            return float(q_func(np.sqrt(snr_linear)))
        elif modulation == '16-QAM':
            return float(0.75 * q_func(np.sqrt(0.2 * snr_linear)))
        return 0.5
    def _compute_data_rate(self,modulation):
        rate_map = {'BPSK' : 1.0, 'QPSK' : 2.0, '16-QAM' : 4.0}
        return rate_map.get(modulation , 1.0)
            

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Randomize initial distance between 10m and 3000m
        self.distance = np.random.uniform(10.0, 3000.0)

        # Compute initial channel gain (default RIS mode = 0)
        self.chan_gain_db = self._compute_channel_gain_db(self.distance, ris_mode=0)

        # Compute initial SNR using first power level from action_list
        initial_p_tx_dbm, _, _ = self.action_list[0]
        p_rx_dbm = initial_p_tx_dbm + self.chan_gain_db
        self.snr_db = p_rx_dbm - self.noise_floor

        # Return normalized state vector and empty info dict
        return self._get_normalized_state(), {}

    def step(self, action):
        self.current_step += 1
        
        # 1. Unpack action tuple
        p_tx_dbm, modulation, ris_mode = self.action_list[action]
        
            # 2. Compute Channel Gain & Received Signal parameters for current state
        self.chan_gain_db = self._compute_channel_gain_db(
                self.distance, ris_mode=ris_mode
            )
        p_rx_dbm = p_tx_dbm + self.chan_gain_db
        self.snr_db = p_rx_dbm - self.noise_floor
        
            # 3. Calculate metrics
        ber = self._compute_ber(self.snr_db, modulation)
        data_rate = self._compute_data_rate(modulation)
        
            # 4. Reward calculation
        snr_gap = self.snr_db - self.target_snr
        
        if ber <= 0.001:
                # High reward when target BER is met, plus bonus for higher data rates
                reward = data_rate + 5.0 - (self.lam * max(0, p_tx_dbm))
        else:
                # Continuous reward signal guiding agent towards target SNR when BER threshold fails
                reward = snr_gap - (self.lam * max(0, p_tx_dbm))
        
            # 5. Environment transitions to next state (user mobility)
        self.distance = np.clip(
                self.distance + np.random.uniform(-1.0, 1.0), 10.0, 3000.0
            )
        
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
# 4. BASELINE HEURISTICS (NEW WEEK 7)
# ==========================================


def baseline_random(env):
    """Baseline 1: Selects a completely random action from the joint action space."""
    return random.randint(0, env.action_space.n - 1)


def baseline_fixed(env):
    """Baseline 2: Fixed heuristic using Max Power (20 dBm), QPSK, and No RIS (mode 0)."""
    target_tuple = (20, "QPSK", 0)
    return env.action_list.index(target_tuple)

def baseline_greedy(snr_db,env):
    """Baseline 3: Traditional rule-based control with RIS phase alignment (mode 1)."""
    if snr_db >= 15.0:
        target_tuple = (5, "16-QAM", 1)  # High SNR -> Low power, high modulation, RIS ON
    elif snr_db >= 8.0:
        target_tuple = (10, "QPSK", 1)   # Medium SNR -> Moderate power, QPSK, RIS ON
    else:
        target_tuple = (20, "BPSK", 1)   # Low SNR -> Max power, robust modulation, RIS ON

    return env.action_list.index(target_tuple)

# ==========================================
# 5. TRAINING LOOP & EVALUATION
# ==========================================

if __name__ == "__main__":  # boilerplate
    env = WirelessEnv()  # create an instance of wireless env
    state_dim = env.observation_space.shape[0]  # converts tuple to integers
    action_dim = env.action_space.n  # tells how many action choices agent has

    agent = DQNAgent(state_dim, action_dim)

    num_episodes = 2000
    batch_size = 64
    target_update_freq = 10

    # Track recent episode rewards to print a proper moving average
    recent_rewards = []
    # [NEW WEEK 9] List to store the average rewards specifically for our plot
    plot_rewards = []

    print("=== TRAINING START ===")
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
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
        ):  # Every 10 episodes, copy the weights of policy_net over to target_net
            agent.update_target_network()

        if episode % 50 == 0:
            # Corrected calculation to show average return over the last 50 episodes
            avg_reward_50_episodes = np.mean(recent_rewards[-50:])

            # [NEW WEEK 9] Save this average reward to draw on our graph later
            plot_rewards.append(avg_reward_50_episodes)
            print(
                f"Episode {episode:3d}/{num_episodes} | Avg Total Reward (last 50 eps): {avg_reward_50_episodes:.2f} | Epsilon: {agent.epsilon:.2f}"
            )

    print("\n=== EVALUATION RUN: DQN AGENT vs BASELINES ===")
    eval_state, _ = env.reset()
    print(f"User Initial Distance: {env.distance:.1f} meters\n")

    for step in range(1, 6):
         # Capture SNR before the DQN step alters the environment state
            current_snr = env.snr_db
        
            # 1. Greedy Baseline evaluates the pre-step state
            greedy_act = baseline_greedy(current_snr, env)
            g_power, g_mod, g_ris = env.action_list[greedy_act]
        
            # 2. DQN Agent selects and executes action
            dqn_action = agent.select_action(eval_state, evaluate=True)
            chosen_power, chosen_mod, chosen_ris = env.action_list[dqn_action]
        
            eval_state, reward, _, _, _ = env.step(dqn_action)
            ber = env._compute_ber(env.snr_db, chosen_mod)
        
            print(
                f"Step {step} | Dist: {env.distance:.1f}m | "
                f"DQN Choice: [{chosen_power:2d} dBm, {chosen_mod:6s}, RIS:{chosen_ris}] -> SNR: {env.snr_db:.1f}dB, BER: {ber:.4f}, Reward: {reward:.2f} | "
                f"Greedy Rule: [{g_power:2d} dBm, {g_mod:6s}, RIS:{g_ris}]"
            )

            # [NEW WEEK 9] Generate and show the Learning Curve Plot
    print("\n=== GENERATING LEARNING CURVE PLOT ===")
    plt.figure(figsize=(10, 6))
    
    # We recorded a data point every 50 episodes, so the x-axis should match that
    x_axis = range(50, num_episodes + 1, 50)
    
    plt.plot(x_axis, plot_rewards, label='DQN Agent', color='blue', linewidth=2)
    plt.title('Agent Learning Curve (NTN Drone Scenario)')
    plt.xlabel('Training Episodes')
    plt.ylabel('Average Reward (per 50 episodes)')
    plt.grid(True)
    plt.legend()
    
    # This will open a new window displaying your graph!
    plt.show()
   