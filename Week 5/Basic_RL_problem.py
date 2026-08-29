import numpy as np
import gymnasium as gym
from gymnasium import spaces

class WirelessPowerEnv(gym.Env):
    def __init__(self):
        super(WirelessPowerEnv, self).__init__()
        
        # 1. Actions: 5 discrete power levels (0 = lowest power, 4 = highest power)
        self.action_space = spaces.Discrete(5)
        self.power_levels = np.array([10, 14, 18, 22, 26]) # Power in dBm
        
        # 2. States: 11 discrete SNR levels (from 0 dB to 20 dB, stepped by 2)
        self.observation_space = spaces.Discrete(11)
        self.snr_bins = np.linspace(0, 20, 11)
        
        # 3. Wireless Constants
        self.target_snr = 12.0 # Target SNR in dB
        self.noise_floor = -90.0 # Noise floor in dBm
        
    def _get_snr(self, power):
        # Simulating a simple wireless channel with a random path loss between -95 dB and -105 dB
        path_loss = np.random.uniform(-105, -95)
        received_power = power + path_loss
        snr = received_power - self.noise_floor
        return np.clip(snr, 0, 20)
        
    def _get_state(self, snr):
        # Maps the continuous SNR value to the closest discrete state index
        return int(np.argmin(np.abs(self.snr_bins - snr))) # argmin finds and returns the index  of the smallest value within an array.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start with a random initial power action to get the first state
        random_power = self.power_levels[self.action_space.sample()]
        initial_snr = self._get_snr(random_power)
        self.state = self._get_state(initial_snr)
        return self.state, {}

    def step(self, action):
        power = self.power_levels[action]
        snr = self._get_snr(power)
        next_state = self._get_state(snr)
        
        # 4. Reward Calculation
        # Penalty for missing the target SNR
        snr_penalty = -abs(snr - self.target_snr)
        # Penalty for using power (normalized to keep weights balanced)
        power_penalty = -(power / self.power_levels[-1]) 
        
        reward = snr_penalty + 0.5 * power_penalty
        
        # This is a continuous tracking task, so it doesn't naturally "end"
        terminated = False 
        truncated = False
        
        return next_state, reward, terminated, truncated, {}

# --- Training the Agent using Q-Learning ---

env = WirelessPowerEnv()

# Initialize Q-table with zeros (States x Actions)
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.1    # Learning rate
gamma = 0.9    # Discount factor
epsilon = 0.1  # Exploration rate

# Simple training loop over 1000 steps
state, _ = env.reset()
for _ in range(1000):
    # Epsilon-greedy action selection
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample() # Explore
    else:
        action = np.argmax(q_table[state]) # Exploit
        
    next_state, reward, _, _, _ = env.step(action)
    
    # Q-table update formula
    old_value = q_table[state, action]
    next_max = np.max(q_table[next_state])
    
    q_table[state, action] = old_value + alpha * (reward + gamma * next_max - old_value)
    state = next_state

print("Training finished! Final optimized Q-Table:")
print(q_table)