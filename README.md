# Wireless-communications-Internship-IPT
6G Wireless Systems & Deep Reinforcement Learning (DRL)
Project Overview
This repository contains the simulations, custom environments, and final research project developed during a 10-week internship focused on next-generation 6G wireless networks. The curriculum bridges the gap between fundamental wireless communication concepts (path loss, fading, link budgets) and cutting-edge artificial intelligence. The primary goal of this repository is to demonstrate how Deep Reinforcement Learning (DRL) can dynamically optimize complex wireless environments, including Reconfigurable Intelligent Surfaces (RIS) and Non-Terrestrial Networks (NTNs).

Key Learning Modules & Progression
The repository is structured around the progressive milestones of the internship:

Wireless Foundations (Weeks 1–4): Python simulations modeling free-space propagation, log-normal shadowing, outage probability, and multipath fading (Rayleigh/Rician). Includes BER vs. SNR comparisons for BPSK/QPSK over AWGN channels.

Tabular Q-Learning (Week 5): Implementation of a custom Gymnasium environment for transmit power control. The RL agent uses a discrete Q-table to maintain a target Signal-to-Noise Ratio (SNR) of 12 dB while minimizing unnecessary power consumption.

Deep Reinforcement Learning (Weeks 6–7): Transitioning from tabular methods to a Deep Q-Network (DQN) to handle continuous, multi-variable states (Distance, Channel Gain, and SNR). Includes a comparison of the DRL agent against random and heuristic baseline models.

Advanced 6G Scenarios (Weeks 8–10): Integration of RIS-assisted links (phase shift optimization) and NTN topologies (Satellite/UAV links) into the DRL simulator, culminating in a research-style resource allocation project with performance plots and reward curves.
