import numpy as np
import matplotlib.pyplot as plt

# ─── FUNCTIONS ───────────────────────────────────────────

def fspl_db(distance_m, freq_hz):
    """Free Space Path Loss in dB"""
    c = 3e8
    return 20*np.log10(distance_m) + 20*np.log10(freq_hz) + 20*np.log10(4*np.pi/c)

def received_power_dbm(p_tx, g_tx, g_rx, fspl, losses):
    return p_tx + g_tx + g_rx - fspl - losses

def link_margin(p_rx, sensitivity):
    return p_rx - sensitivity

def coverage_distance(distances, margins):
    for i, m in enumerate(margins):
        if m < 0:
            return distances[i]
    return distances[-1]

# ─── SYSTEM PARAMETERS ───────────────────────────────────

P_TX   = 30    # dBm
G_TX   = 0     # dBi
G_RX   = 0     # dBi
LOSSES = 2     # dB
SENS   = -90   # dBm receiver sensitivity

freqs = {
    "2.4 GHz": 2.4e9,
    "3.5 GHz": 3.5e9,
    "28 GHz":  28e9
}

distances = np.logspace(1, 4, 500)  # 10m to 10km, log scale

# ─── SIMULATION ──────────────────────────────────────────

results = {}

for name, freq in freqs.items():
    pl   = [fspl_db(d, freq) for d in distances]
    prx  = [received_power_dbm(P_TX, G_TX, G_RX, p, LOSSES) for p in pl]
    marg = [link_margin(p, SENS) for p in prx]
    cov  = coverage_distance(distances, marg)

    results[name] = {
        "path_loss": pl,
        "rx_power":  prx,
        "margin":    marg,
        "coverage":  cov
    }

# ─── PLOTS ───────────────────────────────────────────────

colors = {"2.4 GHz": "blue", "3.5 GHz": "orange", "28 GHz": "red"}
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Path Loss vs Distance
for name, data in results.items():
    axes[0].semilogx(distances, data["path_loss"], label=name, color=colors[name])
axes[0].set_title("Path Loss vs Distance")
axes[0].set_xlabel("Distance (m)")
axes[0].set_ylabel("Path Loss (dB)")
axes[0].legend()
axes[0].grid(True)

# Plot 2: Received Power vs Distance
for name, data in results.items():
    axes[1].semilogx(distances, data["rx_power"], label=name, color=colors[name])
axes[1].axhline(y=-90, color='black', linestyle='--', label='Sensitivity (-90 dBm)')
axes[1].set_title("Received Power vs Distance")
axes[1].set_xlabel("Distance (m)")
axes[1].set_ylabel("Received Power (dBm)")
axes[1].legend()
axes[1].grid(True)

# Plot 3: Coverage Distance Bar Chart
names = list(results.keys())
covs  = [results[n]["coverage"] / 1000 for n in names]
bars  = axes[2].bar(names, covs, color=[colors[n] for n in names])
axes[2].bar_label(bars, fmt="%.2f km")
axes[2].set_title("Coverage Distance Comparison")
axes[2].set_ylabel("Coverage Distance (km)")
axes[2].grid(True, axis='y')

plt.tight_layout()
plt.savefig("link_budget_simulation.png", dpi=150)
plt.show()

# ─── PRINT SUMMARY ───────────────────────────────────────

for name, data in results.items():
    print(f"{name}: Coverage = {data['coverage']/1000:.2f} km")