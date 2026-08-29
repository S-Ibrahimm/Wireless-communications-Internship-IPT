import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ── Import from  existing simulator ──
from week2simulator import distances, results, SENS

# ── Shadowing Parameter ─────────────────────────────
SIGMA = 8  # dB, urban shadowing 

# ── Add Shadowing + Outage to Each Frequency ────────
colors = {"2.4 GHz": "blue", "3.5 GHz": "orange", "28 GHz": "red"}

outage_results = {}

for name, data in results.items():
    
    # Margin at each distance (already computed in simulator.py)
    margin = np.array(data["margin"])
    
    # Outage probability = Q(margin / sigma)
    p_out = 1 - norm.cdf(margin / SIGMA)
    
    # Received power with random shadowing added
    shadow = np.random.normal(0, SIGMA, len(distances))
    prx_shadow = np.array(data["rx_power"]) - shadow

    outage_results[name] = {
        "outage_prob":      p_out,
        "rx_power_shadow":  prx_shadow
    }

# ── Plot 1: Received Power With and Without Shadowing ─
fig1, ax1 = plt.subplots(figsize=(9, 5))

for name in results:
    ax1.semilogx(distances, results[name]["rx_power"],
                 label=f"{name}", color=colors[name])
    ax1.semilogx(distances, outage_results[name]["rx_power_shadow"],
                 label=f"{name} + shadowing",
                 color=colors[name], linestyle='--', alpha=0.5)

ax1.axhline(y=SENS, color='black', linestyle='--', label='Sensitivity (-90 dBm)')
ax1.set_title("Received Power vs Distance (With and Without Shadowing)")
ax1.set_xlabel("Distance (m)")
ax1.set_ylabel("Received Power (dBm)")
ax1.legend()
ax1.grid(True)
plt.tight_layout()
plt.savefig("plot_shadowing.png", dpi=150)
plt.show()

# ── Plot 2: Outage Probability vs Distance ────────────
fig2, ax2 = plt.subplots(figsize=(9, 5))

for name in results:
    ax2.semilogx(distances, outage_results[name]["outage_prob"],
                 label=name, color=colors[name])

ax2.axhline(y=0.1, color='black', linestyle='--', label='10% Outage Threshold')
ax2.set_title("Outage Probability vs Distance")
ax2.set_xlabel("Distance (m)")
ax2.set_ylabel("Outage Probability")
ax2.legend()
ax2.grid(True)
plt.tight_layout()
plt.savefig("plot_outage.png", dpi=150)
plt.show()

# ── Print Outage Summary ──────────────────────────────
print("\n===== OUTAGE PROBABILITY SUMMARY =====")
print(f"Shadowing std deviation (sigma) = {SIGMA} dB\n")

for name in results:
    p = outage_results[name]["outage_prob"]
    # Find distance where outage crosses 10%
    for i, val in enumerate(p):
        if val > 0.10:
            print(f"{name}: Outage exceeds 10% beyond {distances[i]/1000:.2f} km")
            break