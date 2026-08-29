import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# ─── Q-FUNCTION ──────────────────────────────────────────
def q_function(x):
    return 0.5 * erfc(x / np.sqrt(2))

# ─── BER FUNCTIONS ───────────────────────────────────────

def bpsk_awgn(snr_linear):
    """BPSK over AWGN channel"""
    return q_function(np.sqrt(2 * snr_linear))

def qpsk_awgn(snr_linear):
    """QPSK over AWGN channel — same as BPSK theoretically"""
    return q_function(np.sqrt(2 * snr_linear))

def bpsk_rayleigh(snr_linear):
    """BPSK over Rayleigh fading channel"""
    return 0.5 * (1 - np.sqrt(snr_linear / (1 + snr_linear)))

def qpsk_rayleigh(snr_linear):
    """QPSK over Rayleigh fading channel"""
    return 0.5 * (1 - np.sqrt(snr_linear / (2 + snr_linear)))

# ─── SNR RANGE ───────────────────────────────────────────

snr_db     = np.linspace(0, 30, 300)       # 0 to 30 dB
snr_linear = 10 ** (snr_db / 10)           # convert to linear

# ─── COMPUTE BER ─────────────────────────────────────────

ber_bpsk_awgn     = bpsk_awgn(snr_linear)
ber_qpsk_awgn     = qpsk_awgn(snr_linear)
ber_bpsk_rayleigh = bpsk_rayleigh(snr_linear)
ber_qpsk_rayleigh = qpsk_rayleigh(snr_linear)

# ─── PLOT: BER vs SNR ────────────────────────────────────

plt.figure(figsize=(9, 6))

plt.semilogy(snr_db, ber_bpsk_awgn,
             label="BPSK - AWGN",
             color="blue", linewidth=2)

plt.semilogy(snr_db, ber_qpsk_awgn,
             label="QPSK - AWGN",
             color="orange", linewidth=2, linestyle='--')

plt.semilogy(snr_db, ber_bpsk_rayleigh,
             label="BPSK - Rayleigh",
             color="red", linewidth=2)

plt.semilogy(snr_db, ber_qpsk_rayleigh,
             label="QPSK - Rayleigh",
             color="green", linewidth=2, linestyle='--')

# Target BER line
plt.axhline(y=1e-3, color='black', linestyle=':', label='Target BER = 1e-3')

plt.title("BER vs SNR — BPSK/QPSK over AWGN and Rayleigh Fading")
plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.legend()
plt.grid(True, which='both')
plt.tight_layout()
plt.savefig("plot_ber_snr.png", dpi=150)
plt.show()

# ─── FIND SNR NEEDED FOR TARGET BER = 1e-3 ───────────────

TARGET_BER = 1e-3

def find_snr_at_ber(ber_array, snr_db_array, target):
    """Find SNR value where BER first crosses target"""
    for i, b in enumerate(ber_array):
        if b <= target:
            return snr_db_array[i]
    return None

snr_bpsk_awgn     = find_snr_at_ber(ber_bpsk_awgn,     snr_db, TARGET_BER)
snr_qpsk_awgn     = find_snr_at_ber(ber_qpsk_awgn,     snr_db, TARGET_BER)
snr_bpsk_rayleigh = find_snr_at_ber(ber_bpsk_rayleigh, snr_db, TARGET_BER)
snr_qpsk_rayleigh = find_snr_at_ber(ber_qpsk_rayleigh, snr_db, TARGET_BER)

# ─── PRINT SUMMARY ───────────────────────────────────────

print("\n===== BER SIMULATION RESULTS =====")
print(f"SNR required to achieve BER = {TARGET_BER}:\n")
print(f"  BPSK over AWGN:     {snr_bpsk_awgn:.1f} dB")
print(f"  QPSK over AWGN:     {snr_qpsk_awgn:.1f} dB")
print(f"  BPSK over Rayleigh: {snr_bpsk_rayleigh:.1f} dB")
print(f"  QPSK over Rayleigh: {snr_qpsk_rayleigh:.1f} dB")

print("\n===== KEY OBSERVATIONS =====")
print(f"  Rayleigh vs AWGN penalty (BPSK): "
      f"{snr_bpsk_rayleigh - snr_bpsk_awgn:.1f} dB extra SNR needed")
print(f"  Rayleigh vs AWGN penalty (QPSK): "
      f"{snr_qpsk_rayleigh - snr_qpsk_awgn:.1f} dB extra SNR needed")