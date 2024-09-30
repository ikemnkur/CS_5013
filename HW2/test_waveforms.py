import numpy as np
import matplotlib.pyplot as plt

# Constants
c = 3e8  # Speed of light in m/s
# fc = 9.73e9  # Central frequency in Hz
fc = 1
pulse_width = 1e-6  # Pulse width in seconds
if_freq = 90e6  # IF frequency in Hz

# Time array
t = np.linspace(0, pulse_width, 1000)

# Bandwidth (example value, adjust as needed)
bandwidth = 20e6  # 20 MHz bandwidth

def constant_chirp(t):
    return np.cos(2 * np.pi * 5000000 * t)

def linear_chirp(t):
    chirp_rate = bandwidth / pulse_width
    return np.cos(2 * np.pi * (fc * t + 0.5 * chirp_rate * t**2))

def exponential_chirp(t):
    k = 5  # Exponential factor, adjust for desired chirp rate
    return np.cos(2 * np.pi * fc * (np.exp(k * t / pulse_width) - 1) / k)

def quadratic_chirp(t):
    return np.cos(2 * np.pi * (fc * t + (bandwidth / (3 * pulse_width**2)) * t**3))

# Generate chirp signals
chirps = {
    "Constant (5MHz)": constant_chirp(t),
    "Linear": linear_chirp(t),
    "Exponential": exponential_chirp(t),
    "Quadratic": quadratic_chirp(t)
}

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle(f"Radar Chirp Signals (fc = {fc/1e9:.2f} GHz, PW = {pulse_width*1e6:.1f} µs, BW = {bandwidth/1e6:.0f} MHz)")

for (name, chirp), ax in zip(chirps.items(), axs.ravel()):
    # Plot the chirp signal
    ax.plot(t * 1e6, chirp)
    ax.set_title(f"{name} Chirp")
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-1.1, 1.1)  # Set y-axis limits to slightly beyond [-1, 1]
    ax.grid(True)

    # Add instantaneous frequency plot
    ax2 = ax.twinx()
    inst_freq = np.diff(np.unwrap(np.angle(chirp))) / (2 * np.pi * (t[1] - t[0])) + fc
    ax2.plot(t[1:] * 1e6, (inst_freq - fc) / 1e6, 'r--', alpha=0.5)
    ax2.set_ylabel('Frequency Offset (MHz)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Set frequency offset y-axis limits based on bandwidth
    ax2.set_ylim(-bandwidth/1e6, bandwidth/1e6)

plt.tight_layout()
plt.show()

# Print a few values of the constant chirp signal
print("First few values of constant chirp signal:")
print(chirps["Constant"][:10])