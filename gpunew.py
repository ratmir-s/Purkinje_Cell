import random
import numpy as np
import matplotlib.pyplot as plt
import statistics
from numba import cuda
import math

# ---------------------------
# Parameters and Initialization
# ---------------------------
alpha = 0.52
beta = 1
I = 49
epsilon = 0.008
iterations = 8000000  # Total simulation steps (must be a multiple of batch_size)
dt = 0.001
sets = 7               # Number of different granule cell activity sets
delta = 0.001          # (unused here, but kept for consistency)
T_s = 0.125            # Slow time constant
T_f = 0.025            # Fast time constant
N = 1000               # Number of neurons (synapses) to simulate

batch_size = 100       # Number of simulation iterations per kernel launch
num_batches = iterations // batch_size

# Create granule cell activities for 7 sets (each set is a vector of length N)
granule_cell_activities = []
for _ in range(sets):
    granule_cell_activities.append([random.uniform(0, math.sqrt(2)) for _ in range(N)])
granule_cell_activities = np.array(granule_cell_activities, dtype=np.float32)

# Initialize synaptic weights as a vector of length N
weights = np.random.uniform(0, math.sqrt(2), size=(N)).astype(np.float32)

print("Initial weights (sample):", weights[:10])
print("Mean weight:", statistics.mean(weights))
print("Granule cell activities shape:", granule_cell_activities.shape)

# (Optional) Plot a sample of granule cell activities (first 50 neurons for visualization)
plt.figure(figsize=(10, 6))
for i in range(sets):
    plt.plot(granule_cell_activities[i][:50], marker='o', linestyle='-', label=f'Set {i}')
plt.title("Granule Cell Activities (first 50 neurons)")
plt.legend()
plt.show()


# ---------------------------
# Helper Functions
# ---------------------------
def compute_u(alpha, beta, g, sigma):
    n = g.size
    return np.sum(g * sigma) / n

def compute_integral(alpha, beta, g, sigma, dt, iterations):
    n = g.size
    Pu = np.sum(g * sigma) / n
    return (alpha + beta * Pu) * dt * iterations

# ---------------------------
# Batched GPU Kernel
# ---------------------------
@cuda.jit
def batched_update_kernel(g, E_s, E_star, E, weights, dt, T_s, T_f, epsilon, batch_size, t_start, I):
    i = cuda.grid(1)
    if i < g.size:
        # Local time for each thread starting from t_start
        t_local = t_start
        for b in range(batch_size):
            if t_local > 0.02:
                chi_val = 1
            else:
                chi_val = -I
            # Update eligibility traces and weight for neuron i
            E_star[i] = E_star[i] * (1 - 2 * dt / T_f) + g[i]
            E_s[i] = E_s[i] * (1 - dt / T_s - dt / T_f)
            E[i] = E_s[i] + E_star[i] - 2 * E_s[i]
            update_val = epsilon * chi_val * E[i] * dt
            temp = weights[i] + update_val
            if temp >= 0:
                weights[i] = temp
            t_local += dt

# ---------------------------
# Setup Device Arrays and Launch Parameters
# ---------------------------
weights_device = cuda.to_device(weights)
E_s_device = cuda.device_array_like(weights_device)
E_star_device = cuda.device_array_like(weights_device)
E_device = cuda.device_array_like(weights_device)

# Initialize eligibility traces to zero
E_s_device.copy_to_device(np.zeros_like(weights))
E_star_device.copy_to_device(np.zeros_like(weights))
E_device.copy_to_device(np.zeros_like(weights))

threads_per_block = 256
blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

# ---------------------------
# Simulation Loop (Batched)
# ---------------------------
integral_value = 0.0
firing_times = [0.0]        # List of firing times
firing_intervals = []       # List of differences between consecutive firing times
t = 0.0
cs = 0                    # index for granule cell activity sets
g = granule_cell_activities[cs]
sigma = weights.copy()    # Host copy of weights

Pi_history = []  # To track Purkinje cell output over batches




for batch in range(num_batches):
    # Update time since last firing
    t_phase = t - firing_times[-1]
    t += dt * batch_size  # Advance time for this batch

    # Cycle through granule cell activity sets
    cs = batch % sets
    g = granule_cell_activities[cs]

    # Use the current t_phase as starting local time for this batch
    t_start = t_phase

    # Copy current granule cell activity to device
    g_device = cuda.to_device(g)

    # Launch batched kernel
    batched_update_kernel[blocks_per_grid, threads_per_block](
        g_device, E_s_device, E_star_device, E_device, weights_device,
        dt, T_s, T_f, epsilon, batch_size, t_start, I
    )

    # After batch: update sigma from device
    sigma = weights_device.copy_to_host()
    u_step = compute_u(alpha, beta, g, sigma)
    integral_step = compute_integral(alpha, beta, g, sigma, dt, batch_size)
    integral_value += integral_step
    Pi_history.append(u_step)

    # Firing mechanism: if integral crosses threshold, record firing time
    if integral_value >= 1:
        current_firing = t
        firing_times.append(current_firing)
        # Compute difference from the previous firing time and append to intervals list
        if len(firing_times) >= 2:
            firing_intervals.append(current_firing - firing_times[-2])
        print(f"System fired at t = {t:.4f}")
        integral_value = 0

final_weights = weights_device.copy_to_host()
print("Final weights (sample):", final_weights[:10])
print("Firing times:", firing_times, "Count:", len(firing_times))
print("Firing intervals:", firing_intervals, "Count:", len(firing_intervals))

# Plot Purkinje cell output over batches
plt.figure(figsize=(10, 6))
plt.plot(Pi_history, label="Purkinje Cell Output (R)")
plt.xlabel("Batch")
plt.ylabel("Output")
plt.title("Purkinje Cell Output Over Batches")
plt.legend()
plt.ylim(bottom=0)
plt.show()

# --- New Code for Poincaré Plot with Annotations ---
# Convert firing_intervals to a NumPy array


# Convert firing_intervals to a NumPy array
intervals = np.array(firing_intervals)

# Poincaré Scatter Plot (Current vs. Next Interval)
x = intervals[:-1]  # current intervals
y = intervals[1:]   # next intervals

plt.figure(figsize=(12, 8))
plt.scatter(x, y, s=100, color='blue', alpha=0.5)
plt.xlabel("Current Interval")
plt.ylabel("Next Interval")
plt.title("Poincaré Plot of Interspike Intervals")
plt.show()

# Density Plot using Hexbin (shows point density without individual annotations)
plt.figure(figsize=(12, 8))
plt.hexbin(x, y, gridsize=50, cmap='Blues')
plt.colorbar(label='Count')
plt.xlabel("Current Interval")
plt.ylabel("Next Interval")
plt.title("Hexbin Density Plot of Interspike Intervals")
plt.show()

# Histogram of the Interspike Intervals
plt.figure(figsize=(10, 6))
plt.hist(intervals, bins=50, color='green', alpha=0.7)
plt.xlabel("Interspike Interval")
plt.ylabel("Frequency")
plt.title("Histogram of Firing Intervals")
plt.show()




# --- Local‐baseline deviation for a sub‐window of spikes ---

# --- Region-based deviation plot (n=5000…15000) ---

import numpy as np
import matplotlib.pyplot as plt

# 1) Full array of spike times
ft = np.array(firing_times, dtype=float)

# 2) Define region [n1,n2]
n1, n2 = 5000, 7000

# 3) Compute region size and timespan
N_reg = n2 - n1
T_reg = ft[n2] - ft[n1]

# 4) Build the n and t arrays for that region
n_reg = np.arange(n1, n2+1)
t_reg = ft[n1:n2+1]

# 5) Compute deviation Δ(n) = t(n) - (T_reg/N_reg)*n
delta_reg = t_reg - (T_reg / N_reg) * n_reg
plt.figure(figsize=(10,6))
plt.plot(n_reg, delta_reg, marker='o', linestyle='-')
plt.axhline( color='gray', linewidth=0.7)
plt.xlabel("Impulse index n")
plt.ylabel(r"$t(n)\;-\;\dfrac{T_{\rm reg}}{N_{\rm reg}}\,n$")
plt.title(f"Deviation from uniform spacing (n={n1}…{n2})")
plt.show()

# —– FFT of the deviation —–
import numpy as np
import matplotlib.pyplot as plt

# Number of points
Nfft = delta_reg.size

# 1) Remove DC offset
signal = delta_reg - np.mean(delta_reg)

# 2) Compute FFT
Y = np.fft.fft(signal)

# 3) Build frequency axis (in cycles per index)
f = np.fft.fftfreq(Nfft, d=1.0)

# 4) Mask to positive frequencies, excluding zero
mask = (f > 0)

# 5) Plot magnitude spectrum on log scale
plt.figure(figsize=(10,6))
plt.plot(f[mask], np.abs(Y[mask]), linestyle='-', marker='o', markersize=4)
plt.yscale('log')
plt.xlabel("Frequency (cycles per index)")
plt.ylabel("Magnitude (log scale)")
plt.title("FFT of Deviation (zero‐mean, f>0)")
plt.xlim(0, max(f[mask]))
plt.show()


