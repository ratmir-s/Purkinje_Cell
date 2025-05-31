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
I = 50
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

# ★ Save a pristine copy, so we can re-start from identical weights each α-run:
initial_weights = weights.copy()

# ★ Define the α values you want to test:
alpha_values = np.arange(0.5, 0.76, 0.02)   # [0.50,0.55,0.60,0.65,0.70,0.75]


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
# ---------------------------
# α-Sweep Simulation
# ---------------------------
results = {}   # will hold (n_reg, delta) per alpha

for alpha in alpha_values:
    print(f"\n>>> Running α = {alpha:.2f}")

    # 1) reset state
    integral_value = 0.0
    firing_times   = [0.0]
    Pi_history     = []

    # 2) reload initial weights & zero traces on the GPU
    weights_device = cuda.to_device(initial_weights)
    E_s_device     = cuda.device_array_like(weights_device); E_s_device.copy_to_device(np.zeros_like(initial_weights))
    E_star_device  = cuda.device_array_like(weights_device); E_star_device.copy_to_device(np.zeros_like(initial_weights))
    E_device       = cuda.device_array_like(weights_device); E_device.copy_to_device(np.zeros_like(initial_weights))

    # 3) batched simulation exactly as before, but now using this `alpha`
    t = 0.0
    for batch in range(num_batches):
        t_phase = t - firing_times[-1]
        t += dt * batch_size

        cs      = batch % sets
        g       = granule_cell_activities[cs]
        g_dev   = cuda.to_device(g)

        batched_update_kernel[blocks_per_grid, threads_per_block](
            g_dev, E_s_device, E_star_device, E_device, weights_device,
            dt, T_s, T_f, epsilon, batch_size, t_phase, I
        )

        sigma         = weights_device.copy_to_host()
        u_step        = compute_u(alpha, beta, g, sigma)
        integral_step = compute_integral(alpha, beta, g, sigma, dt, batch_size)
        integral_value += integral_step
        Pi_history.append(u_step)

        if integral_value >= 1:
            firing_times.append(t)
            integral_value = 0

    # 4) build local-baseline deviation for the fixed window n1…n2
    ft    = np.array(firing_times, float)
    n1,n2 = 5000, 7000
    n2    = min(n2, ft.size-1)
    N_reg = n2 - n1
    T_reg = ft[n2] - ft[n1]
    n_reg = np.arange(n1, n2+1)
    t_reg = ft[n1:n2+1]
    delta = t_reg - (T_reg/N_reg)*n_reg

    results[alpha] = (n_reg, delta)

# 5) once all α are done, plot them together
plt.figure(figsize=(10,6))
for alpha,(n_reg,delta) in results.items():
    plt.plot(n_reg, delta, label=f"α={alpha:.2f}")
plt.axhline(0, color='gray', lw=0.7)
plt.xlabel("Impulse index n")
plt.ylabel("t(n) - T/N *n")
plt.title("Deviation from uniform spacing for various α")
plt.legend()
plt.show()
