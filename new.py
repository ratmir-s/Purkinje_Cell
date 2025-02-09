import random
import numpy as np
import matplotlib.pyplot as plt
import statistics
import pandas as pd
from numba import jit, cuda
import math


#alpha = []
# Example value for alpha, can be adjusted
alpha = 0.5
beta = 0.1    # Example value for beta, can be adjusted
I = 50       # Positive constant for the χ function
epsilon = 0.000066 # Small positive constant parameter
iterations = 100000 # Number of iterations
dt = 0.001    # Time step
sets = 7
#dt_list = [0.01,0.001,0.0001]
avg = []
weights = []
granule_cell_activities = []


# Granule cell activities: periodic vector function

D = 10  # Periodicity as per the images
for i in range(sets):
    granule_cell_activities.append([random.uniform(0, 1) for _ in range(10)])  # Uniform distribution [0, 1]

# Convert to NumPy array
granule_cell_activities = np.array(granule_cell_activities, dtype=np.float32)

# Synaptic weights: uniform distribution over [0, sqrt(2)]
weights = np.random.uniform(0, math.sqrt(2), size=(10)).astype(np.float32)


#for i in range(sets):  # 7 sets
    # Random weights
    #weights.append([random.random() for _ in range(10)])  # Random weights
    #granule_cell_activities.append([random.random() for _ in range(10)])  # Random activities
    #alpha.append(random.uniform(0, 0.4))

#alpha = [x+(0.05) for x in alpha]
'''print(alpha)
for i in range(sets):
    granule_cell_activities.append([random.uniform(0, 1) for _ in range(10)])  # List of lists
weights.append([0.5] * 10)  # Shared weights for all sets

# Convert lists to NumPy arrays
granule_cell_activities = np.array(granule_cell_activities, dtype=np.float32)  # Shape: (sets, 10)
weights = np.array(weights[0], dtype=np.float32)'''

print(weights)
print(granule_cell_activities)
colors = ['red','blue','green','yellow','orange','black','grey']

for i in range(7):
    #for j in range(10):
    #y_g = granule_cell_activities[i].append(alpha)
    plt.plot(granule_cell_activities[i],'ro',color=colors[i],linestyle='-')
    plt.plot()
    print(granule_cell_activities[i])
plt.show()

#define Chi of t

def chi(t):
    if t > 0.02:
        return 1
    else:
        return -I

def compute_integral(alpha, beta, g, sigma, dt, t):
    # Compute Purkinje cell output: Pu(t) = 1/n * sum(g_i * sigma_i)
    n = len(g)
    Pu_t = np.sum(g * sigma) / n  # Element-wise multiplication and average

    # Integral increment
    integral = (alpha + beta * Pu_t) * dt
    return integral

def compute_u(alpha, beta, g, sigma, dt, t):
    R = sum(g_i * sigma_i for g_i, sigma_i in zip(g, sigma))  # PC output R(t)
    #integral = (alpha + beta * R) * dt  # Incremental change in integral
    return R


def update_eligibility_traces(E_s, E_star, g, delta, T_s, T_f):
    # Update E_i^* and E_i^s based on the provided equations
    E_star_new = [E_star[i] * (1 - 2 * delta / T_f) + g[i] for i in range(len(E_star))]
    E_s_new = [E_s[i] * (1 - delta / T_s - delta / T_f) for i in range(len(E_s))]
    E_new = [E_s_new[i] + E_star_new[i] - 2 * E_s[i] for i in range(len(E_s))]
    return E_new, E_star_new, E_s_new

def it_val_g(int_start,int_end,t):
    int_range = int((int_end - int_start) / dt)
    if t>=(int_end-0.001) and t<=int_end:
        s = pd.Series(is_val[-int_range:])
        x = s * 1000
        plt.plot(t_val[-int_range:],x,linestyle='-')
        #for value in firing_times:
            #if value in t_val[-int_range:]:
                #plt.axvline(value,color='orange')
        print(statistics.mean(is_val[-int_range:]))
        #plt.ylim(0,0.00000542)
        plt.xlabel('Time (s)')
        plt.ylabel('Integral Step Value at T')
        plt.ylim(0,1.2)

def it_val_gt(int_start,int_end,t,a):
    int_range = int((int_end - int_start) / dt)
    if t>=(int_end-0.001) and t<=int_end:
        plt.plot(t_val[-int_range:],iu_val[-int_range:],linestyle='-',color='green')
        #for value in firing_times:
            #if value in t_val[-int_range:]:
                #plt.axvline(value,color='orange')
def e_g(t,strt,end):
    if t>=strt and t<=end:
        e_list.append([(t+random.uniform(1, 201)) for _ in range(10)])
        plt.plot(t_val,ef)
        plt.xlabel('Time (s)')
        plt.ylabel('e')
        plt.ylim(bottom=0)

def firing_g(activation):
    if activation > 2:
        plt.plot((firing_times[activation - 1] - firing_times[activation - 2]), (t - firing_times[activation - 1]),
                 'ro')
        plt.ylabel('Next firing time')
        plt.xlabel('Firing time')


def update_traces_and_weights(g, E_s, E_star, E, weights, chi_val, dt, T_s, T_f, epsilon):
    i = cuda.grid(1)
    if i < g.size:
        # Update E_i^* (fast trace)
        E_star[i] = E_star[i] * (1 - 2 * dt / T_f) + g[i]

        # Update E_i^s (slow trace)
        E_s[i] = E_s[i] * (1 - dt / T_s - dt / T_f)

        # Update E_i (combined trace)
        E[i] = E_s[i] + E_star[i] - 2 * E_s[i]

        # Update weights using eligibility traces
        weights[i] += epsilon * chi_val * E[i] * dt



def scalar_prod_g():
    pass

def int_time_g():
    pass

#main loop

# Convert arrays to GPU-compatible format
weights_device = cuda.to_device(np.array(weights[0], dtype=np.float32))
granule_cell_activities_device = cuda.to_device(np.array(granule_cell_activities[0], dtype=np.float32))

# Allocate space for eligibility traces on GPU
E_s_device = cuda.device_array_like(weights_device)
E_star_device = cuda.device_array_like(weights_device)
E_device = cuda.device_array_like(weights_device)


integral_value = 0
firing_times = [0]
iv_list = []
t_val = []
i_val = []
is_val = []
iu_val = []
t10_val = []
t = 0
activation = 0
cs = 0
g = granule_cell_activities[cs]  # Switch sets every 100 iterations
sigma = weights[cs]
t0 = 0
e = granule_cell_activities[0]
e1 = granule_cell_activities[0]
ef = []
e_list = []
# Time constants and delta
delta = 0.001  # 1ms time step
T_s = 0.125    # Slow time constant
T_f = 0.025    # Fast time constant

# Initialize eligibility traces
E_star = [0.0] * 10  # E_i^*
E_s = [0.0] * 10     # E_i^s
E = [0.0] * 10       # E_i

threads_per_block = 256
blocks_per_grid = (weights_device.size + (threads_per_block - 1)) // threads_per_block

# Initialize history lists for neuron 0
E_history = []        # Combined trace E for neuron 0
E_star_history = []   # Fast trace E* for neuron 0
E_s_history = []      # Slow trace Es for neuron 0
g_history = []        # Granule cell activity g for neuron 0
weight_history = []
weight_history1 = []
weight_history2 = []
weight_history3 = []
weight_history4 = []
weight_history5 = []
weight_history6 = []
weight_history7 = []
weight_history8 = []
weight_history9 = []
weight_history10 = []# Weight for neuron 0
Pi_history = []       # Purkinje cell output (R)

for iteration in range(iterations):
    t_phase = t - firing_times[-1]  # Time since last firing
    t += dt
    t0 += dt
    if t0 >= 1:
        t0 = 0

    # Switch granule cell activities and weights every 100 iterations
    if iteration % 100 == 0:
        cs = int(iteration / 100)
        g = granule_cell_activities[cs % 7]
        sigma = weights
        a = 0.5  # Keep alpha constant for now

    # Update eligibility traces at each time step
    E, E_star, E_s = update_eligibility_traces(E_s, E_star, g, delta, T_s, T_f)

    # Track eligibility traces for neuron 0
    E_history.append(E[0])        # Combined trace E for neuron 0
    E_star_history.append(E_star[0])  # Fast trace E* for neuron 0
    E_s_history.append(E_s[0])    # Slow trace Es for neuron 0

    # Track granule cell activity for neuron 0
    g_history.append(g[0])

    # Calculate integral and output (R)
    integral_step = compute_integral(a, beta, g, sigma, dt, t)
    u_step = compute_u(a, beta, g, sigma, dt, t)

    # Track Purkinje cell output
    Pi_history.append(u_step)

    # Track values for plotting and debugging
    is_val.append(integral_step)  # Integral steps
    iu_val.append(u_step)  # Output values
    integral_value += integral_step  # Update integral value
    t_val.append(t)  # Track time
    i_val.append(integral_value)  # Track integral values

    # Firing mechanism (check if integral exceeds threshold)
    if integral_value >= 1:
        firing_times.append(t)  # Log firing time
        print(f"System fired at t={t:.4f}")
        iv_list.append(integral_value)
        integral_value = 0  # Reset integral
        activation += 1

    # Update weights using eligibility traces
    for i in range(len(sigma)):
        sigma[i] += epsilon * chi(t_phase) * E[i] * dt
    weights = sigma  # Update weights for the current set

    # Track weight of neuron 0 for plotting
    weight_history.append(sigma[0])
    weight_history1.append(sigma[1])
    weight_history2.append(sigma[2])
    weight_history3.append(sigma[3])
    weight_history4.append(sigma[4])
    weight_history5.append(sigma[5])
    weight_history6.append(sigma[6])
    weight_history7.append(sigma[7])
    weight_history8.append(sigma[8])
    weight_history9.append(sigma[9])
    #Update eligibility trace averages (optional for debugging)
    ef.append([((e[l] + e1[l]) / 2) for l in range(10)])

    # Optional visualization function calls (e.g., integral plotting)

    it_val_g(0, 1000, t)
    it_val_g(0, 1, t)









    #every 1000 iterations randomize alpha, beta, and epsilon
    #every 1000 iterations swap dt between 0.1,0.01, and 0.001
    '''if iteration % 1000 == 0 and iteration > 0:
        print(iteration)
        alpha,beta,epsilon = [random.random() for _ in range(3)]
    #if iteration % 1000 == 0 and iteration > 0:
        #dt = dt_list[iteration % 3]'''


plt.show()
print("Firing times:", firing_times, len(firing_times))
print(activation)
plt.show()
#for i in range(10):
    #e_g(i)
plt.figure(figsize=(10, 6))
plt.ylim(bottom=0)
plt.plot(E_history, label="E_i (combined trace)")
#plt.plot(E_star_history, label="E_i^* (fast trace)")
#plt.plot(E_s_history, label="E_i^s (slow trace)")
'''plt.xlabel("Iterations")
plt.ylabel("Eligibility Traces")
plt.title("Eligibility Traces Over Time for Neuron 0")
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(g_history, label="g_i (granule cell activity)")
plt.xlabel("Iterations")
plt.ylabel("Activity")
plt.title("Granule Cell Activity Over Time for Neuron 0")
plt.legend()
plt.show()'''

plt.figure(figsize=(10, 6))
plt.plot(weight_history, label="Synaptic Weight for Neuron 0")
plt.plot(weight_history1, label="Synaptic Weight for Neuron 1")
plt.plot(weight_history2, label="Synaptic Weight for Neuron 2")
plt.plot(weight_history3, label="Synaptic Weight for Neuron 3")
plt.plot(weight_history4, label="Synaptic Weight for Neuron 4")
plt.plot(weight_history5, label="Synaptic Weight for Neuron 5")
plt.plot(weight_history6, label="Synaptic Weight for Neuron 5")
plt.plot(weight_history7, label="Synaptic Weight for Neuron 5")
plt.plot(weight_history8, label="Synaptic Weight for Neuron 5")
plt.plot(weight_history9, label="Synaptic Weight for Neuron 5")
plt.plot(weight_history10, label="Synaptic Weight for Neuron 5")
plt.xlabel("Iterations")
plt.ylabel("Weight")
plt.title("Synaptic Weight Changes Over Time")
plt.legend()
plt.ylim(bottom=0)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(Pi_history, label="Purkinje Cell Output (R)")
plt.xlabel("Iterations")
plt.ylabel("Output")
plt.title("Purkinje Cell Output Over Time")
plt.legend()
plt.ylim(bottom=0)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(Pi_history[-100000], label="Purkinje Cell Output (R)")
plt.xlabel("Iterations")
plt.ylabel("Output")
plt.title("Purkinje Cell Output Over Time")
plt.legend()
plt.ylim(bottom=0)
plt.show()

'''plt.show()
plt.plot(t_val[0:100000],i_val[0:100000])
plt.show()'''
