import random
import numpy as np
import matplotlib.pyplot as plt

alpha = 0.5  # Example value for alpha, can be adjusted
beta = 0.1    # Example value for beta, can be adjusted
I = 1.0       # Positive constant for the χ function
epsilon = 0.01  # Small positive constant parameter
iterations = 1000000 # Number of iterations
dt = 0.001    # Time step
sets = 7
#dt_list = [0.01,0.001,0.0001]

weights = []
granule_cell_activities = []



for i in range(sets):  # 7 sets
    weights.append([random.random() for _ in range(10)])  # Random weights
    granule_cell_activities.append([random.random() for _ in range(10)])  # Random activities

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
    return 1 if t > 0 else -I

def compute_integral(alpha, beta, g, sigma, dt, t):
    R = sum(g_i * sigma_i for g_i, sigma_i in zip(g, sigma))  # PC output R(t)
    integral = (alpha + beta * R) * dt  # Incremental change in integral
    return integral

def it_val_g(int_start,int_end,t):
    int_range = int((int_end - int_start) * 1000)
    if t>=(int_end-0.001) and t<=int_end:
        plt.plot(t_val[-int_range:],is_val[-int_range:],linestyle='-')
        for value in firing_times:
            if value in t_val[-int_range:]:
                plt.axvline(value,color='orange')
        plt.xlabel('Time (s)')
        plt.ylabel('Integral Step Value at T')


def firing_g(activation):
    if activation > 2:
        plt.plot((firing_times[activation - 1] - firing_times[activation - 2]), (t - firing_times[activation - 1]),
                 'ro')
        plt.ylabel('Next firing time')
        plt.xlabel('Firing time')


#main loop

integral_value = 0
firing_times = []
iv_list = []
t_val = []
i_val = []
is_val = []
t10_val = []
t = 0
activation = 0
cs = 0
g = granule_cell_activities[cs]  # Switch sets every 100 iterations
sigma = weights[cs]



for iteration in range(iterations):
    t = t+dt

    if iteration % 100 == 0:
        cs = int(iteration/100)
        g = granule_cell_activities[cs % 7]  # Switch sets every 100 iterations
        sigma = weights[cs % 7]
        # Switch weights every 100 iterations
    # Calculate integral at each time step]
    integral_step = compute_integral(alpha, beta, g, sigma, dt, t)

    is_val.append(integral_step)

    integral_value += integral_step
    t_val.append(t)
    i_val.append(integral_value)

    # Check if the integral value has reached an integer value
    if integral_value >= 1:
        firing_times.append(t)
        print(f"System fired at t={t:.4f}")
        iv_list.append(integral_value)
        #plot firing time (y) to previous firing time (x)
        #firing_g(activation)
        integral_value = 0
        activation +=1
    # Update weights using the differential equation


    for i in range(len(sigma)):
        sigma[i] += epsilon * chi(t) * g[i] * dt  # Update weights based on chi and g
    weights[cs % 7] = sigma

    #graph the i_val over time over a certain interval
    it_val_g(0,1,t)






    #every 1000 iterations randomize alpha, beta, and epsilon
    #every 1000 iterations swap dt between 0.1,0.01, and 0.001
    '''if iteration % 1000 == 0 and iteration > 0:
        print(iteration)
        alpha,beta,epsilon = [random.random() for _ in range(3)]
    #if iteration % 1000 == 0 and iteration > 0:
        #dt = dt_list[iteration % 3]'''



print("Firing times:", firing_times, len(firing_times))
print(activation)
plt.show()

