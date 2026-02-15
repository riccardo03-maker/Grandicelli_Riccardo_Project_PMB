from evolution_functions import evolution_velocity
import numpy as np
import matplotlib.pyplot as plt

N=10000000
D_on=40000 #1/seconds
D_off=400 #1/seconds
delta_t=0.000001 #seconds
k_possible=np.logspace(1, 4, 10)

velocity_matrix=np.zeros((10, 10)) #matrix with the mean values of velocities for different diffusion coefficients
errors_matrix=np.zeros((10,10)) #matrix with the standard deviations of the velocities

for i, k_on in enumerate(k_possible):
    for j, k_off in enumerate(k_possible):
        velocities=np.zeros(2)
        for k in range(2):
            velocities[k]=evolution_velocity(N, D_on, D_off, k_on, k_off, delta_t) #evolve for 10^7 time steps
        velocity_matrix[i, j]=np.mean(velocities)
        errors_matrix[i, j]=np.std(velocities)

plt.imshow(velocity_matrix, origin="lower", extent=[k_possible[0], k_possible[9], k_possible[0], k_possible[9]], aspect="auto")
#creates a heatmap of velocities versus k_on and k_off

plt.xscale("log")
plt.yscale("log")
plt.colorbar(label="Velocity (nm/s)")
plt.xlabel("k_off (1/s)")
plt.ylabel("k_on (1/s)")
plt.title("Velocity in function of the rate constants")
plt.show()