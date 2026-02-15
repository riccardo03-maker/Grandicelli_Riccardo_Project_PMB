from evolution_functions import evolution_velocity
import numpy as np
import matplotlib.pyplot as plt

N=10000000
k_on=670 #1/seconds
k_off=130 #1/seconds
delta_t=0.000001 #seconds
D_possible=np.logspace(2, 5, 10)

velocity_matrix=np.zeros((10, 10)) #matrix with the mean values of velocities for different diffusion coefficients
errors_matrix=np.zeros((10,10)) #matrix with the standard deviations of the velocities

for i, D_on in enumerate(D_possible):
    for j, D_off in enumerate(D_possible):
        velocities=np.zeros(2)
        for k in range(2):
            velocities[k]=evolution_velocity(N, D_on, D_off, k_on, k_off, delta_t) #evolve for 10^7 time steps
        velocity_matrix[i, j]=np.mean(velocities)
        errors_matrix[i, j]=np.std(velocities)

plt.imshow(velocity_matrix, origin="lower", extent=[D_possible[0], D_possible[9], D_possible[0], D_possible[9]], aspect="auto")
#creates a heatmap of velocities versus D_on and D_off

plt.xscale("log")
plt.yscale("log")
plt.colorbar(label="Velocity (nm/s)")
plt.xlabel("D_off (nm^2/s)")
plt.ylabel("D_on (nm^2/s)")
plt.title("Velocity in function of the diffusion coefficients")
plt.show()