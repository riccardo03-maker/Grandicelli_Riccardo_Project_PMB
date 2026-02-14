from evolution_functions import evolution_velocity, gaussian
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

N=10000000
k_on=670 #1/seconds
k_off=130 #1/seconds
delta_t=0.000001 #seconds
D_on=40000 #nanometer^2/second
D_off=400 #nanometer^2/second


np.random.seed(54)   #set seed for reproducibility
velocities=np.zeros(100)
for i in range(100):
  velocities[i]=evolution_velocity(N, D_on, D_off, k_on, k_off, delta_t) #evolve for 10^7 time steps

n, bins, _ =plt.hist(velocities, bins=80, density=True) #histogram of velocities
centers = 0.5 * (bins[:-1] + bins[1:]) #take the central x for each bin

initial_guess=[n.max(), np.mean(velocities), np.std(velocities)] #initial guess of the fit parameters

amp, mu, sigma=curve_fit(gaussian, centers, n, p0=initial_guess)[0] #fit parameters

plt.plot(centers, gaussian(centers, amp, mu, sigma))
plt.title("Histogram of velocities with two-state system")
plt.xlabel("Velocity (nm/s)")
plt.ylabel("Count")
plt.show()

print("Mean velocity: " + str(np.mean(velocities)))
print("Standard deviation: " + str(np.std(velocities)))