D_possible=[1e2, 4e2, 1e3, 4e3, 1e4, 4e4, 1e5]
from evolution_functions import evolution_velocity
import numpy as np
import matplotlib.pyplot as plt

N=10000000
k_on=670 #1/seconds
k_off=130 #1/seconds
delta_t=0.000001 #seconds

np.random.seed(54)   #set seed for reproducibility

for i, D_on in enumerate(D_possible):
    for j, D_off in enumerate(D_possible):
        velocities=np.zeros(100)
        for i in range(100):
            velocities[i]=evolution_velocity(N, D_on, D_off, k_on, k_off, delta_t) #evolve for 10^7 time steps