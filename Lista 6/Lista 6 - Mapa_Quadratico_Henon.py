#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

def henon_map(x, y, alpha):
    x_next = x * np.cos(alpha) - (y - x**2) * np.sin(alpha)
    y_next = x * np.sin(alpha) + (y - x**2) * np.cos(alpha)
    return x_next, y_next

# Number of iterations
steps = 100000

# Number of initial conditions
num_conditions = 10

# Generate figures for different initial conditions
for _ in range(num_conditions):
    # Generate random initial condition
    x0, y0 = np.random.uniform(-1, 1, size=2)

    # Find alpha that satisfies cos(alpha) = 0.24
    alpha = np.arccos(0.22)

    # Array initialization
    X = np.zeros(steps + 1)
    Y = np.zeros(steps + 1)
    X[0], Y[0] = x0, y0

    # Generate points
    for j in range(steps):
        x_next, y_next = henon_map(X[j], Y[j], alpha)

        # Check if the next point is within the square
        if abs(x_next) <= 1 and abs(y_next) <= 1:
            X[j+1] = x_next
            Y[j+1] = y_next
        else:
            break

    # Plot figure for each initial condition
    #plt.plot(X, Y, '.', color='red', alpha=0.8, markersize=0.3)
    plt.scatter(X, Y, s=0.4)
# Plot settings
plt.title('Hénon Map')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.show()


print(x0, y0)


# In[ ]:




