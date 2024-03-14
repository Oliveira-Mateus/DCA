#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


q = 4  # Valor arbitrário de q
ics = [[1, 0], [3, 0], [4, 0], [5,0], [8,0], [12,0]]  # Condições iniciais
epsilon = 0.0001  # Valor arbitrário de epsilon
w = 2  # Valores de w

sols = []
for ic in ics:
    sol = solve_ivp(lambda t, x: [x[1], -w**2 * x[0] - (3/2) * epsilon * w**2 / q * x[0]], [0, 10], ic, t_eval=np.linspace(0, 10, 200))
    sols.append(sol)

plt.figure(figsize=(10, 8))
for i, sol in enumerate(sols):
    plt.plot(sol.t, sol.y[0], label=f"q(0) = {ics[i][0]}, p(0) = {ics[i][1]}")


plt.legend(loc="upper right")
plt.xlim(0, 5)
plt.xlabel("t")
plt.ylabel("p(t)")
plt.title("Soluções do sistema de equações diferenciais")
plt.grid(True)
plt.savefig("LetraB.png", dpi=600)
plt.show()


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


q = 1  # Valor arbitrário de q
ics = [[1, 0], [1.5, 0], [2, 0],[3.5,0],[4.,0],[4.5,0]]  # Condições iniciais
epsilon = 0.001  # Valor arbitrário de epsilon
w =1  # Valores de w

sols = []
for ic in ics:
    sol = solve_ivp(lambda t, x: [x[1], -w**2 * x[0] - (3/(2*q)) * epsilon * w**2 * (x[0])**2], [0, 40], ic, t_eval=np.linspace(0, 40, 200))
    sols.append(sol)

plt.figure(figsize=(16, 8))
for i, sol in enumerate(sols):
    plt.plot(sol.t, sol.y[0], label=f"x(0) = {ics[i][0]}, x'(0) = {ics[i][1]}")


plt.legend(loc="upper right")
plt.xlim(0, 40)
plt.xlabel("t")
plt.ylabel("p(t)")
plt.title("Soluções do sistema de equações diferenciais")
plt.grid(True)
plt.savefig("LetraB.png", dpi=600)
plt.show()


# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Definição das equações diferenciais
def equations(t, y, w, epsilon, q0):
    q, q_dot = y
    q_double_dot = -w**2 * (q + epsilon * (3/2 * q0) * q**2)
    return [q_dot, q_double_dot]

# Parâmetros e condições iniciais
w = 1.0  # Frequência angular
epsilon = 0.0001  # Valor pequeno de epsilon
q0 = 1.0  # Valor arbitrário de q0
initial_conditions = [[1, 0], [1.5, 0], [2, 0], [3.5, 0], [4, 0], [4.5, 0]]

# Intervalo de tempo para integração
t_start = 0.0
t_end = 20.0
num_points = 100

# Resolvendo o sistema de equações diferenciais para cada condição inicial
for ic in initial_conditions:
    solution = solve_ivp(lambda t, y: equations(t, y, w, epsilon, q0), [t_start, t_end], ic, t_eval=np.linspace(t_start, t_end, num_points))

    # Plotando o gráfico da solução
    t_values = solution.t
    q_values = solution.y[0]
    plt.plot(t_values, q_values)

plt.xlabel('t')
plt.ylabel('q')
plt.title('Solução do sistema de equações diferenciais')
plt.grid(True)
plt.show()


# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Definição das equações diferenciais
def equations(t, y, w, epsilon, q0):
    q, p = y
    q_dot = p
    p_dot = -w**2 * (q + epsilon * (3/2 * q0) * q**2)
    return [q_dot, p_dot]

# Parâmetros e condições iniciais
w = 1.0  # Frequência angular
epsilon = 0.01  # Valor pequeno de epsilon
q0 = 1.0  # Valor arbitrário de q0
initial_conditions = [1.0, 0.0]  # Condições iniciais para q e p

# Intervalo de tempo para integração
t_start = 0.0
t_end = 10.0
num_points = 100

# Resolvendo o sistema de equações diferenciais
solution = solve_ivp(lambda t, y: equations(t, y, w, epsilon, q0), [t_start, t_end], initial_conditions, t_eval=np.linspace(t_start, t_end, num_points))

# Plotando o gráfico da solução
t_values = solution.t
q_values = solution.y[0]

plt.plot(t_values, q_values)
plt.xlabel('t')
plt.ylabel('q')
plt.title('Solução do sistema de equações diferenciais')
plt.grid(True)
plt.show()


# In[6]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Definição das equações diferenciais
def equations(t, y, w, epsilon, q0):
    q, p = y
    q_dot = p
    p_dot = -w**2 * (q + epsilon * (3/2 * q0) * q**2)
    return [q_dot, p_dot]

# Parâmetros e condições iniciais
w = 3.0  # Frequência angular
epsilon = 0.01  # Valor pequeno de epsilon
q0 = 1.0  # Valor arbitrário de q0
initial_conditions = [[1, 0], [1.5, 0], [2, 0], [3.5, 0], [4, 0], [4.5, 0]]

# Intervalo de tempo para integração
t_start = 0.0
t_end = 10.0
num_points = 100

# Resolvendo o sistema de equações diferenciais para cada condição inicial
for ic in initial_conditions:
    solution = solve_ivp(lambda t, y: equations(t, y, w, epsilon, q0), [t_start, t_end], ic, t_eval=np.linspace(t_start, t_end, num_points))

    # Plotando o gráfico da solução
    t_values = solution.t
    q_values = solution.y[0]

    plt.plot(t_values, q_values)

plt.xlabel('t')
plt.ylabel('q')
plt.title('Solução do sistema de equações diferenciais')
plt.grid(True)
plt.show()


# In[7]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Definição das equações diferenciais
def equations(t, y, w, epsilon, q0):
    q, p = y
    q_dot = p
    p_dot = -w**2 * (q + epsilon * (3/2 * q0) * q**2)
    return [q_dot, p_dot]

# Parâmetros e condições iniciais
w_values = [1.6, 1.5, 1.4, 1.3, 1.2, 1.1]  # Valores de frequência angular correspondentes às condições iniciais
epsilon = 0.01  # Valor pequeno de epsilon
q0 = 1.0  # Valor arbitrário de q0
initial_conditions = [[1, 0], [1.5, 0], [2, 0], [3.5, 0], [4, 0], [4.5, 0]]

# Intervalo de tempo para integração
t_start = 0.0
t_end = 20.0
num_points = 200

# Resolvendo o sistema de equações diferenciais para cada condição inicial e frequência angular
for ic, w in zip(initial_conditions, w_values):
    solution = solve_ivp(lambda t, y: equations(t, y, w, epsilon, q0), [t_start, t_end], ic, t_eval=np.linspace(t_start, t_end, num_points))

    # Plotando o gráfico da solução
    t_values = solution.t
    q_values = solution.y[0]

    plt.plot(t_values, q_values)

plt.xlabel('t')
plt.ylabel('q')
plt.xlim(0,10)
plt.title('Solução do sistema de equações diferenciais')
plt.grid(True)
plt.show()


# In[8]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Definição das equações diferenciais
def equations(t, y, w, epsilon, q0):
    q, p = y
    q_dot = p
    p_dot = -w**2 * (q + epsilon * (3/2 * q0) * q**2)
    return [q_dot, p_dot]

#Definindo C1 e D1
def C1(epsilon, w, t, C0, D0):
    return C0 + 3*m*w*(C0/m)**(3/2)*np.sin(D0+w*t)

def D1(epsilon, w, t, C0, D0):
    return D0+epsilon*(3/q0)*np.sqrt(C0*w/m)*((np.cos(D0+w*t)/3) + np.cos(D0+w*t))


# Parâmetros e condições iniciais
w_values = [1.6, 1.5, 1.4, 1.3, 1.2, 1.1]  # Valores de frequência angular correspondentes às condições iniciais
epsilon = 0.01  # Valor pequeno de epsilon
q0 = 1.0  # Valor arbitrário de q0
initial_conditions = [[1, 0], [1.5, 0], [2, 0], [3.5, 0], [4, 0], [4.5, 0]]

# Intervalo de tempo para integração
t_start = 0.0
t_end = 20.0
num_points = 200

plt.figure(figsize=(10, 6))
# Resolvendo o sistema de equações diferenciais para cada condição inicial e frequência angular
for ic, w in zip(initial_conditions, w_values):
    solution = solve_ivp(lambda t, y: equations(t, y, w, epsilon, q0), [t_start, t_end], ic, t_eval=np.linspace(t_start, t_end, num_points))

    # Plotando o gráfico da solução
    t_values = solution.t
    q_values = solution.y[0]

    plt.plot(t_values, q_values)

plt.xlabel('t')
plt.ylabel('q')
plt.xlim(0,10)
plt.title('Solução do sistema de equações diferenciais')
plt.grid(True)
plt.show()


# In[9]:


import numpy as np
import matplotlib.pyplot as plt

# Parâmetros
C0 = 3.0  # Valor inicial de C0
D0 = 5.0  # Valor inicial de D0
epsilon = 0.01  # Valor de epsilon
w = 1.2  # Frequência angular
q0 = 1.0  # Valor de q0
m = 0.5 # massa

# Funções C1(t) e D1(t)
def C1(t):
    return C0 + epsilon * 3 * m * w * (C0 / m) ** (3 / 2) * np.sin(D0 + w * t)

def D1(t):
    return D0 + epsilon * (3 / q0) * np.sqrt(C0 * w / m) * ((np.cos(D0 + w * t) / 3) + np.cos(D0 + w * t))

# Função q(C1(t), D1(t))
def q(C, D):
    return C * np.sin(D)

# Intervalo de tempo
t_start = 0.0
t_end = 20.0
num_points = 200

# Valores de t
t_values = np.linspace(t_start, t_end, num_points)

# Valores de C1(t) e D1(t)
C_values = C1(t_values)
D_values = D1(t_values)

# Valores de q(C1(t), D1(t))
q_values = q(C_values, D_values)

# Plot do gráfico
plt.plot(t_values, q_values)
plt.xlabel('t')
plt.ylabel('q')
plt.title('Gráfico de q(C1(t), D1(t))')
plt.grid(True)
plt.show()


# In[10]:


import numpy as np
import matplotlib.pyplot as plt

# Parâmetros
C0_values = [1.0, 2.0, 3.0]  # Valores iniciais de C0
D0_values = [2.0, 3.0, 4.0]  # Valores iniciais de D0
w_values = [1.0, 1.2, 1.4]  # Valores de w
epsilon = 0.01  # Valor de epsilon
q0 = 1.0  # Valor de q0
m = 0.5  # Massa

# Funções C1(t) e D1(t)
def C1(t, C0, D0, w):
    return C0 + epsilon * 3 * m * w * (C0 / m) ** (3 / 2) * np.sin(D0 + w * t)

def D1(t, C0, D0, w):
    return D0 + epsilon * (3 / q0) * np.sqrt(C0 * w / m) * ((np.cos(D0 + w * t) / 3) + np.cos(D0 + w * t))

# Função q(C1(t), D1(t))
def q(C, D):
    return C * np.sin(D)

# Intervalo de tempo
t_start = 0.0
t_end = 20.0
num_points = 200

# Valores de t
t_values = np.linspace(t_start, t_end, num_points)

# Plot dos gráficos para diferentes valores de C0, D0 e w
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

for i, (C0, D0, w) in enumerate(zip(C0_values, D0_values, w_values)):
    # Valores de C1(t) e D1(t)
    C_values = C1(t_values, C0, D0, w)
    D_values = D1(t_values, C0, D0, w)

    # Valores de q(C1(t), D1(t))
    q_values = q(C_values, D_values)

    # Plot do gráfico
    axs[i].plot(t_values, q_values)
    axs[i].set_xlabel('t')
    axs[i].set_ylabel('q')
    axs[i].set_title(f'Gráfico de q(C1(t), D1(t)) para C0 = {C0}, D0 = {D0}, w = {w}')
    axs[i].grid(True)

plt.tight_layout()
plt.show()


# In[11]:


import numpy as np
import matplotlib.pyplot as plt

# Parâmetros
C0_values = [1.0, 1.0, 1.00]  # Valores iniciais de C0
D0_values = [1.50, 1.50, 1.50]  # Valores iniciais de D0
w_values = [1.1, 1.15, 1.2]  # Valores de w
epsilon = 0.01  # Valor de epsilon
q0 = 1.0  # Valor de q0
m = 0.5  # Massa

# Intervalo de tempo
t_start = 0.0
t_end = 20.0
num_points = 200

# Valores de t
t_values = np.linspace(t_start, t_end, num_points)

# Plot do gráfico sobreposto
plt.figure(figsize=(8, 6))

for C0, D0, w in zip(C0_values, D0_values, w_values):
    # Funções C1(t) e D1(t)
    C_values = C0 + epsilon * 3 * m * w * (C0 / m) ** (3 / 2) * np.sin(D0 + w * t_values)
    D_values = D0 + epsilon * (3 / q0) * np.sqrt(C0 * w / m) * ((np.cos(D0 + w * t_values) / 3) + np.cos(D0 + w * t_values))

    # Função q(C1(t), D1(t))
    q_values = C_values * np.sin(D_values)

    # Plot do gráfico
    plt.plot(t_values, q_values)

plt.xlabel('t')
plt.ylabel('q')
plt.title('Gráfico de q(C1(t), D1(t))')
plt.grid(True)
#plt.legend()
plt.show()


# In[ ]:




