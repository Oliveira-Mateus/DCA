#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

def lorenz_system(t, state, sigma, beta, rho):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return np.array([dx_dt, dy_dt, dz_dt])

def runge_kutta4(f, t0, tf, dt, state, *args):
    t = np.arange(t0, tf, dt)
    n = len(t)
    states = np.zeros((n, len(state)))
    states[0] = state

    for i in range(1, n):
        k1 = f(t[i-1], states[i-1], *args)
        k2 = f(t[i-1] + 0.5*dt, states[i-1] + 0.5*dt*k1, *args)
        k3 = f(t[i-1] + 0.5*dt, states[i-1] + 0.5*dt*k2, *args)
        k4 = f(t[i-1] + dt, states[i-1] + dt*k3, *args)
        states[i] = states[i-1] + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    return states

# Parâmetros do sistema de Lorenz
sigma = 10
beta = 8/3
rho = 100
#rho = 166.3
#rho = 212

# Condição inicial
state0 = np.array([1.0, 0.0, 0.0])

# Intervalo de tempo
t0 = 0
tf = 40
dt = 0.001

# Simulação do sistema de Lorenz
states = runge_kutta4(lorenz_system, t0, tf, dt, state0, sigma, beta, rho)

# Extração das variáveis
x = states[:, 0]
y = states[:, 1]
z = states[:, 2]
t = np.arange(t0, tf, dt)

# Plot dos gráficos
plt.figure(figsize=(12, 4))


plt.plot(t, x)
plt.ylim(-40,30)
plt.xlim(20,40)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title(f'r = {rho}')
plt.savefig('x_t_plot_r=100.png')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(t, y)
plt.xlim(20,40)
plt.ylim(-60,40)
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title(f'r = {rho}')
plt.savefig('y_t_plot_r=100.png')  # Salvar figura
plt.tight_layout()
plt.show()

# Plot do gráfico de x vs z
plt.figure(figsize=(10, 6))
plt.plot(x, z)
plt.xlim(-40,30)
plt.ylim(50,150)
plt.xlabel('x(t)')
plt.ylabel('z(t)')
plt.title(f'r = {rho}')
plt.savefig('x_z_plot_r=100.png')  # Salvar figura de x vs z
plt.show()
#------------------------------------------------------------------------
rho = 166.3
# Simulação do sistema de Lorenz
states = runge_kutta4(lorenz_system, t0, tf, dt, state0, sigma, beta, rho)

# Extração das variáveis
x = states[:, 0]
y = states[:, 1]
z = states[:, 2]
t = np.arange(t0, tf, dt)

# Plot dos gráficos
plt.figure(figsize=(12, 4))

#plt.subplot(1, 3, 1)
plt.plot(t, x, color='red')
plt.xlim(10,40)
plt.ylim(-60,50)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title(f'r = {rho}')
plt.savefig('x_t_plot_r=166.png')
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 4))
plt.plot(t, y, color='red')
plt.xlim(10,40)
plt.ylim(-120,100)
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title(f'r = {rho}')
plt.savefig('y_t_plot_r=166.png')  # Salvar figura


plt.tight_layout()
plt.show()

# Plot do gráfico de x vs z
plt.figure(figsize=(10, 6))
plt.plot(x, z, color='red')
plt.xlim(-60,50)
plt.ylim(60,280)
plt.xlabel('x(t)')
plt.ylabel('z(t)')
plt.title(f'r = {rho}')
plt.savefig('x_z_plot_r=166.png')  # Salvar figura de x vs z
plt.show()


#------------------------------------------------------------------------
rho = 212
# Simulação do sistema de Lorenz
states = runge_kutta4(lorenz_system, t0, tf, dt, state0, sigma, beta, rho)

# Extração das variáveis
x = states[:, 0]
y = states[:, 1]
z = states[:, 2]
t = np.arange(t0, tf, dt)

# Plot dos gráficos
plt.figure(figsize=(12, 4))
plt.plot(t, x, color='black')
plt.xlim(2,20)
plt.ylim(-50,40)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title(f'r = {rho}')
plt.savefig('x_t_plot_r=212.png')
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 4))
#plt.subplot(1, 3, 2)
plt.plot(t, y, color='black')
plt.xlim(2,20)
plt.ylim(-120,80)
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title(f'r = {rho}')
plt.savefig('y_t_plot_r=212.png')  # Salvar figura
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
# Plot do gráfico de x vs z
plt.plot(x, z, color='black')
plt.xlim(-50,40)
plt.ylim(120,300)
plt.xlabel('x(t)')
plt.ylabel('z(t)')
plt.title(f'r = {rho}')
plt.savefig('x_z_plot_r=212.png')  # Salvar figura de x vs z
plt.show()


# In[2]:


import numpy as np
import matplotlib.pyplot as plt

def logistic_map(x, r):
    return r * x * (1 - x)

def lyapunov_exponent(r, num_iterations=1000):
    x0 = np.random.rand()
    x = x0
    sum_lyapunov = 0

    for _ in range(num_iterations):
        x_next = logistic_map(x, r)
        delta = 0.0001
        x_perturbed = x + delta
        x_next_perturbed = logistic_map(x_perturbed, r)

        sum_lyapunov += np.log(np.abs((x_next_perturbed - x_next) / delta))
        x = x_next

    return sum_lyapunov / num_iterations

# Parâmetros do mapa logístico
r_values = np.linspace(2.5, 4.0, 500)  # Intervalo de valores de r
lyapunov_values = []

# Calcular o expoente de Lyapunov para cada valor de r
for r in r_values:
    lyapunov = lyapunov_exponent(r)
    lyapunov_values.append(lyapunov)


plt.figure(figsize=(12, 8))
# Plotar o gráfico do expoente de Lyapunov em função de r
plt.plot(r_values, lyapunov_values)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('r')
plt.ylabel('Expoente de Lyapunov(λ)')
plt.title('Mapa logístico do Expoente de Lyapunov λ')
#plt.savefig('Expoente_de_Lyapunov.png')
plt.show()


# In[3]:


import numpy as np
import matplotlib.pyplot as plt

def logistic_map(r, x):
    return r * x * (1 - x)

def orbit_diagram(r, x0, num_iterations, num_discard):
    x = np.zeros(num_iterations)
    x[0] = x0

    for i in range(1, num_iterations):
        x[i] = logistic_map(r, x[i-1])

    return x[num_discard:]

# Parâmetros
r_values = np.linspace(2.4, 4.0, 1000)  # Valores de r
x0 = 0.2  # Condição inicial
num_iterations = 140 # Número de iterações por valor de r
num_discard = 20  # Número de transientes iniciais a serem descartados

# Gerar diagrama de órbita
orbit_data = []
for r in r_values:
    x = orbit_diagram(r, x0, num_iterations, num_discard)
    orbit_data.extend([(r, xi) for xi in x])

# Plot do diagrama de órbita
plt.figure(figsize=(12, 8))
plt.scatter(*zip(*orbit_data), s=1, c='black')
plt.xlabel('r')
plt.ylabel('x')
plt.title('Diagrama de Órbita do Mapa Logístico')
#plt.savefig('Diagrama_de_Orbita_do_Mapa_Logistico.png')
plt.show()


# In[ ]:




