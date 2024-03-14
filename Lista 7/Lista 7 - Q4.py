#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Função que define o potencial V(x)
def potential(x):
    return x**4 / 4 - x**2 / 2

# Função que retorna as derivadas dx/dt = p e dp/dt = -dV/dx - b*y
def derivatives(t, state, b):
    x, p = state
    dx_dt = p
    dp_dt = -potential_derivative(x) - b * p
    return [dx_dt, dp_dt]

# Função que calcula a derivada do potencial em relação a x
def potential_derivative(x):
    return x**3 - x

# Valores iniciais de x e p
x0 = np.linspace(-2, 2, 40)
p0 = np.linspace(-2, 2, 40)

# Criação da malha de pontos (x, p)
X, P = np.meshgrid(x0, p0)

# Parâmetro b para o termo dissipativo
b = 0.0

# Cálculo das derivadas dx/dt e dp/dt em cada ponto
dX_dt, dP_dt = derivatives(0, [X, P], b)

# Plotagem do retrato de fase

plt.quiver(X, P, dX_dt, dP_dt, scale=20)
plt.xlabel('x')
plt.ylabel('p')
plt.plot(1, 0, 'ro', label='Ponto Fixo Estável (1, 0)')
plt.title('Retrato de Fase do Potencial com Termo Dissipativo(Flechas)')
plt.grid(True)
#plt.savefig('Retrato_de_Fase_DissipativoFlechas.png', format='png')
plt.show()


# In[3]:


import numpy as np
import matplotlib.pyplot as plt

# Função que define o potencial V(x)
def potential(x):
    return x**4 / 4 - x**2 / 2

# Função que retorna as derivadas dx/dt = p e dp/dt = -dV/dx - b*y
def derivatives(t, state, b):
    x, p = state
    dx_dt = p
    dp_dt = -potential_derivative(x) - b * p
    return [dx_dt, dp_dt]

# Função que calcula a derivada do potencial em relação a x
def potential_derivative(x):
    return x**3 - x

# Valores iniciais de x e p
x0 = np.linspace(-2, 2, 35)
p0 = np.linspace(-2, 2, 35)

# Criação da malha de pontos (x, p)
X, P = np.meshgrid(x0, p0)

# Parâmetro b para o termo dissipativo
b = 0.0

# Cálculo das derivadas dx/dt e dp/dt em cada ponto
dX_dt, dP_dt = derivatives(0, [X, P], b)

# Plotagem do retrato de fase com streamplot
plt.streamplot(X, P, dX_dt, dP_dt, color='blue', linewidth=1, arrowsize=0.8)
plt.xlabel('x')
plt.ylabel('p')
plt.title('Retrato de Fase do Potencial Conservativo')
plt.plot(1, 0, 'ro', label='Ponto Fixo Estável (1, 0)', color='black')
plt.grid(True)
#plt.savefig('Retrato_de_Fase_Conservativo.png', format='png')
plt.show()


# In[4]:


import numpy as np
import matplotlib.pyplot as plt

# Função que define o potencial V(x)
def potential(x):
    return x**4 / 4 - x**2 / 2

# Função que retorna as derivadas dx/dt = p e dp/dt = -dV/dx - b*y
def derivatives(t, state, b):
    x, p = state
    dx_dt = p
    dp_dt = -potential_derivative(x) - b * p
    return [dx_dt, dp_dt]

# Função que calcula a derivada do potencial em relação a x
def potential_derivative(x):
    return x**3 - x

# Valores iniciais de x e p
x0 = np.linspace(-1.5, 1.5, 40)
p0 = np.linspace(-1.5, 1.5, 40)

# Criação da malha de pontos (x, p)
X, P = np.meshgrid(x0, p0)

# Parâmetro b para o termo dissipativo
b = 0.2

# Cálculo das derivadas dx/dt e dp/dt em cada ponto
dX_dt, dP_dt = derivatives(0, [X, P], b)

# Plotagem do retrato de fase com streamplot e colorização
fig, ax = plt.subplots()
positive_mask = dX_dt > 0  # Máscara para pontos com dx/dt > 0
negative_mask = dX_dt < 0  # Máscara para pontos com dx/dt < 0

ax.streamplot(X, P, dX_dt, dP_dt, color='gray', linewidth=1, arrowsize=1)
ax.streamplot(X, P, dX_dt, dP_dt, color='blue', linewidth=1, arrowsize=1,
              start_points=np.column_stack((X[positive_mask], P[positive_mask])),
              minlength=0.1)
ax.streamplot(X, P, dX_dt, dP_dt, color='red', linewidth=1, arrowsize=1,
              start_points=np.column_stack((X[negative_mask], P[negative_mask])),
              minlength=0.1)

plt.xlabel('x')
plt.ylabel('p')
plt.title('Retrato de Fase do Potencial com Termo Dissipativo')
plt.plot(1, 0, 'ro', label='Ponto Fixo Estável (1, 0)', color='black')
plt.grid(True)
#plt.savefig('Retrato_de_Fase_Bacia_atrac.png', format='png')
plt.show()


# In[ ]:




