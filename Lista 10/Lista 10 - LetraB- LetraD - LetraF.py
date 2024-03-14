#!/usr/bin/env python
# coding: utf-8

# In[55]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#Soliton-Soliton
def phi_ss(x, t, u):
    return 4 * np.arctan(u * np.sinh(function_gamma(u) * x) * (1 / np.cosh(u * function_gamma(u) * t)))

def function_gamma(u):
    return ( 1 / (1-u**2) )**(1/2)

# Valores dos parâmetros
u1 = 0.3
u2 = 0.7

# Intervalo de x e t
x = np.linspace(-10, 10, 1000)
t = np.linspace(-10, 20, 1000)

# Criação da malha de pontos (x, t)
X, T = np.meshgrid(x, t)

# Avaliação da função phi_ss em cada ponto da malha
Z1 = phi_ss(X, T, u1)
Z2 = phi_ss(X, T, u2)


# Criar uma nova colormap personalizada a partir da 'jet'
colors = cm.jet(np.linspace(0, 1, 255))
colors[:, 1] = 0  # Alterar o canal verde (índice 1) para 0 (preto)
new_cmap = cm.colors.ListedColormap(colors)


# Plotagem do gráfico 3D
fig = plt.figure(figsize=(10, 6))
ax = fig.gca(projection='3d')
ax.plot_surface(X, T, Z1, cmap=new_cmap)
# Rotaciona a observação
ax.view_init(30, -103)  # Angulo de elevação 1°, angulo de azimute 2°

ax.set_xlim(-10, 10)  # Intervalo para o eixo x
ax.set_ylim(-10, 10)  # Intervalo para o eixo y
ax.set_zlim(-10, 10)  # Intervalo para o eixo z

# Configuração dos rótulos dos eixos
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('φss(x, t)')
ax.set_title('Interação Soliton-Soliton (u=0.3)')


#fig.savefig('Soliton-Soliton_1.png')
# Exibição do gráfico
plt.show()
#-----------------------------------------------------------------------
# Plotagem do gráfico 3D
fig = plt.figure(figsize=(10, 6))
ax = fig.gca(projection='3d')
ax.plot_surface(X, T, Z2, cmap=new_cmap)



# Rotaciona a observação
ax.view_init(30, -103)  # Angulo de elevação 1°, angulo de azimute 2°

ax.set_xlim(-10, 10)  # Intervalo para o eixo x
ax.set_ylim(-10, 10)  # Intervalo para o eixo y
ax.set_zlim(-10, 10)  # Intervalo para o eixo z

# Configuração dos rótulos dos eixos
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('φss(x, t)')
ax.set_title('Interação Soliton-Soliton (u=0.7)')

#fig.savefig('Soliton-Soliton_2.png')
# Exibição do gráfico
plt.show()


# In[56]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Soliton-AntiSoliton
def phi_sa(x, t, u):
    return 4 * np.arctan((1/u) * (1 / np.cosh( function_gamma(u) * x)) * np.sinh(u * function_gamma(u) * t))

def function_gamma(u):
    return ( 1 / (1-u**2) )**(1/2)

# Valores dos parâmetros
u1 = 0.1
u2 = 0.6

# Intervalo de x e t
x = np.linspace(-10, 10, 1000)
t = np.linspace(-10, 20, 1000)

# Criação da malha de pontos (x, t)
X, T = np.meshgrid(x, t)

# Avaliação da função phi_ss em cada ponto da malha
Z1 = phi_sa(X, T, u1)
Z2 = phi_sa(X, T, u2)

# Plotagem do gráfico 3D
fig = plt.figure(figsize=(10, 6))
ax = fig.gca(projection='3d')
ax.plot_surface(X, T, Z1, cmap='jet')
# Rotaciona a observação
ax.view_init(30, -103)  # Angulo de elevação 1°, angulo de azimute 2°

ax.set_xlim(-10, 10)  # Intervalo para o eixo x
ax.set_ylim(-10, 10)  # Intervalo para o eixo y
ax.set_zlim(-10, 10)  # Intervalo para o eixo z

# Configuração dos rótulos dos eixos
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('φsa(x, t)')
ax.set_title('Interação Soliton-AntiSoliton (u=0.1)')

#fig.savefig('Soliton-AntiSoliton_1.png')
# Exibição do gráfico
plt.show()
#-----------------------------------------------------------------------
# Plotagem do gráfico 3D
fig = plt.figure(figsize=(10, 6))
ax = fig.gca(projection='3d')
ax.plot_surface(X, T, Z2, cmap='jet')
# Rotaciona a observação
ax.view_init(30, -130)  # Angulo de elevação 1°, angulo de azimute 2°

ax.set_xlim(-10, 10)  # Intervalo para o eixo x
ax.set_ylim(-10, 10)  # Intervalo para o eixo y
ax.set_zlim(-10, 10)  # Intervalo para o eixo z

# Configuração dos rótulos dos eixos
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('φsa(x, t)')
ax.set_title('Interação Soliton-AntiSoliton (u=0.6)')

#fig.savefig('Soliton-AntiSoliton_2.png')
# Exibição do gráfico
plt.show()


# In[57]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Solução do tipo breather
def phi_b(x, t, u):
    return 4 * np.arctan((1/ ( function_gamma(u) * u)) * (1 / np.cosh( (function_gamma(u))**(-1) * x)) * np.sin(u * t))

def function_gamma(u):
    return ( 1 / (1-u**2) )**(1/2)

# Valores dos parâmetros
u1 = 0.3
u2 = 0.8

# Intervalo de x e t
x = np.linspace(-10, 10, 1000)
t = np.linspace(-10, 20, 1000)

# Criação da malha de pontos (x, t)
X, T = np.meshgrid(x, t)

# Avaliação da função phi_ss em cada ponto da malha
Z1 = phi_b(X, T, u1)
Z2 = phi_b(X, T, u2)

# Plotagem do gráfico 3D
fig = plt.figure(figsize=(10, 6))
ax = fig.gca(projection='3d')
ax.plot_surface(X, T, Z1, cmap='inferno')
# Rotaciona a observação
ax.view_init(30, -103)  # Angulo de elevação 1°, angulo de azimute 2°

ax.set_xlim(-10, 10)  # Intervalo para o eixo x
ax.set_ylim(-10, 10)  # Intervalo para o eixo y
ax.set_zlim(-10, 10)  # Intervalo para o eixo z

# Configuração dos rótulos dos eixos
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('φb(x, t)')
ax.set_title('Interação Breather (u=0.3)')

#fig.savefig('Breather_1.png')
# Exibição do gráfico
plt.show()fer
#-----------------------------------------------------------------------
# Plotagem do gráfico 3D
fig = plt.figure(figsize=(10, 6))
ax = fig.gca(projection='3d')
ax.plot_surface(X, T, Z2, cmap='inferno')
# Rotaciona a observação
ax.view_init(30, -130)  # Angulo de elevação 1°, angulo de azimute 2°

ax.set_xlim(-10, 10)  # Intervalo para o eixo x
ax.set_ylim(-10, 10)  # Intervalo para o eixo y
ax.set_zlim(-10, 10)  # Intervalo para o eixo z

# Configuração dos rótulos dos eixos
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('φb(x, t)')
ax.set_title('Interação Breather (u=0.8)')

#fig.savefig('Breather_2.png')
# Exibição do gráfico
plt.show()


# In[ ]:




