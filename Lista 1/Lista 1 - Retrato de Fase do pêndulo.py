#!/usr/bin/env python
# coding: utf-8

# # Mateus Alves de Oliveira

# ## Letra A ( $ E>mgr$, $ E<mgr $ e $ E_c = E = mgr $ )

# Sendo pêndulo plano e oscilando sobre um eixo fixo, se imaginarmos o ponto de partida inicial do pêndulo, o local mais alto, ou seja, onde a Energia potencial gravitacional é máxima. no primeiro caso $E>mgr$, temos que o pêndulo sempre estará oscilando no eixo e sempre dará "voltas" em torno do eixo. Quando $E<mgr$, com a evolução temporal, o pêndulo irá deixando de oscilar até parar no ponto mais baixo. Por fim, se $E=E_c=mgr$, o pêndulo inicia a oscilação, tem uma "volta" completa e para no mesmo lugar, no ponto mais alto. Claro, considerando o caso ideal, onde não há influências externas. 

# ## Letra B (Mudança de variáveis)

# ### Equações acopladas do oscilador harmônico

# 
# \begin{align*}
# \dot{\theta} &= \omega \\
# \dot{\omega} &= -\frac{g}{r}\sin{\theta}
# \end{align*}
# 
# 

# ### Mostrando a mudança de $\theta \rightarrow  x $ e de $\omega  \rightarrow  y$.

# Sabendo das definições $\tau = t / T$ e $T = \sqrt{r/g}$, onde $T$ é definido como sendo um tempo característico do sistema. E sabendo de $\theta = x$ e $y = T\omega $, então podemos mostrar variáveis adimensionais.

# \begin{align*}
# \frac{dx}{dt} = \frac{y}{T} &\Rightarrow \frac{d\left(\tau T\right)}{dx} = \frac{T}{y} \\
# T\frac{d\tau}{dx} = \frac{T}{y} &\Rightarrow \frac{dx}{d\tau} = y\\
# \therefore x' &= y
# \end{align*}

# Agora, fazendo para $y$, temos então

# \begin{align*}
# \frac{d\omega}{dt} = -\frac{g}{r}\sin{\theta} &\Rightarrow \omega = y / T \\
# \frac{d}{d\left(\tau T\right)}\left[\frac{y}{T} \right] = -\frac{g}{r}\sin{\theta} &\Rightarrow T^2 \frac{d\tau}{dy} = -\frac{r}{g} \frac{1}{\sin{x}}  \\
# \left(\sqrt{\frac{r}{g}}\right)^2 \frac{d\tau}{dy} = -\frac{r}{g} \frac{1}{\sin{x}} &\Rightarrow  \frac{d\tau}{dy} = - \frac{1}{\sin{x}} \\
# \therefore y' &= -\sin{x} 
# \end{align*}

# ## Letra C (Integração numérica e retrato de fase)

# In[5]:


import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from math import *
from numpy import *
from pylab import *


# In[6]:


# Definindo a função adimensional
def E(t, variaveis):
    x, y = variaveis
    dx = y
    dy = -sin(x)
    return [dx, dy]


# In[7]:


# A sintaxe eh solve_ivp(sistema [t0,t1], initial_condition).
# a opcao dense_output para ficar mais facil recuperar as solucoes para que sejam plotadas.

# trajetórias em "tempo" positivo
for i in range(-4,4,1):
    for j in range(-4,4,1):
        solucao = solve_ivp(E, [0,pi+50], [i,j],dense_output=True)
        t = np.linspace(0, 10, 500)
        plt.plot(solucao.sol(t).T[:, 0],solucao.sol(t).T[:, 1])

# trajetórias em "tempo" negativo
for i in range(-4,4,1):
    for j in range(-4,4,1):
        solucao = solve_ivp(E, [0,-pi-50], [i,j],dense_output=True)
        t = np.linspace(0, -10, 500)
        plt.plot(solucao.sol(t).T[:, 0],solucao.sol(t).T[:, 1])
    
#Plot

plt.xlim(-5.5,5.5)
plt.ylim(-5,5)

axhline(y = 0, color = 'black')
axvline(x = 0, color = 'black')
grid()
#plt.savefig('Retrato_de_Fase.png', format='png')
plt.show()


# Retrato de fase, resultado semelhante a Fig. 1.5b de Jos ́e & Saletan.
