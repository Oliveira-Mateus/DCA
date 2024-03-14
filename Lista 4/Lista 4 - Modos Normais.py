#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

# Definir as constantes do sistema
g = 9.81  # Aceleração da gravidade
m = 7  # Massa da primeira massa
l = 3  # Comprimento do pêndulo da primeira massa
k = 1  # Constante da mola
omega1 = np.sqrt(g/l)  # Frequência simétrico
omega2 = np.sqrt(g/l + 2*k/m)  # Frequência simétrico

print("w1=",omega1)
print("w2 =",omega2)
print("w1/w2 =",omega1/omega2)  # Imprime a razão entre as frequências

# Escreve as coordenadas dos modos normais
def zeta1(t, a1):
    return a1 * np.cos(omega1 * t)

def zeta2(t, a2):
    return a2 * np.cos(omega2 * t)

# Escreve os momentos
def p1(t, a1):
    return -omega1 * a1 * np.sin(omega1 * t)

def p2(t, a2):
    return -omega2 * a2 * np.sin(omega2 * t) == 0

zeta10=1.5
zeta20=3.0

#------------------------------------------------------------------
t = np.linspace(0, 200, 1000)
eta1 = zeta1(t, zeta10) + zeta2(t, zeta20)

plt.xlim(0, 200)  
plt.plot(t, eta1,  linewidth=1)
plt.xlabel('Tempo')
plt.ylabel('η1(t) = ζ1(t) + ζ2(t)')
plt.legend()
#plt.savefig("Eta1_versus_t.png") 
plt.show()


# In[2]:


import numpy as np
import matplotlib.pyplot as plt

# Definir as constantes do sistema
g = 9.81  # Aceleração da gravidade
m = 7  # Massa da primeira massa
l = 3  # Comprimento do pêndulo da primeira massa
k = 1  # Constante da mola
omega1 = np.sqrt(g/l)  # Frequência simétrico
omega2 = np.sqrt(g/l + 2*k/m)  # Frequência simétrico

# Escreve as coordenadas dos modos normais
def zeta1(t, a1):
    return a1 * np.cos((omega1 * t * np.pi )/omega2)

def zeta2(t, a2):
    return a2 * np.cos(omega2 * t)

# Escreve os momentos
def p1(t, a1):
    return -omega1 * a1 * np.sin((omega1 * t * np.pi )/omega2)

def p2(t, a2):
    return -omega2 * a2 * np.sin(omega2 * t) == 0

zeta10=1.5
zeta20=3.0


N = 120 # Quantidade de pontos que o problema exige
t = np.linspace(0, 10, N)
plt.title('Muitos pontos')
plt.scatter(zeta1(t, zeta10), p1(t, zeta10), label='(ζ1, P1)', s=7, color='r')
plt.xlabel('ζ1')
plt.ylabel('P1')
plt.legend()
#plt.grid()
#plt.savefig("Muitos_pontos.png") #Salve pela quantidade de pontos que o problema exige
plt.show()


# In[ ]:




