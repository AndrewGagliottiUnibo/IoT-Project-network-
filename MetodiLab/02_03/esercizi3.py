import math

import numpy as np
import random as rand
import time as tm
import matplotlib.pyplot as plt

# Esercizio 0
# linspace()
print("\n     Esercizio 0")
start = tm.time()
a = np.linspace(0, 4, 2000)
end = tm.time() - start
print(end)
print(a)

# For loop
start = tm.time()
for i in range(0, 2000):
    b = rand.randrange(0, 4)
    a = np.append(a, b)

end = tm.time() - start
print(end)
print(a)

# Esercizio 1
print("\n     Esercizio 1")
c = np.arange(7, 42, 2)
c1 = c.reshape((2, 9))
c1[0, 1] = 2
print(c)
print(c1)

# Esercizio 2
print("\n     Esercizio 2")
c = np.arange(15, 42, 2)
print(c)
c.resize((7, 2))
print(c)

# Esercizio 3
print("\n     Esercizio 3")
c = np.arange(15, 42, 2)
c1 = c.reshape((7, 2))
c1[0, 1] = 2
print(c)
print(c1)

# Esercizio 4
print("\n     Esercizio 4")
d = np.arange(15, 42, 2)
print(d)
d1 = np.resize(d, (7, 2))
d1[0, 1] = 2
print(d1)

# Esercizio 5
print("\n     Esercizio 5")
s = np.linspace(0, 4, 100)
s.reshape((int(100 / 4), 4))
print(s)

# Esercizio 6
print("\n     Esercizio 6")
s = np.linspace(0, 4, 100)
s.reshape(4, (int(100 / 4)))
print(s)

# Esercizio 7
print("\n     Esercizio 7")
a = np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]])
b = np.copy(a)
print(a)
print(b)
a = a.ravel()
b = b.flatten()
print(a)
print(b)

# Esercizio 8
print("\n     Esercizio 8")
f1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
f2 = np.array([16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])
f = f1.copy()
f = f.reshape((4, 4))
print(f)
f = f2.copy()
f = f.reshape((4, 4))
print(f)
f3 = np.dot(f1, f2)
print(f3)

# Esercizio 9
print("\n     Esercizio 9")
A = np.random.randn(2, 10).reshape(20)
B = np.arange(0, 20)
print(A)
print(B)
print(A + B)

# Esercizio 10
print("\n     Esercizio 10")
B = np.full((5, 5), 15)
B3 = np.copy(B)
B3[0, 0] = 150
print(B)
print(B3)

# Esercizio 11
print("\n     Esercizio 11")
f = np.linspace(0, 1, 10).reshape(10, 1)
print(f)

# Esercizio 12
print("\n     Esercizio 12")
a = np.array([14, 13, 12, 11])
b = np.array([[4, 3, 2, 1], [9, 10, 11, 12]])
c = np.concatenate((a.reshape((1, 4)), b), axis=0)
print(c)

# Esercizio 13
print("\n     Esercizio 13")
a = np.array([14, 13, 12, 11])
b = np.array([[4, 3, 2, 1], [9, 10, 11, 12], [4, 3, 2, 1], [9, 10, 11, 12]])
c = np.concatenate((a.reshape((4, 1)), b), axis=1)
print(c)

# Esercizio 14
print("\n     Esercizio 14")
A = np.eye(18, dtype=int) * 10
B = np.eye(18, k=-1, dtype=int) * -12
C = np.eye(18, k=1, dtype=int) * -2
print(A + B + C)

# Esercizio 15
print("\n     Esercizio 15")
A = np.eye(int(math.sqrt(A.size)), dtype=int)
print(A)

# Esercizio 16
print("\n     Esercizio 16")
# selezione delle righe e colonne specifiche
A1 = A[2:10:2, 3:6:2]
A1[0, 0] = 6
print(A)
print(A1)

# Esercizio 17
print("\n     Esercizio 17")
A1 = np.copy(A[2:10:2, 3:6:2])
A1[0, 0] = 8
print(A)
print(A1)

# Esercizio 18
print("\n     Esercizio 18")
A = np.random.randint(low=1, high=21, size=(5, 5))
B = np.random.randint(low=1, high=7, size=(5, 5))
C = A + B
D = A - B
E = A * B
F = A / B
print(B)
print(C)
print(D)
print(E)
print(F)

# Esercizio 19
print("\n     Esercizio 19")
C = A.ravel()
D = B.ravel()
E = np.dot(C, D)
print(E)

# Esercizio 20
print("\n     Esercizio 20")
sumR = np.sum(B, axis=1)
print(sumR)

# Esercizio 21
print("\n     Esercizio 21")
sumC = np.sum(B, axis=0)
print(sumC)

# Esercizio 22
print("\n     Esercizio 22")
sumB = np.sum(B)
print(sumB)

# Esercizio 23
print("\n     Esercizio 23")
minB = np.min(B)
minRB = np.min(B, axis=1)
minCB = np.min(B, axis=0)
print(minB)
print(minRB)
print(minCB)

# Esercizio 24
print("\n     Esercizio 24")
sumC = np.sum(np.abs(B), axis=0)
print(sumC)
sumCMax = np.max(sumC)
print(sumCMax)

# Esercizio 25
print("\n     Esercizio 25")
sumR = np.sum(np.abs(B), axis=1)
print(sumR)
sumRMax = np.max(sumR)
print(sumRMax)

# Esercizio 26
print("\n     Esercizio 26")
print(B)
B[[1, 2]] = B[[2, 1]]
print(B)

# Esercizio 27
print("\n     Esercizio 27")
maxV = np.max(B[:, 0])
maxI = np.argmax(B[:, 0])
print(maxV)
print(maxI)

# Esercizio 28
print("\n     Esercizio 28")
def border(m, n, show=False):
    arr = np.zeros((m, n))
    arr[0, :] = 1
    arr[m - 1, :] = 1
    arr[:, 0] = 1
    arr[:, n - 1] = 1

    if show:
        print(arr)

    return arr

aTate = border(8, 8, True)

# Esercizio 29
def visualizza(func, inf, sup):
    x = np.linspace(inf, sup, 80)
    y = func(x)

    plt.plot(x, y)
    plt.title("     Esercizio 29")
    plt.show()

func = lambda x: x**2 - 2*x + 1
visualizza(func, -2, 0)

# Esercizio 30
def visualizza2(func, inf, sup, func2, inf2, sup2):
    x = np.linspace(inf, sup, 80)
    y = func(x)
    x2 = np.linspace(inf2, sup2, 80)
    y2 = func2(x2)

    plt.figure()
    plt.plot(x, y, label='func')
    plt.plot(x2, y2, label='func2')
    plt.title("     Esercizio 30")
    plt.legend()
    plt.show()

func = lambda x: x**2 - 2*x
func2 = lambda x: x**3 - 5*x
visualizza2(func, -2, 3, func2, -2, 3)
