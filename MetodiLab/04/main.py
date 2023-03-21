import math
import sys
import numpy as np
import matplotlib.pyplot as plt

# Esercizio 1
print("\n           Esercizio 1")
print(sys.float_info)
a = sys.float_info.min
b = sys.float_info.max
c = sys.float_info.epsilon
print(a)
print(b)
print(c)

# Massimo valore rappresentabile
a = (2 ** (1024 - 1)) * (1 + (1 - (2 ** (-53 + 1))))
# Minimo valore rappresentabile
b = (2 ** (-1021 - 1))
print(a)
print(b)

# Esercizio 2
print("\n\n          Esercizio 2")
# Spacing
a = 2 ** (52 + 1 - 53)
print(a)
x = 2 ** 52
print(x)
y = a + x
print(y)
# Dimostrazione del rounding to even
z = x + 0.5
print(z)
q = y + 0.5
print(q)

# Esercizio 3
print("\n\n          Esercizio 3")
a = 2.0
F = 2 * (a - 1) * (a ** 52) * (1024 + 1022 + 1) + 1
print(F)

# Esercizio 4
print("\n\n          Esercizio 4")
x = 2 ** (1 - 53)
y = 2 ** -52
print(x == y)
# Precisione di macchina
z = 1 + x
q = 1 + (x / 2)
print(z)
print(q)
print(z == q)

# Oppure
s10 = np.spacing(10.0**10)
print(s10)

# Esercizio 5
print("\n\n          Esercizio 5")
a = 1.234567890123400e+15
b = -1.234567890123401e+15
c = 0.06
s1 = (a + b) + c
s2 = (a + c) + b
s3 = a + (b + c)
print(s1)
print(s2)
print(s3)
a = 0.23371258e-4
b = 0.33678429e+2
c = -0.33677911e+2
s1 = (a + b) + c
s2 = (a + c) + b
s3 = a + (b + c)
print(s1)
print(s2)
print(s3)

# Esercizio 6
print("\n\n          Esercizio 6")
x = 7777
y1 = math.sqrt((x ** 2) + 1) - x
y2 = 1 / (math.sqrt(x ** 2 + 1) + x)
print(y1)
print(y2)

x = 77777777
y1 = math.sqrt((x ** 2) + 1) - x
y2 = 1 / (math.sqrt(x ** 2 + 1) + x)
print(y1)
print(y2)

# Esercizio 7
print("\n\n          Esercizio 7")

# Array reale
x = 10.0**np.arange(21)

# Array approssimato
A = (1 / x) - (1 / (x + 1))
B = 1 / (x * (x + 1))
print(A)
print(B)

# Calcolo tramite formula teorica e formula data
Er = np.abs(A - B) / np.abs(B)
print(Er)

plt.loglog(x, Er, 'b-')
plt.show()
print(np.spacing)

# Esercizio 8
print("\n\n          Esercizio 8")
k = np.arange(1, 9)
a = 1
b = 10.0**k
c = 1

d = (b**2) - (4 * a * c)
x1 = (-b + np.sqrt(d)) / (2 * a)
x2 = (-b - np.sqrt(d)) / (2 * a)
print(f"k={k}:      x1={x1},    x2={x2}")

x1new = c / (a * x2)
x1true = -10.0**(-k)

err = np.abs(x1true - x1) / np.abs(x1true)
err_new = np.abs(x1true - x1new) / np.abs(x1true)
plt.semilogy(k, err, 'r-', k, err_new, 'g:')
plt.show()

# Esercizio 9
print("\n\n          Esercizio 9")
e = math.exp(1)
x = 10.0**np.arange(17)
print(e)
print(x)

y = ((1 / x) + 1)**x
z = np.full_like(y, e)
print(y)

plt.semilogx(x, z)
plt.semilogx(x, y)
plt.show()
