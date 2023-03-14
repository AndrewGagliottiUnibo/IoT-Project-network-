import math
import sys
import numpy as np

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


# Per il calcolo del valore assoluto come da regola matematica
def relativeError(real, machine):
    return abs(machine - real) / abs(real)


# Array reale
x = np.array([10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3, 10 ** 4,
              10 ** 5, 10 ** 6, 10 ** 7, 10 ** 8, 10 ** 9,
              10 ** 10, 10 ** 11, 10 ** 12, 10 ** 13, 10 ** 14,
              10 ** 15, 10 ** 6, 10 ** 7, 10 ** 18, 10 ** 19,
              10 ** 20])

# Array approssimato
A = (1 / x) - (1 / (x + 1))
print(A)

# Calcolo tramite formula teorica e formula data
Er = np.array([])
for i, j in zip(x, A):
    Er = np.append(Er, relativeError(i, j))
B = 1 / (x * (x + 1))
print(Er)
print(B)

# Esercizio 8
print("\n\n          Esercizio 8")

# Risoluzione
for k in range(1, 9):
    a = 1
    b = 10**k
    c = 1

    d = (b**2) - (4 * a * c)
    x1 = (-b + math.sqrt(d)) / (2 * a)
    x2 = (-b - math.sqrt(d)) / (2 * a)

    print(f"k={k}:      x1={x1},    x2={x2}")

# Esercizio 9
print("\n\n          Esercizio 9")
e = math.exp(1)
print(e)

# Calcolo
for k in range(0, 16):
    n = 10**k

    sol = (1 / (n + 1))**n
    print(sol)

