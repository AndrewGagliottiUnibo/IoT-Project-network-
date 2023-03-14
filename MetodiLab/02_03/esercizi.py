import numpy as np
import time

'''
    Esercizio 1
'''
print("\n\n Esercizio 1")
a = [1.2, 5.4, 6, 1.59]
b = [5.2, 1.2, 1.5, 2]

# Radice quadrata
c = np.array(a)
print(np.sqrt(c))

# Potenza
c = np.array(a)
print(np.exp(c / 2))

# Vettore somma
c = np.array(a)
d = np.array(b)
print(c + d)

# Prodotto
c = np.array(a)
d = np.array(b)
print(c * d)

# Vettore equispaziato
print(np.arange(0, 30, 0.5))

# Vettore di 100 valori
print(np.linspace(1, 2, 100))

# Array 2x6
c = np.linspace(10, 20, 6)
d = np.linspace(20, 10, 6)
arr = np.append(c, d)
print(arr.reshape(2, 6))

# Prodotto scalare tra vettori
c = np.array(a)
d = np.array(b)
e = np.dot(c, d)
print("e = ", e)

# Matrice A
c = np.array(a)
d = np.array(b)
arr = np.append(c, d)
A = arr.reshape(2, 4)
print(A)

# y = A * b
d = np.array(b)
y = A * d
print(y)

# Vettore diagonale D
c = np.array(a)
D = np.diag(c)
print(D)

'''
    Esercizio 2
'''
print("\n\n Esercizio 2")
a = [2, 2, 2, 3]
A = np.full((3, 4), a)
print(A)

a = np.identity(3, dtype="int64") * 2
b = np.array([1, 2, 3])
e = np.vstack((b, b, b))
c = np.array([10, 10, 10])
c = c.reshape((3, 1))
d = np.zeros((3, 3), dtype="int64")
B = np.hstack((a, e, d, d, c))
print(B)

identity = np.identity(5, dtype="int64") * 2
shiftUp = np.eye(5, k=1, dtype="int64")
shiftDown = np.eye(5, k=-1, dtype="int64")
C = identity - shiftDown - shiftUp
print(C)

a = np.array([2, 2, 3, 3])
b = np.array([0, 0, 5, 5])
c = np.array([0, 0, 0, 0])
D = np.vstack((a, a, c, b, b))
print(D)

'''
    Esercizio 3
'''
print("\n\n Esercizio 3")
def sumNNumbers(numbers, value):
    result = 0

    # For loop
    if value:
        for num in range(1, numbers + 1):
            if not (num % 2) == 0:
                result = result + num
        print(result)
    # numpy sum()
    else:
        num = np.arange(1, 100, 2)
        result = np.sum(num)
        print(result)

# Calling functions
sumNNumbers(100, True)
sumNNumbers(100, False)

'''
    Esercizio 4
'''
print("\n\n Esercizio 4")
# Fibonacci ricorsivo
def recurFibo(num):
    if num <= 2:
        return 1
    else:
        return recurFibo(num - 1) + recurFibo(num - 2)

# Fibonacci iterativo
def fib(num):
    actual, result = 0, 1

    for i in range(num - 1):
        actual, result = result, actual + result

    return result

# Formula di Binet
def fiboBinet(num):
    return int(((((1 + np.sqrt(5))/2)**num) - (((1 - np.sqrt(5))/2)**num)) / np.sqrt(5))

# Fibonacci iterativo con numpy
def fibN(num):
    if num <= 2:
        return 1

    fibo = np.ones(num, dtype=int)

    for i in range(2, num):
        fibo[i] = fibo[i - 1] + fibo[i - 2]

    return fibo[-1]

start = time.time()
a = recurFibo(24)
end = time.time() - start
print(a)
print(end)

start = time.time()
a = fib(24)
end = time.time() - start
print(a)
print(end)

start = time.time()
a = fiboBinet(24)
end = time.time() - start
print(a)
print(end)

start = time.time()
a = fibN(24)
end = time.time() - start
print(a)
print(end)
