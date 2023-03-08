import numpy
import matplotlib.pyplot as plt


# Funzioni sfruttate
# Esercizio 6
def myfunc(first, second):
    if first * second < 1000:
        print(first * second)
    else:
        print(first + second)


# Esercizio 7
def funcSumProd(inputList):
    return sum(inputList), numpy.prod(inputList)


def funcSumOrProd(inputListDiff, value):
    if value:
        return sum(inputListDiff)
    else:
        return numpy.prod(inputListDiff)


# Esercizi
# Esercizio 1
a = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
b = []

for i in a:
    if i < 5:
        print(i)
        b.append(i)

print(b)


# Esercizio 2
a = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

c = a + b
print(c)

c = list(dict.fromkeys(c))
print(c)


# Esercizio 3
z = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
print(list(filter(lambda e: e % 2 == 0, z)))
print([i for i in z if i % 2 == 0])


# Esercizio 4
for i in range(10):
    if i < 6:
        print('* ' * i)
    else:
        print('* ' * (10 - i))


# Esercizio 5
s1 = 'by'
s2 = 'blueberry'

for i in range(len(s1)):
    if s2.count(s1[i]) > 0:
        print('true')
    else:
        print('false')


# Esercizio 6
myfunc(3, 4)
myfunc(50, 70)


# Esercizio 7
a = [1, 1, 2, 3, 5, 8, 89]
print(funcSumProd(a))
print(funcSumOrProd(a, True))
print(funcSumOrProd(a, False))


# Esercizio 8
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [i ** 2 for i in x]
z = [i ** 3 for i in x]
# oppure: y = list(filter(lambda i: i**2, x))
print(y)

# Disegno i grafici
plt.subplot(1, 2, 1)
plt.title('y su x')
plt.plot(y, x, 'ro-', linewidth=0.7)
plt.plot(x, y, 'gd--', linewidth=0.7)
plt.grid(axis = 'y')
plt.legend(["Funzione x","Funzione y"])

plt.subplot(1, 2, 2)
plt.title('z su x')
plt.plot(y, x, 'ro-', linewidth=0.7)
plt.plot(z, y, 'bd-', linewidth=0.7)
plt.grid(axis = 'y')
plt.legend(["Funzione x","Funzione z"])

plt.savefig("filename.jpg")  
plt.show()