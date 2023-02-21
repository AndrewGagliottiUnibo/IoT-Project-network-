# Esercizio 1
import numpy

a = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
b = []

for i in a:
    if(i < 5):
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
print(list(filter(lambda i: i % 2 == 0, z)))
print([i for i in z if i % 2 == 0])

# Esercizio 4
for i in range(10):
    if(i < 6):
        print('* ' * i)
    else:
        print('* ' * (10 - i))

# Esercizio 5
s1='by'
s2='blueberry'

for i in range(len(s1)):
    if s2.count(s1[i]) > 0:
        print('true')
    else:
        print('false')

# Esercizio 6
def myfunc(x, y):
    if x * y < 1000:
        print(x * y)
    else:
        print(x + y)

myfunc(3, 4)
myfunc(50, 70)

# Esercizio 7
def funcSumProd(x):
    return (sum(x), numpy.prod(x))

def funcSumOrProd(x, value):
    if value == True:
        return sum(x)
    else:
        return numpy.prod(x)


a = [1, 1, 2, 3, 5, 8, 89]
print(funcSumProd(a))
print(funcSumOrProd(a, True))
print(funcSumOrProd(a, False))

# Esercizio 8
x = [0,1,2,3,4,5,6,7,8,9]
y = [i**2 for i in x]
print(y)

# continua ...