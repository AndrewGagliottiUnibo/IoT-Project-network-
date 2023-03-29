# Traduce una funzione da simbolica a numerica
from sympy.utilities.lambdify import lambdify
import scipy.linalg as spl
import numpy as np
import sympy as sym
import numpy.linalg as npl

# Esercitazione 5_N Esercizio 1
print("\n       Esercitazione 5_N, Esercizio 1")

# Definisco simbolicamente i componenti della funzione
q = sym.symbols('q')
p = sym.symbols('p')
f = -p - sym.sqrt(p**2 + q)
df = sym.diff(f, q, 1)
print("f: {}, Derivata di f: {}".format(f, df))
fp = f.subs(p, 10**5)
dfp = df.subs(p, 10**5)
print("f: {}, Derivata di f: {}".format(fp, dfp))

# Traduzione da simboli a numeri: np sta ad indicare la compatibilità con i numpy array
# Ora le funzioni sono lambda e quindi le posso valutare come voglio
fNum = lambdify(q, fp, np)
dfNum = lambdify(q, dfp, np)

# Valutazione del condizionamento
index = 10.0**(-np.arange(1, 11))
print("Indici: {}".format(index))

# Per numeri successivi e più piccoli dello spacing non fanno progredire la funzione
# q è rilevante fino ad un certo, quando vai oltre ad un numero più piccolo dello
# spacing, la q viene trascurata perchè è vicina a 0 e l'operazione divene (-p + p)
print("Spacing: {}".format(np.spacing((10**5)**2)))

K = np.abs((dfNum(index) * index) / fNum(index))
print("K: {}".format(K))

# Esercitazione 5_N Esercizio 2
print("\n       Esercitazione 5_N, Esercizio 2")
x = np.array([1, 2, 3, -4, -5])
A = np.array([[1, 2, 3, 4, -5], [6, 7, -8, -9, 10], [11, -12, -13, 14, -15]])

print("Senza la funzione norm()")
normXInf = np.max(np.abs(x))
normX1 = np.sum(np.abs(x))
normAInf = np.max(np.sum(np.abs(A), axis=0))
normA1 = np.max(np.sum(np.abs(A), axis=1))
print(normXInf)
print(normX1)
print(normAInf)
print(normA1)

print("\nCon la funzione norm()")
normXInf = npl.norm(x, np.inf)
normX1 = npl.norm(x, 1)
normAInf = npl.norm(A, np.inf)
normA1 = npl.norm(A, 1)
print(int(normXInf))
print(int(normX1))
print(int(normAInf))
print(int(normA1))

# Esercitazione 5_N Esercizio 3
print("\n       Esercitazione 5_N, Esercizio 3")
A = np.array([[4, -1, 6], [2, 3, -3], [1, -2, 9/2]])
normA2E = np.sqrt(np.max(npl.eigvals(A.T @ A)))
normA2N = npl.norm(A, 2)
print(normA2E)
print(normA2N)

# Esercitazione 5_N Esercizio 4
print("\n       Esercitazione 5_N, Esercizio 4")
A = np.array([[4, -1, 6], [2, 3, -3], [1, -2, 9/2]])
K = (np.sqrt(np.max(npl.eigvals(A.T @ A)))) / (np.sqrt(np.min(npl.eigvals(A.T @ A))))
print(K)

# Esercitazione 5_N Esercizio 5
print("\n       Esercitazione 5_N, Esercizio 5")
x = np.arange(1.0, 7.0)
V = np.vander(x, increasing=True)
print(V)
K = npl.norm(V, 1) * npl.norm(npl.inv(V), 1)
KNpl = npl.cond(V, 1)
# Molto alto perchè convivono elementi molto grandi e molto piccoli assieme
print(K)
print(KNpl)

x = np.array([1, 1, 1, 1, 1, 1])
b = V * x
bPert = b + (b * 0.025)
xPert = spl.solve(npl.inv(V), bPert)
print(xPert)

ERb = np.abs((bPert - b) / bPert)
ERx = np.abs((xPert - x) / xPert)
print(ERb)
print(ERx)

# Esercitazione 5_N Esercizio 6
print("\n       Esercitazione 5_N, Esercizio 6")
A = np.array([[6, 63, 622.2], [63, 622.2, 6967.8], [622.2, 6967.8, 73393.5664]])
b = np.array([1.1, 2.33, 1.7])
x = spl.solve(A, b)
print(x)

APert = 0.01 * A
bPert = 0.01 * b
xPert = spl.solve(APert, bPert)
print(xPert)

ERx = np.abs((xPert - x) / xPert)
print(ERx)

# Esercitazione 5_N Esercizio 7
print("\n       Esercitazione 5_N, Esercizio 7")
H = spl.hilbert(4)
b = np.array([1, 1, 1, 1]).T
x = spl.solve(H, b)
print(x)

bPert = b + (b * 0.025)
xPert = spl.solve(npl.inv(H), bPert)
print(xPert)

ERx = np.abs((xPert - x) / xPert)
print(ERx)

# Esercitazione 5_N Esercizio 8
print("\n       Esercitazione 5_N, Esercizio 8")
print("non spiegato")

# Esercitazione 5_N Esercizio 9
print("\n       Esercitazione 5_N, Esercizio 9")
print("non spiegato")
