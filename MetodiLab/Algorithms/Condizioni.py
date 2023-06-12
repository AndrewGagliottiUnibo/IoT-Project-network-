from scipy.io import loadmat
import numpy as np
import scipy.linalg as spl
import RisolviSis as RS
import matplotlib.pyplot as plt

# m == n?
#
# Innanzittutto devo verificare se il sistema è sovradeterminato o meno
# m =/= n il sistema è sovradeterminato.
# Inoltre in base ai valori sulle dimensioni posso anche capire quanto sia
# grande o meno la matrice e decidere meglio quale algortimo applicare.
n, m = A.shape
print("Dimensione di A:", n, m)


# Densa o Sparsa?
#
# Se ho una matrice quadrata controllo se questa sia densa o sparsa:
# se più del 33% degli elementi è diverso da 0 allora la matrice è densa.
n_zeri = np.count_nonzero(A) / (n * m)
perc_n_zeri = n_zeri * 100
print("Percentuale elementi diversi da zero:", perc_n_zeri, "%")


# Simmetrica? Lo faccio solo se è grande e sparsa.
#
# Se ho una matrice quadrata e grande e sparsa, verifico ora che la matrice sia
# simmetrica oppure no: in base a questa valutazione capisco quale metodo
# utilizzare.
# Se la matrice è uguale alla trasposta allora questa sarà definita positiva.
flag = (A == A.T)
if (np.all(flag) == False):
    print("La matrice non è simmetrica")
    # Controllo se é a diagonale dominante
else:
    print("La matrice è simmetrica")
    # GS SOR, Gradiente, Gradiente Coniugato

    # Se la matrice e' simmetrica devo vedere se è anche
    # definita positiva: uso gli autovalori e Silvester
    eig = np.linalg.eigvals(A)
    if (np.all(eig > 0)):
        print("La matrice è definita positiva")


# Diagonale dominante?
#
# A questo punto passo ad analizzare la diagonale della matrice: da questo
# capisco se la diagonale è dominante o meno e se lo è anche in modo stretto.
# A e' a diagonale dominante se il valore assoluto dell'elemento sulla diagonale
# e' >= della somma in valore in valore assoluto di tutti i valori sulla
# rispettiva riga.
def check_diagonale(A):
    n = A.shape[0]
    flag = True
    for i in range(n):
        diag_elem = np.abs(A[i, i])
        print("elemento diagonale:", diag_elem)
        # per tutte le righe: elementi in riga sommati - elemento diagonale.
        row_sum = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
        if diag_elem < row_sum:
            print("Matrice non a diagonale dominante")
            flag = False
            return flag

    # Se e' a diagonale dominante hai: Jacobi, Gauss-Siedel e GS SOR
    return flag

diag = check_diagonale(A)
print("Matrice a diagonale dominante? ", diag)


# Piccola e Densa: Simmetrica? Controllo se è simmetrica.
# 
# Se la matrice è uguale alla trasposta allora questa sarà definita positiva.
flag = (A == A.T)
if (np.all(flag) == False):
    print("La matrice non è simmetrica")
else:
    print("La matrice è simmetrica e definita positiva")

    # Se la matrice e' simmetrica devo vedere se è anche
    # definita positiva: uso gli autovalori e Silvester
    eig = np.linalg.eigvals(A)
    if (np.all(eig > 0)):
        print("La matrice è definita positiva")
        # Allora usi Cholesky o QR
    else:
        print("La matrice non è definita positiva")
        if(np.linalg.det(A) != 0):
            print("Il Determinante è diverso da 0")
            # Allora usi LU o QR
        else:
            print("Il Determinante è uguale a 0")
            # Usi QR

# m > n ?
#
# Vedo se il determinante di A è diverso da 0
rank = np.linalg.matrix_rank(A1)
if (A.shape[0] == rank):
    print("Rango massimo e il determinante è diverso da 0.")
    # Calcoli l'indice di condizionamento
else:
    print("Rango non massimo e il determinante è nullo.")
    # SVDLS


# Condizionamento
#
ind_cond = np.linalg.cond(A1)
print(ind_cond)

if (ind_cond < A1.shape[0]**3):
    print("Matrice ben condizionata")
    # Eqnorm
elif (ind_cond < A1.shape[0]**10):
    # anche se la vera condizione è un'altra
    print("Matrice mediamente mal condizionata")
    # QRLS
else:
    print("Matrice mal condizionata")
