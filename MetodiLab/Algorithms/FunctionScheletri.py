from scipy.io import loadmat
import numpy as np
import scipy.linalg as spl
import RisolviSis as RS
import matplotlib.pyplot as plt
import math


# LU: fattorizzazione LU
#
# A non singolare: ne esiste l'inversa, ha rango massimo e determinante 
# diverso da 0. La soluzione in questo modo é unica: si vuole dimostrare
# che esiste una matrice di permutazione che valida la seguente relazione: 
# PA = LU che risolve il seguente sistema lineare:
# { Lz = Pb
# { Ux = z
def LUSolve(P, L, U, b):
    # Matrice triangolare inferiore con diagonale con tutti termini pari a 1
    y, flag = RS.Lsolve(L, np.dot(P, b))
    # Matrice triangolare superiore
    x, flag = RS.Usolve(U, y)
    return x, flag

cond = np.linalg.cond(A)
print(cond)

# Risolvo il problema chiamando la funzione di scipy.linalg che mi da' gratis la
# fattorizzazione.
P, L, U = spl.lu(A)
sol, flag = LUSolve(P.T, L, U, b)
print(sol)

# Di quanto ci discostiamo dalla soluzione esatta?
x_esatta = np.ones_like(b)
err = np.linalg.norm(sol - x_esatta) / np.linalg.norm(x_esatta)
print("Errore soluzione LU: ", err * 100)


# LU: fattorizzazione LU con pivoting massimo
#
# Le regole di partenza sono sempre le stesse viste per la fattorizzazione LU 
# classica: A a rango massimo, matrice di permutazione P da ricercare e 
# soddisfare il sistema lineare che trova la soluzione. Quest'ultima richiesta
# può essere semplificata nel seguente modo.
#
# Uso un algoritmo in place: la fattorizzazione LU e' stabile in senso debole 
# in quanto la matrice L viene costruita indipendentemente dalle caratteristiche 
# di A mentre U dipende in maniera esponenziale dall’ordine della matrice.
cond = np.linalg.cond(A)
print(cond)

PV, L, U = spl.lu(A)
P = PV.T
y, flag = RS.Lsolve(L, P@b)

if (flag == 0):
    sol, flag1 = RS.Usolve(U, y)

print(sol)

# Di quanto ci discostiamo dalla soluzione esatta?
x_esatta = np.ones_like(b)
err = np.linalg.norm(sol - x_esatta) / np.linalg.norm(x_esatta)
print("Errore soluzione LU: ", err * 100)


# QR: fattorizzazione QR
#
# Se la matrice che dobbiamo analizzare non e' simmetrica e nemmeno definita 
# positiva posso usare una fattorizzazione che è sempre valida: Q matrice 
# matrice ortogonale e R matrice triangolare superiore non singolare. Il sistema
# da risolvere è il seguente:
# { Qz = b
# { Rx = z
cond = np.linalg.cond(A)
print(cond)
Q, R = spl.qr(A)
n = A.shape[0]

# Qz = b
z = Q.T @ b

# Rx = z - la matrice è triangolare superiore quindi devo per forza sfruttare 
# la sua relativa risoluzione,Usolve()
sol, flag = RS.Usolve(R, z)
print(sol)

# Di quanto ci discostiamo dalla soluzione esatta? L'errore relativo e' solitamente 
# più piccolo nel caso in cui la soluzione sia calcolata con il metodo QR. 
# L'algoritmo è stabile in senso forte.
x_esatta = np.ones_like(b)
err = np.linalg.norm(sol - x_esatta) / np.linalg.norm(x_esatta)
print("Errore soluzione QR: ", err * 100)


# Cholesky: fattorizzazione di Cholesky
#
# Il metodo si applica alle matrici simmetriche e definite positive dalle quali è 
# possibile ottenere una matrice L triangolare inferiore che presenta elementi 
# diagonali positivi tali per cui vale la relazione A = L * L.T. Il sistema lineare 
# da soddisfare in questo caso e':
# { Ly = b
# { L.T x = y
cond = np.linalg.cond(A)
print(cond)

# Genero la matrice L
L = spl.cholesky(A, lower=True)

# Risolvo il sistema
y, flag = RS.Lsolve(L, b)
sol, flag = RS.Usolve(L.T, y)
print(sol)

# Di quanto ci discostiamo dalla soluzione esatta?
# L'algoritmo è stabile in senso forte.
x_esatta = np.ones_like(b)
err = np.linalg.norm(sol - x_esatta) / np.linalg.norm(x_esatta)
print("Errore soluzione Cholesky: ", err * 100)


# Jacobi
#
# Definito anche metodo di decomposizione in cui la matrice di partenza viene
# decomposta in 3 matrici che sono poi relazionate diversamente fra loro: A = D + E + F.
# Per Jacobi vale la relazione: M = D e N = -(E + F).
# L’algoritmo di Jacobi è definito se gli elementi diagonali di A sono diversi da 0 (a meno
# che A sia non singolare e quindi riordinabile affiché il metodo sia applicabile) e siccome
# ogni elemento dell’iterato è indipendente dagli altri, il metodo e ' parallelizzabile.
# Il metodo non restituisce una soluzione ottima ma converge ad una soluzione definibile
# ottima (approssimata).
def jacobi(A, b, x0, toll, itmax):
    n = A.shape[0]
    # Estraggo la diagonale dalla matrice A, costruisco la matrice diagonale
    d = np.diag(A)
    D = np.diag(d)
    # Creo la matrice triangolare inferiore da A, -1 perchè escludo la diagonale
    E = np.tril(A, - 1)
    # Estraggo dalla matrice A la sua matrice triangolare superiore,
    # con diagonale esclusa
    F = np.triu(A, 1)

    # Decomposizione adottata nel metodo di Jacobi: si tiene a mente che M = D
    N = -(E + F)

    # Controllo il raggio spettrale che rispetti le regole di ammissibilità della soluzione.
    # Si utilizza la condizione sufficiente come valida condizione di ricerca di una soluzione.
    # Cercando il raggio spettrale, devo vedere che questo sia minore di 1 affinché ci sia una 
    # veloce convergenza del metodo iterativo. Il raggio spettrale di una matrice è il suo 
    # autovalore di modulo massimo
    invM = np.diag(1 / d)
    T = np.dot(invM, N)
    autT = np.linalg.eigvals(T)
    rho = np.max(np.abs(autT))
    print("Raggio spettrale: ", rho)
    if (rho > 1):
        print("Raggio spettrale maggiore di 1, nessuna soluzione")

    # Cuore dell'algoritmo: inizializzo il contatore delle iterazioni.
    # Tengo traccia del vettore degli errori per poi fare un grafico
    # rappresentativo.
    it = 0
    err_vet = []
    err = 1000

    while it <= itmax and err >= toll:
        x = (b + np.dot(N, x0)) / d.reshape(n, 1)
        # Se c'è convergenza le soluzioni non cambiano più
        err = np.linalg.norm(x - x0) / np.linalg.norm(x)
        err_vet.append(err)
        # Al passo successivo, x0 sarà la x del passo precedente
        x0 = x.copy()
        it += 1

    return x, it, err_vet

# Posso scegliere un qualunque vettore iniziale
n = A.shape[0]
x0 = np.zeros((n, 1))
# la tolleranza e il numero max di iterazioni li impostiamo perchè il metodo converge 
# ad una soluzione, non trova la soluzione al sistema lineare. Solitamente si basa 
# su una percentuale di errore da cui sono affetti i dati.
itmax = 100
toll = 1e-8

xJ, itJ, err_vetJ = jacobi(A, b, x0, toll, itmax)
print("Soluzione:   ", xJ)
print("Iterazioni:  ", itJ)
plt.semilogy(np.arange(itJ), err_vetJ)
plt.show()


# Gauss-Seidel
#
# Definito anche metodo di decomposizione in cui la matrice di partenza viene
# decomposta in 3 matrici che sono poi relazionate diversamente fra loro: A = D + E + F.
# Per Gauss-Siedel vale la relazione: M = E + D e N = -F.
# Gauss-Siedel converge sicuramente se la matrice e' simmetrica e definita positiva.
# Per calcolare la nuova componente di una iterazione il metodo utilizza tutte quelle
# calcolate fino a quel punto: come conseguenza di ciò l'algoritmo non è parallelizzabile.
# Il metodo non restituisce una soluzione ottima ma converge ad una soluzione definibile
# ottima (approssimata).
def gauss_seidel(A, b, x0, toll, itmax):
    d = np.diag(A)
    # Estraggo la diagonale dalla matrice A, costruisco la matrice diagonale
    D = np.diag(d)
    # Creo la matrice triangolare inferiore da A, -1 perchè escludo la diagonale
    E = np.tril(A, - 1)
    # Estraggo dalla matrice A la sua matrice triangolare superiore,
    # con diagonale esclusa
    F = np.triu(A, 1)

    # Decomposizione adottata nel metodo di Gauss-Siedel:
    M = D + E
    N = -F

    # Condizione necessaria e sufficiente alla convergenza.
    # Il raggio spettrale di una matrice è il suo autovalore di modulo massimo.
    # Controllo il raggio spettrale che rispetti le regole di ammissibilità della soluzione.
    # Si utilizza la condizione sufficiente come valida condizione di ricerca di una soluzione.
    invM = np.linalg.inv(M)
    T = np.dot(invM, N)
    autT = np.linalg.eigvals(T)
    rho = np.max(np.abs(autT))
    print("Raggio spettrale: ", rho)
    if (rho > 1):
        print("Raggio spettrale maggiore di 1, nessuna soluzione")

    # Cuore dell'algoritmo: inizializzo il contatore delle iterazioni.
    # Tengo traccia del vettore degli errori per poi fare un grafico
    # rappresentativo.
    it = 0
    err_vet = []
    err = 1000

    while it <= itmax and err >= toll:
        temp = b - np.dot(F, x0)
        x, flag = RS.Lsolve(M, temp)
        err = np.linalg.norm(x - x0) / np.linalg.norm(x)
        err_vet.append(err)
        x0 = x.copy()
        it += 1

    return x, it, err_vet

# Posso scegliere un qualunque vettore iniziale
n = A.shape[0]
x0 = np.zeros((n, 1))
# la tolleranza e il numero max di iterazioni li impostiamo perchè il metodo converge 
# ad una soluzione, non trova la soluzione al sistema lineare. Solitamente si basa su 
# una percentuale di errore da cui sono affetti i dati.
itmax = 100
toll = 1e-8

xG, itG, err_vetG = gauss_seidel(A, b, x0, toll, itmax)
print("Soluzione GS:    ", xG)
print("Iterazioni GS:   ", itG)
plt.semilogy(np.arange(itG), err_vetG)
plt.show()


# Gauss-Seidel SOR
#
# Definito anche metodo di decomposizione in cui la matrice di partenza viene
# decomposta in 3 matrici che sono poi relazionate diversamente fra loro: A = D + E + F.
# Per Gauss-Siedel SOR vale la relazione: M = E + D e N = -F.
# Gauss-Siedel SOR converge sicuramente se la matrice e' simmetrica e definita positiva.
# Per calcolare la nuova componente di una iterazione il metodo utilizza tutte quelle
# calcolate fino a quel punto: come conseguenza di ciò l'algoritmo non è parallelizzabile.
# Il metodo non restituisce una soluzione ottima ma converge ad una soluzione definibile
# ottima (approssimata).
# A differenza di Gauss-Siedel classico, il metodo subisce una accelerazione verso la
# soluzione ottima sfruttando un parametro omega di rilassamento.
def gauss_seidel_sor(A, b, x0, omega, toll, itmax):
    errore = 1000
    d = np.diag(A)
    D = np.diag(d)
    Dinv = np.diag(1/d)
    # Estraggo la diagonale dalla matrice A, costruisco la matrice diagonale.
    E = np.tril(A, -1)
    # Creo la matrice triangolare inferiore da A, -1 perchè escludo la diagonale.
    F = np.triu(A, 1)

    # Decomposizione adottata nel metodo di Gauss-Siedel SOR: devo introdurre
    # un parametro omega che riduca il più possibile il raggio spettrale.
    # Questo si fa' perché il problema principale della convergenza del metodo
    # e' legato al mal condizionamento di A, che causa il rallentamento oppure
    # la perdita della convergenza stessa del metodo.
    M_omega = D + omega * E
    N_omega = (1 - omega) * D - omega * F
    T = np.dot(np.linalg.inv(M_omega), N_omega)
    M = D + E
    N = -F

    # Il raggio spettrale di una matrice è il suo autovalore di modulo massimo.
    autovalori = np.linalg.eigvals(T)
    raggiospettrale = np.max(np.abs(autovalori))
    print("raggio spettrale Gauss-Seidel SOR ", raggiospettrale)

    # Cuore dell'algoritmo: inizializzo il contatore delle iterazioni.
    # Tengo traccia del vettore degli errori per poi fare un grafico
    # rappresentativo.
    it = 0
    x_old = x0.copy()
    x_new = x0.copy()
    er_vet = []

    while it <= itmax and errore >= toll:
        temp = b - np.dot(F, x_old)
        x_tilde, flag = RS.Lsolve(M, temp)
        # Il parametro omega agisce qui per accelerare la convergenza
        x_new = (1 - omega) * x_old + omega * x_tilde
        errore = np.linalg.norm(x_new - x_old) / np.linalg.norm(x_new)
        er_vet.append(errore)
        x_old = x_new.copy()
        it += 1
    return x_new, it, er_vet

# Posso scegliere un qualunque vettore iniziale
n = A.shape[0]
x0 = np.zeros((n, 1))

# se 0 < omega < 1, il metodo di GS non ha convergenza
# se omega > 1, il metodo di GS converge lentamente
omega = 1.4
# la tolleranza e il numero max di iterazioni li impostiamo perchè il
# metodo converge ad una soluzione, non trova la soluzione al sistema
# lineare. Solitamente si basa su una percentuale di errore da cui
# sono affetti i dati.
itmax = 100
toll = 1e-8

xG, itG, err_vetG = gauss_seidel_sor(A, b, x0, omega, toll, itmax)
print("Soluzione GSS:    ", xG)
print("Iterazioni GSS:   ", itG)
plt.semilogy(np.arange(itG), err_vetG)
plt.show()


# Gradiente discesa ripida
#
def steepestdescent(A, b, x0, itmax, toll):
    # Definisco l'approssimazione iniziale della soluzione.
    x = x0

    # Calcolo il residuo ...
    r = A.dot(x) - b

    # ... che poi imposto come direzione di discesa.
    p = -r

    # Definisco l'errore di partenza e altre cose che mi serviranno per dare il 
    # via al cuore dell'algoritmo.
    norm_b = np.linalg.norm(b)
    errore = np.linalg.norm(r) / norm_b
    vec_sol = []
    vec_sol.append(x)
    vet_residuo = []
    vet_residuo.append(errore)

    # Il metodo della discesa ripida ha come caratteristica particolare di scegliere 
    # la direzione p k-esima come l’antigradiente della F calcolato nell’iterato k-esimo, 
    # che in questo caso è letta come la direzione di massima decrescita.
    it = 0
    while errore >= toll and it < itmax:
        it += 1

        # Scelta dello step size: per ottenere il minimo valore possibile del gradiente 
        # lungo la direzione scelta.
        A_p = A.dot(p)
        rTr = np.dot(r.T, r)
        alpha = rTr / np.dot(p.T, A_p)

        # Aggiorno l'iterato.
        x = x + alpha * p
        r = r + alpha * A_p

        # Salvo i dati per l'output.
        vec_sol.append(x)
        errore = np.linalg.norm(r) / norm_b
        vet_residuo.append(errore)

        # "Discendo".
        p = -r

    return x, vet_residuo, vec_sol, it

# Indice di condizionamento della matrice A: indica quanto e' lenta la convergenza.
print("Condizionameto di A", np.linalg.cond(A))

toll = 1e-8
it_max = 10000
x0 = np.zeros_like(b)

# Dal grafico si nota carattere a zigzag del metodo del gradiente, dovuto al fatto che 
# il gradiente di una iterata è ortogonale al gradiente di quello precedente. L'avanzamento 
# a zig zag dell'algoritmo è anche causa della lentezza della convergenza alla soluzione.
x_gr, vet_r_gr, vec_sol_gr, itG = steepestdescent(A, b, x0, it_max, toll)
print("Iterazioni Gradiente ", itG)
plt.semilogy(np.arange(itG + 1), vet_r_gr)


# Gradiente coniugato
#
def conjugate_gradient(A, b, x0, itmax, toll):
    # Definisco l'approssimazione iniziale della soluzione.
    x = x0

    # Calcolo del residuo ...
    r = A.dot(x) - b

    # ... che poi imposto come direzione di discesa.
    p = -r

    # Definisco l'errore di partenza e altre cose che mi serviranno per dare il via al 
    # cuore dell'algoritmo.
    norm_b = np.linalg.norm(b)
    errore = np.linalg.norm(r) / norm_b
    vec_sol = []
    vec_sol.append(x)
    vet_residuo = []
    vet_residuo.append(errore)

    # Il metodo del gradiente coniugato, a differenza di quello del gradiente a discesa 
    # ripida, non tiene conto solo della direzione del gradiente ma anche di quella che 
    # era la direzione scelta nell'iterata precedente. Il numero di iterazioni che occorrono 
    # per raggiungere la precisione richiesta e' di gran lunga inferiore alla dimensione del 
    # sistema e questo rende il metodo molto utile per problemi di grosse dimensioni.
    it = 0
    while errore >= toll and it < itmax:
        it += 1

        # Scelta dello step size: per ottenere il minimo valore possibile del gradiente 
        # lungo la direzione scelta.
        A_p = A.dot(p)
        rtr = np.dot(r.T, r)
        alpha = rtr / np.dot(p.T, A_p)

        # Aggiorno l'iterato tenendo conto sia della direzione del gradiente e sia 
        # della iterazione precedente.
        x = x + alpha * p
        r = r + alpha * A_p
        # Direzione scelta in modo che punti verso il centro della ellissi di convergenza.
        gamma = np.dot(r.T, r) / rtr

        # Salvo i dati per l'output.
        vec_sol.append(x)
        errore = np.linalg.norm(r) / norm_b
        vet_residuo.append(errore)

        # "Discendo".
        p = -r + gamma * p

    return x, vet_residuo, vec_sol, it

# Indice di condizionamento della matrice A: a differenza del metodo dello steepest 
# descent, a parita di indice di condizionamento questo risulta più veloce.
print("Condizionameto di A", np.linalg.cond(A))

toll = 1e-8
it_max = 10000
x0 = np.zeros_like(b)

# Dal grafico si nota carattere a zigzag del metodo del gradiente coniugato, 
# seppur questo sia molto più veloce rispetto a quello dello steepest descent.
x_cg, vet_r_cg, vec_sol_cg, itCG = conjugate_gradient(A, b, x0, it_max, toll)
print("Iterazioni Gradiente Coniugato ", itCG)
plt.semilogy(np.arange(itCG + 1), vet_r_cg)


# Metodo delle equazioni normali
#
# La risoluzione di un sistema sovradeterminato risulta essere un problema mal posto
# in quanto potrebbe accadere che la soluzione non esista o non sia unica. Per
# renderlo ben posto lo si riformula come "risoluzione nel senso dei minimi quadrati"
# Definito un vettore residuo r(x) = Ax - b, cerchiamo una x* che rende minima
# la norma 2 al quadrato del residuo.
def eqnorm(A, b):

    # Se la matrice A ha rango massimo ed è ben condizionata (condizione necessaria)
    # possiamo procedere con il metodo delle equazioni normali: poniamo G = A.T @ A,
    # matrice simmetrica che viene associata alla x che si vuole cercare andando
    # a definire una funzione F(x) per la quale il gradiente si annulli.
    # Questa nuova matrice sarà quadrata n x n con determinante diverso da 0.
    G = A.T @ A

    print("Indice di condizionamento di G ", np.linalg.cond(G))

    # Il problema descritto precedentemente si può risolvere facilmente grazie alla
    # risoluzione del seguente sistema lineare: G x = A.T b. Siccome G è simmetrica
    # e definita positiva, il sistema può essere risolto utilizzando il metodo
    # di Cholesky. In questo modo sono anche sicuro che il risultato ottenuto sia
    # un minimo della funzione F(x).
    f = A.T @ b
    L = spl.cholesky(G, lower=True)

    y, flag = RS.Lsolve(L, f)

    if (flag == 0):
        x, flag = RS.Usolve(L.T, y)

    return x

# La soluzione del problema dei minimi quadrati mediante equazioni normali richiede
# solo che la matrice A del sistema sovradeterminato A x = b abbia rango massimo.
# L’idea è di individuare una trasformazione ortogonale che, applicata al residuo
# r = Ax - b, lo trasformi in modo tale da rendere più facile la soluzione del problema
# di minimizzarne la norma: le trasformazioni ortogonali lasciano inalterata la
# norma 2 di un vettore.
sol = eqnorm(A, b)
print("Soluzione nel senso dei minimi quadrati:\n  ", sol)
print("Norma soluzione: ", np.linalg.norm(sol))

# Di quanto ci discostiamo dalla soluzione esatta?
x_esatta = np.ones_like(sol)
err = np.linalg.norm(sol - x_esatta) / np.linalg.norm(x_esatta)
print("Errore soluzione Eqnorm: ", err * 100)


# Metodo QRLS
#
# La risoluzione di un sistema sovradeterminato risulta essere un problema mal posto
# in quanto potrebbe accadere che la soluzione non esista o non sia unica. Per
# renderlo ben posto lo si riformula come "risoluzione nel senso dei minimi quadrati"
# Definito un vettore residuo r(x) = Ax - b, cerchiamo una x* che rende minima
# la norma 2 al quadrato del residuo.
# Se A ha rango massimo ed è mediamente mal condizionata si può usare il metodo QR
# per la soluzione del problema dei minimi quadrati, calcolando i due fattori Q
# ed R di A, lavorando sempre solo sulla matrice A, senza dover passare alla matrice
# A.T @ A, che è molto più mal condizionata, e sfruttando una fattorizzazione abbastanza
# stabile. Sotto queste condizioni il risultato sarà molto più preciso di altri metodi.
def QRLS(A, b):
    n = A.shape[1]
    Q, R = spl.qr(A)

    # Risolvo il sistema triangolare superiore: prime n righe e tutte le colonne
    # per avere una matrice quadrata
    h = Q.T @ b
    x, flag = RS.Usolve(R[0:n, :], h[0:n])
    residuo = np.linalg.norm(h[n:])**2

    return x, residuo

sol, residuo = QRLS(A, b)
print("Soluzione nel senso dei minimi quadrati:\n  ", sol)
print("Residuo: ", residuo)
print("Norma soluzione: ", np.linalg.norm(sol))

# Di quanto ci discostiamo dalla soluzione esatta?
x_esatta = np.ones_like(sol)
err = np.linalg.norm(sol - x_esatta) / np.linalg.norm(x_esatta)
print("Errore soluzione QRLS: ", err * 100)


# Metodo SVDLS
#
# La risoluzione di un sistema sovradeterminato risulta essere un problema mal posto
# in quanto potrebbe accadere che la soluzione non esista o non sia unica. Per
# renderlo ben posto lo si riformula come "risoluzione nel senso dei minimi quadrati"
# Definito un vettore residuo r(x) = Ax - b, cerchiamo una x* che rende minima
# la norma 2 al quadrato del residuo.
# Se A non ha rango massimo si sfrutta il metodo di decomposizione ai valori singolari,
# secondo il quale la matrice A viene decomposto in due vettori U e V.T detti rispettivamente
# vettori singolari sinistri e destri per cui valgono le seguenti proprietà:
# - tutti i valori singolari sono reali >= 0
# - il rapporto tra il massimo e il minimo dei singolari ci dà l'indice di condizionamento
#   di A
# - i valori singolari non nulli ci dicono quale é il rango di A.
# - il primo dei valori singolari é sempre il massimo.
def SVDLS(A, b):
    n = A.shape[1]  # numero di colonne di A
    m = A.shape[0]  # numero di righe

    # La decomposizione di A
    U, s, V_T = spl.svd(A)
    V = V_T.T

    # Calcolo del rango della matrice, numero dei valori singolari maggiori di una soglia
    thresh = np.spacing(1) * m * s[0]
    k = np.count_nonzero(s > thresh)

    # Controllo sul rango
    print("Rango: ", k)
    if (k < n):
        print("La matrice non è a rango massimo")
    else:
        print("La matrice è a rango massimo")

    # La condizione aggiuntiva che inseriamo per risolvere il problema dei minimi
    # quadrati: minimizzo il residuo cercando una soluzione fra le infinite di
    # norma minima.
    d = U.T @ b
    d1 = d[:k].reshape(k, 1)
    s1 = s[:k].reshape(k, 1)
    # Risolve il sistema diagonale di dimensione k x k avene come matrice dei
    # coefficienti la matrice Sigma.
    c = d1 / s1
    x = V[:, :k] @ c
    residuo = np.linalg.norm(d[k:])**2
    return x, residuo

sol, residuo = SVDLS(A1, b1)
print("Soluzione nel senso dei minimi quadrati:\n  ", sol)
print("Residuo: ", residuo)
print("Norma soluzione: ", np.linalg.norm(sol))

# Di quanto ci discostiamo dalla soluzione esatta?
x_esatta = np.ones_like(sol)
err = np.linalg.norm(sol - x_esatta) / np.linalg.norm(x_esatta)
print("Errore soluzione SVDLS: ", err * 100)


# K-esimo polinomio di Lagrange
#
# Date le coppie (x_i, y_i) che rappresentano dei nodi di interpolazione,
# si definiscono x i nodi e y le valutazioni di un fenoemeno in quei nodi.
# Determinare un polinomio di interpolazione significa innanzitutto determinarne
# i suoi coefficienti, tali che soddisfino la condizione di interpolazione 
# P(x_i) = y_i.
# Successivamente a ciò si potranno determinare dati che stanno sia dentro 
# (interpolazione) che fuori (estrapolazione) dal range dei dati forniti.
# Il polinomio è facilmente rappresentabile come una matrice di Vandermonde,
# che ricordiamo essere molto mal condizionata e quindi soggetta a grossi 
# errori, se sottoposta anche a minime perturbazioni, a cui è associato un 
# vettore colonna della valutazione del fenomeno nei nodi.
# In questo caso il sistema lineare ammette una ed una sola soluzione se e 
# solo se la matrice è quadrata ed il rango è massimo. La matrice di Vandermonde 
# ha sempre rango massimo se tutti gli x_i sono distinti, conseguentemente il 
# polinomio interpolatore esiste sempre ed è unico.
# Al crescere del numero dei punti di interpolazione, e quindi del grado del 
# polinomio interpolatore non si ha la convergenza del polinomio interpolatore 
# alla funzione che ha generato i dati: ha al centro dell’intervallo una buona 
# approssimazione e delle fitte oscillazioni agli estremi.
def plagr(x_nodi, k):
    x_zeri = np.zeros_like(x_nodi)
    n = x_nodi.size
    if k == 0:
        x_zeri = x_nodi[1:n]
    else:
        x_zeri = np.append(x_nodi[0 : k], x_nodi[k + 1 : n])

    num = np.poly(x_zeri)
    den = np.polyval(num, x_nodi[k])

    # Il polinomio interpolatore è unico
    p = num / den

    return p


# Polinomio di Lagrange da una set di punti
#
# Costruisce n+1 polinomi di Lagrange che rappresentano una base per lo spazio
# dei polinomi di grado <= n: ai coeffienti di questi polinomi corrispondono una
# matrice identità e il vettore soluzione.
def InterpL(x, f, xx):
    n = x.size
    m = xx.size
    L = np.zeros((m, n))
    for k in range(n):
        p = plagr(x, k)
        # Il polinomio di Lagrange k-esimo valutato nei punti xx.
        # La costante di Lebesgue risulta essere il coefficiente di amplificazione degli errori 
        # relativi sui dati e pertanto identifica il numero di condizionamento del problema 
        # di interpolazione polinomiale
        L[:, k] = np.polyval(p, xx)

    return np.dot(L, f)
