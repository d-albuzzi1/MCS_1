import numpy as np
from scipy.sparse import csr_matrix
import time

def gauss_seidel(A, b, tol, max_iter):
    """
    Risolve il sistema lineare Ax = b con il metodo iterativo di Gauss-Seidel.
    """
    # Se A non è una matrice sparse CSR, la converto
    if not isinstance(A, csr_matrix):
        A = csr_matrix(A)

    n = A.shape[0]  # dimensione del sistema
    x = np.zeros(n)  # soluzione iniziale

    start_time = time.perf_counter()

    for k in range(max_iter):
        for i in range(n):

            row_start = A.indptr[i]
            row_end = A.indptr[i + 1]
            Ai = A.indices[row_start:row_end]
            Av = A.data[row_start:row_end]

            sum_ = 0
            diag = None

            for idx, j in enumerate(Ai):
                if j == i:
                    diag = Av[idx]
                else:
                    sum_ += Av[idx] * x[j]

            if diag is None:
                raise ZeroDivisionError("Elemento diagonale mancante in A")

            x[i] = (b[i] - sum_) / diag

        # Calcolo il residuo r = A*x - b e errore relativo
        r = A @ x - b
        error = np.linalg.norm(r) / np.linalg.norm(b)

        # Condizione per fermare l'algoritmo
        if error < tol:
            end_time = time.perf_counter()
            return x, k + 1, True, end_time - start_time

    # Se non convergo entro max_iter, ritorno lo stato finale, con convergenza=false
    end_time = time.perf_counter()
    return x, max_iter, False, end_time - start_time
