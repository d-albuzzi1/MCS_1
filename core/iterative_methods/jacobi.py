import numpy as np
import time

def jacobi(A, b, tol, max_iter):
    """
    Risolve il sistema lineare Ax = b con il metodo iterativo di Jacobi.
    """
    n = len(b)
    x = np.zeros(n)

    D_inv = 1.0 / A.diagonal()

    start_time = time.perf_counter()

    for k in range(max_iter):

        r = A @ x - b
        x_new = x - D_inv * r
        r = A @ x_new - b

        # Calcola l'errore relativo: ||r|| / ||b||
        err_rel = np.linalg.norm(r) / np.linalg.norm(b)

        # Verifica la convergenza
        if err_rel < tol:
            end_time = time.perf_counter()
            return x_new, k + 1, True, end_time - start_time

        x = x_new

    # Se il metodo non converge entro il numero massimo di iterazioni, ritorna allo stesso
    end_time = time.perf_counter()
    return x, max_iter, False, end_time - start_time
