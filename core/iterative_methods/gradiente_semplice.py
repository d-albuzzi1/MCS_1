import numpy as np
import time

def gradiente_semplice(A, b, tol, max_iter):
    """
    Risolve il sistema lineare Ax = b con il metodo iterativo del Gradiente Semplice.
    """
    x = np.zeros_like(b, dtype=float)
    r = b - A @ x

    start_time = time.perf_counter()

    for k in range(1, max_iter + 1):

        Ar = A @ r

        r_dot = np.dot(r, r)
        alpha = r_dot / np.dot(r, Ar)

        x = x + alpha * r
        r = r - alpha * Ar

        # Calcola l'errore relativo: ||r|| / ||b||
        error = np.linalg.norm(r) / np.linalg.norm(b)

        # Verifica la convergenza
        if error < tol:
            end_time = time.perf_counter()
            return x, k, True, end_time - start_time

    # Se il metodo non converge entro max_iter, ritorna allo stesso
    end_time = time.perf_counter()
    return x, max_iter, False, end_time - start_time
