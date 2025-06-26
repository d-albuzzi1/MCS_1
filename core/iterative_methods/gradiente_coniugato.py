import numpy as np
import time

def gradiente_coniugato(A, b, tol, max_iter):
    """
    Risolve il sistema lineare Ax = b con il metodo iterativo del Gradiente Coniugato.
    """
    x = np.zeros_like(b)
    r = b - A @ x
    p = r.copy()

    rs_old = np.dot(r, r)

    start_time = time.perf_counter()

    for k in range(max_iter):

        Ap = A @ p

        alpha = rs_old / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.dot(r, r)

        # Verifica il criterio di arresto usando l'errore relativo
        relative_residual = np.sqrt(rs_new) / np.linalg.norm(b)

        if relative_residual < tol:
            end_time = time.perf_counter()
            return x, k + 1, True, end_time - start_time

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    # Se non converge entro max_iter, ritorna comunque i dati
    end_time = time.perf_counter()
    return x, max_iter, False, end_time - start_time
