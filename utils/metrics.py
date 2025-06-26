import numpy as np

def relative_error(x_exact, x_approx):
    """
    Calcola l'errore relativo tra la soluzione esatta e quella approssimata.
    """
    # Calcola la norma del vettore differenza
    diff_norm = np.linalg.norm(x_exact - x_approx)

    # Calcola la norma della soluzione esatta
    exact_norm = np.linalg.norm(x_exact)

    error_rel = diff_norm / exact_norm

    return error_rel
