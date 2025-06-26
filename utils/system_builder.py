import numpy as np
from utils.matrix_loader import load_matrix

def prepare_system(file_path):
    """
    Prepara un sistema lineare Ax = b per il test, caricando A da file
    e generando b con una soluzione esatta x composta da soli 1.
    """
    # Carica la matrice A da file
    A = load_matrix(file_path)

    # Crea un vettore soluzione esatta composto da tutti 1
    x_exact = np.ones(A.shape[0])

    b = A @ x_exact

    return A, b, x_exact
