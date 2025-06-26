import numpy as np
from scipy.sparse import coo_matrix
from pathlib import Path

def load_matrix(path: Path):
    """
    Carica una matrice sparsa da file in formato triplette (riga, colonna, valore).
    """
    with open(path, 'r',  encoding="utf-8") as f:
        # leggi dimensioni dalla prima riga
        first_line = f.readline()
        n_rows, n_cols, n_entries = map(int, first_line.strip().split())

        # lettura delletriplette
        data = np.loadtxt(f)

    # split in righe, colonne e valori
    rows = data[:, 0].astype(int) - 1  # da 1-based a 0-based
    cols = data[:, 1].astype(int) - 1
    vals = data[:, 2]

    A = coo_matrix((vals, (rows, cols)), shape=(n_rows, n_cols)).tocsc()

    return A
