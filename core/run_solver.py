from utils.metrics import relative_error

def run_solver(name, solver_func, A, b, x_exact, tol, max_iter, csv_writer, matrix_name):
    """
    Esegue un metodo iterativo su un sistema Ax = b,
    valuta l'errore relativo e stampa i risultati e li salva su file CSV.
    """
    # Esecuzione del metodo iterativo con i risultati
    x_approx, iterations, converged, time_taken = solver_func(A, b, tol, max_iter)

    # Calcolo dell'errore relativo tra la soluzione esatta e quella approssimata
    rel_err = relative_error(x_exact, x_approx)

    result_summary = (
        f"{name:25} | Matrice: {matrix_name:8} | Iterazioni: {iterations:6} | "
        f"Errore relativo: {rel_err:.2e} | Tempo: {time_taken:.4f}s | Convergenza: {converged}"
    )
    print(result_summary)

    # Scrittura dati su csv
    csv_writer.writerow([
        name,
        matrix_name,
        tol,
        iterations,
        f"{rel_err:.2e}",
        f"{time_taken:.4f}",
        converged
    ])
