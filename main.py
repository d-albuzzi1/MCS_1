from core.run_solver import run_solver
from core.iterative_methods.jacobi import jacobi
from core.iterative_methods.gradiente_semplice import gradiente_semplice
from core.iterative_methods.gauss_seidel import gauss_seidel
from core.iterative_methods.gradiente_coniugato import gradiente_coniugato

from utils.csv_utils import prepare_csv_writer, csv_as_table
from utils.system_builder import prepare_system
from utils.graph_plot_utils import load_all_results, plot_metrics_per_matrix

from pathlib import Path
from datetime import datetime


def run_method(method_name, solver_func, matrix_files, tol_list, max_iter, timestamp):
    """
    Esegue un metodo iterativo su una lista di matrici e tolleranze,
    scrivendo i risultati su file CSV e generando una tabella immagine per csv.
    """
    csv_file, csv_writer = prepare_csv_writer(method_name, timestamp)
    print(f"\n>>> Metodo: {method_name}\n")

    for name in matrix_files:
        print(f"\n===  Matrice: {name}.mtx ===")

        # Costruisce il sistema lineare A, b, x_exact
        A, b, x_exact = prepare_system(Path(f"data/{name}.mtx"))

        for tol in tol_list:
            print(f"\n Tolleranza: {tol}")

            # Esegue il solver e salvando i risultati nel CSV
            run_solver(method_name, solver_func, A, b, x_exact, tol, max_iter, csv_writer, name)

    csv_file.close()
    table_img_path = Path(csv_file.name).with_suffix('.png')
    csv_as_table(csv_file.name, table_img_path)


def main():
    """
    Esegue tutti i metodi iterativi su tutte le matrici e tolleranze specificate.
    Al termine genera grafici riepilogativi dai risultati salvati.
    """

    tol_list = [1e-4, 1e-6, 1e-8, 1e-10]
    max_iter = 20000
    matrix_files = ["spa1", "spa2", "vem1", "vem2"]
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    methods = [
        ("Jacobi", jacobi),
        ("Gauss-Seidel", gauss_seidel),
        ("Gradiente Coniugato", gradiente_coniugato),
        ("Gradiente Semplice", gradiente_semplice),
    ]

    # Esecuzione per ogni metodo
    for method_name, solver in methods:
        run_method(method_name, solver, matrix_files, tol_list, max_iter, now)

    # Crea e salva i grafici per ogni metodo
    CSV_DIR = Path("results")
    OUTPUT_DIR = Path("results/plots")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_all_results(CSV_DIR)

    for method in [name for name, _ in methods]:
        plot_metrics_per_matrix(df, method, OUTPUT_DIR)

if __name__ == "__main__":
    main()
