import pandas as pd
import matplotlib.pyplot as plt

# Lista delle tolleranze ordinate per l’asse x nei grafici
TOL_LIST = ["1e-04", "1e-06", "1e-08", "1e-10"]

# Metriche da rappresentare nei grafici
METRICS = ["Iterazioni", "Errore Relativo", "Tempo (s)"]

def load_all_results(csv_dir):
    """
    Carica i file CSV da ogni metodo e li unisce in un singolo DataFrame.
    """
    all_data = []

    for method_folder in csv_dir.iterdir():
        if not method_folder.is_dir():
            continue

        # Prendo il file CSV più recente
        latest_csvs = sorted(method_folder.glob("__1__*.csv"), reverse=True)
        if not latest_csvs:
            continue

        # Carico il CSV e lo aggiungo alla lista dati
        df = pd.read_csv(latest_csvs[0])
        all_data.append(df)

    # Concateno tutti i dataframe in uno solo
    return pd.concat(all_data, ignore_index=True)

def plot_metrics_per_matrix(df, method_name, output_dir):
    """
    Genera grafici a barre per ciascuna metrica e matrice di un metodo specifico.
    """
    # Filtra il dataframe per il metodo specificato
    df = df[df["Metodo"] == method_name].copy()

    # Ordina la colonna tolleranza come categoria
    df["Tolleranza_str"] = pd.Categorical(df["Tolleranza"].astype(str), categories=TOL_LIST, ordered=True)

    matrices = df["Matrice"].unique()
    colors = ['#e8e833', '#6cc548', '#51d6d8', '#e8a647']

    # Ciclo su ogni matrice per la creazione di un grafico
    for idx, matrix in enumerate(matrices):
        df_matrix = df[df["Matrice"] == matrix]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"{method_name} - Matrice: {matrix}", fontsize=16)

        color = colors[idx % len(colors)]

        # Creazione per ogni metrica del grafo
        for i, metric in enumerate(METRICS):
            ax = axes[i]
            df_metric = df_matrix[["Tolleranza_str", metric]].dropna().sort_values("Tolleranza_str")

            y = df_metric[metric].astype(float).values
            x = df_metric["Tolleranza_str"].astype(str).values

            ax.bar(x, y, color=color)
            ax.set_title(metric)
            ax.set_xlabel("Tolleranza")
            ax.set_ylabel(metric)

            # Scala logaritmica per errore e tempo
            if metric in ["Errore Relativo", "Tempo (s)"]:
                ax.set_yscale("log")

            ax.grid(True, axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        method_dir = output_dir / method_name.lower()
        method_dir.mkdir(parents=True, exist_ok=True)

        plot_path = method_dir / f"{matrix.lower()}_summary.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Salvato correttamente: {plot_path}")
