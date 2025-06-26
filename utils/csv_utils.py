from pathlib import Path
import plotly.graph_objects as go
import pandas as pd
import csv

def prepare_csv_writer(method_name, timestamp):
    """
    Crea un writer CSV e un file per salvare i risultati di un metodo, con nome univoco basato sul timestamp.
    """
    # Creazione cartella risultati
    method_folder = Path(f"results/{method_name.lower().replace('-', '_')}")
    method_folder.mkdir(parents=True, exist_ok=True)

    csv_path = method_folder / f"__1__{timestamp}.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)

    # Intestazione tabella
    header = ["Metodo", "Matrice", "Tolleranza", "Iterazioni", "Errore Relativo", "Tempo (s)", "Convergenza"]
    csv_writer.writerow(header)

    return csv_file, csv_writer

def csv_as_table(csv_path, output_path):
    """
    Converte un CSV in una tabella visuale e la salva come immagine usando Plotly.
    """
    # Caricamento dei dati dal file CSV in un DataFrame pandas
    df = pd.read_csv(csv_path)

    # ELiminzaione delle colonne "Metodo" e "Matrice" da ripetizioni per leggibilità
    df.loc[df['Metodo'].duplicated(), 'Metodo'] = ""
    df.loc[df['Matrice'].duplicated(), 'Matrice'] = ""

    # Crea una tabella png
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df.columns),
            fill_color='paleturquoise',
            align='center'
        ),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color='lavender',
            align='center'
        )
    )])

    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))

    fig.write_image(output_path)
