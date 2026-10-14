import re
import sys
from pathlib import Path

import numpy as np

# Uso (dalla cartella NRM): python3 WorldModel/print_world_model_table.py [metrica] [seed]
#   metrica:  acc      (default) - accuracy di posizione del next state sul test set (modello)
#             baseline           - accuracy di posizione della baseline identity (z' = z)
#             mse                - latent MSE sul test set (modello)
#             recon              - miglior reconstruction MSE dell'autoencoder sul validation
#   seed:     es. 0,1,2 (default: tutti i seed trovati)
# In ogni cella media +- deviazione standard tra i seed. Con un solo seed c'e' solo la media.

METRIC = sys.argv[1] if len(sys.argv) > 1 else "acc"

# risultati: WorldModel/<State>/task<N>/seed<S>/log/log_<dataset>.txt
# (il world model non usa il reward, quindi non c'e' distinzione distance / three_value)
BASE_PATH = Path(__file__).resolve().parent

# Symbolic e Image affiancati sulla stessa riga
STATE_TYPES = ["Symbolic", "Image"]
COL_GAP = 3     # spazi tra le colonne di uno stesso stato (expert / medium / replay)
GROUP_GAP = 6   # spazi tra Symbolic e Image
# spiegazione della metrica mostrata, stampata sopra la tabella
DESCRIPTIONS = {
    "acc": "accuracy di posizione sul test (%): dato lo stato attuale e l'azione, il modello prevede lo stato\n"
           "successivo nello spazio latente; lo si decodifica e si guarda a quale casella corrisponde.\n"
           "E' la percentuale di transizioni in cui la casella prevista e' quella giusta.",
    "baseline": "accuracy di posizione della baseline identity (z' = z, cioe' 'resto fermo') sul test (%):\n"
                "indovina solo quando l'agente sbatte contro un muro e non si muove.",
    "mse": "latent MSE sul test: errore quadratico medio tra lo stato latente previsto e quello vero.\n"
           "Dipende dalla scala dello spazio latente: non confrontare Symbolic con Image.",
    "recon": "reconstruction MSE dell'autoencoder (migliore sul validation): quanto bene si ricostruisce\n"
             "l'osservazione partendo dallo stato latente.",
}
DATASETS = [("medium_expert", "expert"), ("medium", "medium"), ("medium_replay", "replay")]
NUM_TASKS = 10

PATTERNS = {
    "acc": re.compile(r"Next-state position accuracy - model:\s*([0-9.]+)%"),
    "baseline": re.compile(r"Next-state position accuracy - model:.*baseline \(z'=z\):\s*([0-9.]+)%"),
    "mse": re.compile(r"Latent MSE - model:\s*([0-9.]+)"),
    "recon": re.compile(r"\[AE\] Best val reconstruction MSE:\s*([0-9.]+)"),
}
# cifre decimali: le MSE sono numeri molto piccoli
DECIMALS = 6 if METRIC in ("mse", "recon") else 2


def find_seeds():
    return sorted({int(p.name[4:]) for p in BASE_PATH.glob("*/task*/seed*") if p.name[4:].isdigit()})


SEEDS = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else find_seeds()


def parse_log(path):
    if not path.exists():
        return None
    match = PATTERNS[METRIC].search(path.read_text(encoding="utf-8"))
    return float(match.group(1)) if match else None


def cell(state_type, task, dataset):
    """'media +- std' tra i seed disponibili, numero di seed trovati."""
    values = [v for v in (parse_log(BASE_PATH / state_type / f"task{task}" / f"seed{s}" / "log" / f"log_{dataset}.txt")
                          for s in SEEDS) if v is not None]
    if not values:
        return "N/A", 0
    mean = f"{np.mean(values):.{DECIMALS}f}"
    if len(values) == 1:
        return mean, 1
    # deviazione standard campionaria tra i seed
    return f"{mean} ± {np.std(values, ddof=1):.{DECIMALS}f}", len(values)


def centered_at(line, text, center):
    """aggiunge 'text' a 'line' centrandolo sulla colonna 'center' (senza sovrapporsi)"""
    start = max(center - len(text) // 2, len(line) + 1 if line else 0)
    return line.ljust(start) + text


def main():
    if METRIC not in PATTERNS:
        print(f"Metrica '{METRIC}' non valida. Scegli tra: {', '.join(PATTERNS)}")
        return

    # celle: riga -> (stato, dataset) -> testo, piu' la riga 'media' sui task
    row_labels = [f"task{t}" for t in range(1, NUM_TASKS + 1)] + ["media"]
    columns = [(st, label) for st in STATE_TYPES for _, label in DATASETS]
    cells, incomplete = {r: {} for r in row_labels}, 0
    for st in STATE_TYPES:
        for file_name, label in DATASETS:
            task_means = []
            for task in range(1, NUM_TASKS + 1):
                text, n = cell(st, task, file_name)
                cells[f"task{task}"][(st, label)] = text
                incomplete += 0 < n < len(SEEDS)
                vals = [v for v in (parse_log(BASE_PATH / st / f"task{task}" / f"seed{s}" / "log" / f"log_{file_name}.txt")
                                    for s in SEEDS) if v is not None]
                if vals:
                    task_means.append(np.mean(vals))
            cells["media"][(st, label)] = f"{np.mean(task_means):.{DECIMALS}f}" if task_means else "N/A"

    widths = {c: max(len(c[1]), *(len(cells[r][c]) for r in row_labels)) for c in columns}
    first_width = max(len("Task"), *(len(r) for r in row_labels))
    positions, pos = {}, first_width
    for i, c in enumerate(columns):
        pos += GROUP_GAP if i % len(DATASETS) == 0 else COL_GAP
        positions[c] = pos
        pos += widths[c]
    total = pos
    center = lambda c: positions[c] + widths[c] // 2

    # Symbolic / Image centrati sopra la colonna medium, nomi dei dataset centrati sulle colonne
    state_line = ""
    for st in STATE_TYPES:
        state_line = centered_at(state_line, st, center((st, DATASETS[len(DATASETS) // 2][1])))
    dataset_line = "Task".ljust(first_width)
    for c in columns:
        dataset_line = dataset_line.ljust(positions[c]) + c[1].center(widths[c])

    print(f"World model - metrica: {METRIC} - seed {', '.join(map(str, SEEDS))} - test set")
    print(DESCRIPTIONS[METRIC])
    print("In ogni cella: media ± deviazione standard tra i seed; riga 'media' = media sui task")
    if incomplete:
        print(f"ATTENZIONE: {incomplete} combinazioni non hanno ancora tutti i seed (la media usa quelli presenti)")
    print()
    print(state_line)
    print(dataset_line)
    print("-" * total)
    for r in row_labels:
        if r == "media":
            print("-" * total)
        line = r.ljust(first_width)
        for c in columns:
            line = line.ljust(positions[c]) + cells[r][c].center(widths[c])
        print(line)


if __name__ == "__main__":
    main()
