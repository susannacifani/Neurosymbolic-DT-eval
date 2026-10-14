import re
import sys
from pathlib import Path

import numpy as np

# Uso (dalla cartella NRM): python3 Grounding/print_grounding_table.py [seed]
#   seed: es. 0,1,2 (default: tutti i seed trovati)
# Stampa in un colpo solo tutte e 4 le configurazioni (Symbolic/Image x distance/three_value),
# una tabella per configurazione, tutti i 10 task.
#
# In ogni cella due numeri, entrambi sul test set:
#   balanced accuracy / symbol accuracy
# - balanced accuracy: il grounder + automa prevede bene il reward? Media delle recall per
#   classe (le classi di reward sono sbilanciate, "vittoria" e' rara, quindi l'accuracy
#   grezza non e' affidabile)
# - symbol accuracy: il grounder riconosce il simbolo giusto nelle caselle? Calcolata sui
#   simboli della formula (gli oggetti non usati contano come vuoto) e a meno delle
#   simmetrie della formula, cioe' solo su cio' che dai reward si puo' davvero distinguere
# Ogni numero e' media +- deviazione standard tra i seed (con un solo seed c'e' solo la media).

# ordine dei blocchi: prima Symbolic, poi Image sotto;
# dentro ciascuno, da sinistra a destra, prima three_value poi distance
CONFIGS = [
    ("Symbolic", "three_value"),
    ("Symbolic", "distance"),
    ("Image", "three_value"),
    ("Image", "distance"),
]

# ordine delle colonne, da sinistra a destra (nome file -> etichetta mostrata)
DATASETS = [("medium_expert", "expert"), ("medium", "medium"), ("medium_replay", "replay")]
NUM_TASKS = 10

# risultati del grounding: Grounding/<State>/<reward>/task<N>/seed<S>/log/log_<dataset>.txt
GROUNDING_ROOT = Path(__file__).resolve().parent

BALANCED_PATTERN = re.compile(r"BALANCED ACCURACY FINALE GLOBALE SUL TEST SET:\s*([0-9.]+)%")
SYMBOL_PATTERN = re.compile(r"SYMBOL ACCURACY SUL TEST SET \(simboli della formula.*?:\s*([0-9.]+)%")


def find_seeds():
    return sorted({int(p.name[4:]) for p in GROUNDING_ROOT.glob("*/*/task*/seed*") if p.name[4:].isdigit()})


SEEDS = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else find_seeds()


def parse_log(path):
    """(balanced, symbol) di un log, oppure None se il run non e' (ancora) completo."""
    if not path.exists():
        return None
    content = path.read_text(encoding="utf-8")
    balanced = BALANCED_PATTERN.search(content)
    symbol = SYMBOL_PATTERN.search(content)
    if not (balanced and symbol):
        return None
    return float(balanced.group(1)), float(symbol.group(1))


def mean_std(values):
    if len(values) == 1:
        return f"{values[0]:.1f}"
    # deviazione standard campionaria tra i seed
    return f"{np.mean(values):.1f} ± {np.std(values, ddof=1):.1f}"


def cell(log_paths):
    """'balanced / symbol' (media +- std tra i seed), numero di seed trovati."""
    results = [r for r in (parse_log(p) for p in log_paths) if r is not None]
    if not results:
        return "N/A", 0
    return mean_std([r[0] for r in results]) + " / " + mean_std([r[1] for r in results]), len(results)


INCOMPLETE = []
COL_GAP = 2     # spazi tra le colonne di uno stesso reward (expert / medium / replay)
GROUP_GAP = 3   # spazi tra three_value e distance
SUBHEADER = "balanced / symbol"


def collect(state_type, reward_type):
    """testo di ogni cella (riga -> dataset) e riga 'media' sui task (delle medie tra seed)"""
    base_path = GROUNDING_ROOT / state_type / reward_type
    cells = {f"task{t}": {} for t in range(1, NUM_TASKS + 1)}
    cells["media"] = {}
    for file_name, label in DATASETS:
        task_means = []
        for task_id in range(1, NUM_TASKS + 1):
            log_paths = [base_path / f"task{task_id}" / f"seed{seed}" / "log" / f"log_{file_name}.txt" for seed in SEEDS]
            cells[f"task{task_id}"][label], n = cell(log_paths)
            if 0 < n < len(SEEDS):
                INCOMPLETE.append((state_type, reward_type, task_id, file_name))
            results = [r for r in (parse_log(p) for p in log_paths) if r is not None]
            if results:
                task_means.append((np.mean([r[0] for r in results]), np.mean([r[1] for r in results])))
        cells["media"][label] = (f"{np.mean([m[0] for m in task_means]):.1f} / {np.mean([m[1] for m in task_means]):.1f}"
                                 if task_means else "N/A")
    return cells


def centered_at(line, text, center):
    """aggiunge 'text' a 'line' centrandolo sulla colonna 'center' (senza sovrapporsi)"""
    start = max(center - len(text) // 2, len(line) + 1 if line else 0)
    return line.ljust(start) + text


def format_block(state_type):
    """tabella di uno stato: three_value e distance affiancati, nomi centrati sopra le colonne"""
    rewards = [rt for st, rt in CONFIGS if st == state_type]
    data = {rt: collect(state_type, rt) for rt in rewards}
    row_labels = list(next(iter(data.values())).keys())
    columns = [(rt, label) for rt in rewards for _, label in DATASETS]

    # ogni cella e' allineata sulla barra: balanced a destra della sua parte, symbol a sinistra
    def halves(text):
        return text.split(" / ") if " / " in text else (text, "")
    left = {c: max(len("balanced"), *(len(halves(data[c[0]][r][c[1]])[0]) for r in row_labels)) for c in columns}
    right = {c: max(len("symbol"), *(len(halves(data[c[0]][r][c[1]])[1]) for r in row_labels)) for c in columns}

    def aligned(col, text):
        a, b = halves(text)
        return a.rjust(left[col]) + (" / " + b.ljust(right[col]) if b else " " * (3 + right[col]))

    widths = {c: left[c] + 3 + right[c] for c in columns}
    first_width = max(len("Task"), *(len(r) for r in row_labels))

    # posizione di inizio di ogni colonna
    positions, pos = {}, first_width
    for i, col in enumerate(columns):
        pos += GROUP_GAP if i % len(DATASETS) == 0 else COL_GAP
        positions[col] = pos
        pos += widths[col]
    total = pos
    center = lambda col: positions[col] + widths[col] // 2

    # Symbolic / Image a sinistra; three_value e distance centrati sopra la colonna medium
    state_line = state_type
    reward_line = ""
    for rt in rewards:
        reward_line = centered_at(reward_line, rt, center((rt, DATASETS[len(DATASETS) // 2][1])))
    dataset_line, sub_line = "Task".ljust(first_width), "".ljust(first_width)
    for col in columns:
        dataset_line = centered_at(dataset_line, col[1], center(col))
        sub_line = sub_line.ljust(positions[col]) + aligned(col, SUBHEADER)

    lines = [state_line, reward_line, dataset_line, sub_line, "-" * total]
    for r in row_labels:
        if r == "media":
            lines.append("-" * total)
        line = r.ljust(first_width)
        for col in columns:
            line = line.ljust(positions[col]) + aligned(col, data[col[0]][r][col[1]])
        lines.append(line)
    return "\n".join(lines)


def main():
    blocks = [format_block(state) for state in dict.fromkeys(st for st, _ in CONFIGS)]

    print(f"GROUNDING - seed {', '.join(map(str, SEEDS))} - test set")
    print("In ogni cella: balanced accuracy / symbol accuracy")
    print("  balanced accuracy = il grounder + automa prevede bene il reward (media delle recall per classe)")
    print("  symbol accuracy   = il grounder riconosce il simbolo giusto (simboli della formula, a meno delle simmetrie)")
    print("Ogni numero: media ± deviazione standard tra i seed; riga 'media' = media sui task")
    if INCOMPLETE:
        print(f"ATTENZIONE: {len(INCOMPLETE)} combinazioni non hanno ancora tutti i seed (la media usa quelli presenti)")
    for block in blocks:
        print()
        print(block)


if __name__ == "__main__":
    main()
