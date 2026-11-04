# Neurosymbolic Offline RL on LTL Tasks

Offline reinforcement learning on tasks specified in Linear Temporal Logic (LTL), extending [Neural Reward Machines](https://github.com/KRLGroup/NeuralRewardMachines) (NRM) to the Decision Transformer (DT). The grounder learns the LTL symbols from rewards only, and the automaton state it predicts is given to the DT as an extra input.

- `Grid7/`: 7x7 grid with random object layouts (main setting).
- `Grid4/`: 4x4 grid with fixed objects (first version, see [`Grid4/README.md`](Grid4/README.md)).

## Setup

```bash
git clone https://github.com/susannacifani/Neurosymbolic-DT-eval.git
cd Neurosymbolic-DT-eval
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage (Grid7)

Every script has a configuration block at the top: set it and run the script. Completed runs are skipped, so an interrupted launch can be restarted.

| Step | Script |
|------|--------|
| Generate / check the datasets | `Dataset/dataset_generation.py`, `Dataset/check_datasets.py` |
| Train and evaluate the grounder | `Grounding/experiments_grounding.py` |
| Train and evaluate the DT | `DT/experiments_dt.py` |
| DT with the automaton state (from the grounder or the true one) | `DT/experiments_dt_automaton.py` |
| DT with another context length | `DT/experiments_dt_context.py` |
| Result tables (terminal / LaTeX) | `print_*.py`, `latex_*.py` in `DT/` and `Grounding/` |

The grounder must be trained before running the DT with the grounder's automaton state.

Datasets, trained models, logs and result tables are not on git and are written inside each folder.

## Acknowledgements

The environment, the LTL-to-automaton code and the grounder build on [KRLGroup/NeuralRewardMachines](https://github.com/KRLGroup/NeuralRewardMachines) (Umili, Argenziano and Capobianco, ECAI 2024).
