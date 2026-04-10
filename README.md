# Neurosymbolic Offline RL: Evaluating Decision Transformers on LTL Tasks

## 📌 Project Overview
This repository contains an evaluation framework designed to test the capabilities and limits of **Decision Transformers (DT)** in solving non-Markovian, temporally extended tasks defined via **Linear Temporal Logic (LTL)**. 

While Decision Transformers show strong performance in standard offline RL settings, this project empirically analyzes their performance gap when dealing with visual **Symbol Grounding** versus pure symbolic states. The experiments are conducted across tasks of increasing complexity (e.g., Sequential Visits, Global Avoidance) using D4RL-style datasets (Expert, Medium, Medium-Replay).

## 🚀 Key Contributions
* **Offline RL Evaluation:** Implementation of a Decision Transformer architecture tailored for grid-world environments.
* **Dataset Generation:** Generation of D4RL-style offline datasets with injected noise ($\epsilon=0.2$) to test Out-Of-Distribution (OOD) generalization.
* **Empirical Analysis:** A comprehensive comparison of DT performance on purely symbolic states versus raw image states, highlighting the critical failure of pure offline RL in visual symbol grounding without neurosymbolic priors.

## ⚙️ Installation and Usage
1. Clone the repository:
   ```bash
   git clone [https://github.com/susannacifani/Neurosymbolic-DT-eval.git](https://github.com/susannacifani/Neurosymbolic-DT-eval.git)
   cd Neurosymbolic-DT-eval
3. Install the required dependencies: pip install -r requirements.txt
4. Run the generation/training/evaluation with main.py, choosing your configuration (Image/Symbolic, three_value/distance): python RL/Env/main.py
5. Check the results by running: python RL/Env/table_results.py

## ⭐​ Acknowledgements & Credits
This evaluation framework builds upon the grid-world environment and LTL-to-Automata baseline logic provided by the authors of Neural Reward Machines.
Original Repository: [KRLGroup / NeuralRewardMachines](https://github.com/KRLGroup/NeuralRewardMachines) 
