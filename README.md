# CS 6220 – Homework 3: k-NN, Neural Networks, Bias–Variance

## Contents

| Path | Description |
|---|---|
| `HW3.ipynb` | Main solution notebook: k-NN (`knn_predict`, visualization, training/LOOCV error), perceptron (`train_perceptron`), single neuron (`train_neuron`), all experiments for report questions 3–9 (with outputs and animations) |
| `HW3-gradient_flow_mnist.ipynb` | Demo notebook from Canvas (unchanged) |
| `mnist_runs/` | Executed copies of the demo notebook for Q10–Q12 (baseline + activation/cost variants), their plots and training logs |
| `figures/` | All figures produced by `HW3.ipynb` |
| `results/results.json` | Numbers quoted in the report |
| `requirements.txt` | Python dependencies |

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# CPU-only PyTorch is sufficient:  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Running

1. `jupyter notebook HW3.ipynb` and *Run All* (about 2–3 minutes on a laptop CPU). The notebook writes
   `figures/*.png` and `results/results.json`.
2. Q10–Q12: open `HW3-gradient_flow_mnist.ipynb` and *Run All* (MNIST is downloaded into `./data` on the first run).
   The variants in `mnist_runs/` were obtained by changing only `ACTIVATIONS` and `COST_FUNCTION` in the
   parameter cell:

   | Run | `ACTIVATIONS` | `COST_FUNCTION` |
   |---|---|---|
   | 1 baseline | `['sigmoid'] * 5` | `'mse'` |
   | 2 | `['sigmoid'] * 4 + ['identity']` | `'cross_entropy'` |
   | 3 | `['relu'] * 4 + ['identity']` | `'cross_entropy'` |
   | 4 | `['relu'] * 4 + ['sigmoid']` | `'mse'` |

   All other parameters as in the demo: `HIDDEN_UNITS = [32, 32, 32, 32]`, SGD, lr 0.05, batch 128, 60 epochs, L1 gradient norm normalised by layer size.
