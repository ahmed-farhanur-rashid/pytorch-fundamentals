# Linear Regression with PyTorch

A from-scratch implementation of linear regression on the Diabetes dataset. Covers the full ML workflow: data loading, normalization, train/val/test splitting, learning rate search, training, and loss visualization.

---

## Project Structure

```
linear-regression-using-pytorch/
├── data/
│   └── diabetes.csv
├── src/
│   ├── __init__.py
│   ├── data.py         # Dataset class, train/val/test splitting, DataLoader
│   ├── evaluate.py     # Evaluation (no_grad, returns average loss)
│   ├── loss_fn.py      # MSE loss
│   ├── main.py         # Main entry point
│   ├── model.py        # LinearRegression (nn.Module wrapper around nn.Linear)
│   ├── optimizer.py    # SGD optimizer
│   ├── train.py        # Training loop with optional validation
│   ├── plot.py         # Loss curve and LR search plots
│   └── train.py        # Training loop with optional validation
├── visualization/      # Saved plots land here
├── README.md
└── LICENSE
```

---

## Dataset

The [Diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) has 442 patients, 10 features (age, sex, BMI, blood pressure, 6 blood serum values), and one continuous target representing disease progression one year after baseline. The target is min-max normalized to [0, 1] before training.

---

## Setup

```bash
git clone https://github.com/yourusername/linear-regression-using-pytorch.git
cd linear-regression-using-pytorch

python -m venv .venv
source .venv/bin/activate

pip install torch pandas numpy matplotlib
```

---

## Usage

```bash
python src/main.py
```

The script:
1. Loads and normalizes the dataset (80/10/10 split)
2. Trains a linear model for each learning rate in `[1e-3, 5e-3, 1e-2, 5e-2, 1e-1]` for 100 epochs
3. Picks the best learning rate based on final validation loss
4. Saves two plots to `visualization/`

---

## Outputs

`visualization/loss.png` — training and validation loss over 100 epochs for the best learning rate.

`visualization/lr_search.png` — final train/val loss for each candidate learning rate on a log scale.

---

## License

MIT — see `LICENSE`.