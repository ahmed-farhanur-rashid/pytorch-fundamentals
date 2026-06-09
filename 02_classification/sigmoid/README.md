# Sigmoid — Binary Classification

Binary classification using a single sigmoid output on the Heart Disease dataset.

## What is Sigmoid?

Sigmoid squashes any number into the range [0, 1]:

```
σ(z) = 1 / (1 + e^(-z))
```

- Output > 0.5 → predict class 1 (has heart disease)
- Output ≤ 0.5 → predict class 0 (no heart disease)

## Dataset

[Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease) from UCI — 303 patients, 13 features, binary target.

| Feature | Description |
|---|---|
| age | Age in years |
| sex | 0 = female, 1 = male |
| cp | Chest pain type (0-3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl (0/1) |
| restecg | Resting ECG results (0-2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina (0/1) |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels (0-3) |
| thal | Thalassemia (0-3) |

**Target:** 0 = no disease, 1 = has disease

## Project Structure

```
sigmoid/
├── data/
├── src/
│   ├── __init__.py
│   ├── data.py          # Loads Heart Disease dataset
│   ├── model.py         # Sigmoid model (13 → 1)
│   ├── loss_fn.py       # BCELoss
│   ├── optimizer.py     # SGD
│   ├── train.py         # Training loop
│   ├── evaluate.py      # Test evaluation
│   ├── plot.py          # Loss/accuracy, confusion matrix, feature importance
│   └── main.py          # Entry point
├── visualization/
└── README.md
```

## Run

```
python src/main.py
```

## Output

- `visualization/loss_acc.png` — training curves
- `visualization/confusion_matrix.png` — test accuracy
- `visualization/feature_importance.png` — which features matter most

## Key Differences from Softmax

| | Sigmoid | Softmax |
|---|---|---|
| Classes | 2 (binary) | N (multi-class) |
| Output | 1 probability | N probabilities (sum to 1) |
| Loss | `BCELoss` | `CrossEntropyLoss` |
| Use case | Yes/No questions | Which digit? |
