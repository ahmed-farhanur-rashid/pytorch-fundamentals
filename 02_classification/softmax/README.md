# Softmax — Multi-Class Classification

Multi-class classification using softmax for MNIST digit recognition.

## What is Softmax?

Softmax converts raw scores (logits) into probabilities that sum to 1:

```
softmax(z_i) = exp(z_i) / Σ exp(z_j)
```

Each output is in [0, 1], and all outputs sum to 1. The class with the highest probability is the prediction.

## Dataset

[MNIST](http://yann.lecun.com/exdb/mnist/) — 28×28 grayscale images of handwritten digits (0-9).

- 60,000 training images
- 10,000 test images
- 10 classes (digits 0-9)

## Project Structure

```
softmax/
├── data/
├── src/
│   ├── __init__.py
│   ├── data.py          # Loads MNIST, splits train/val
│   ├── model.py         # SoftMax model (784 → 10)
│   ├── loss_fn.py       # CrossEntropyLoss
│   ├── optimizer.py     # SGD
│   ├── train.py         # Training loop
│   ├── evaluate.py      # Test evaluation
│   ├── plot.py          # Loss/accuracy, weights, confusion matrix
│   └── main.py          # Entry point
├── visualization/
└── README.md
```

## Run

```
python src/main.py
```

## Output

- `visualization/loss_acc.png` — training curves (loss + accuracy)
- `visualization/weights.png` — learned weights per digit class
- `visualization/confusion_matrix.png` — test accuracy
- `visualization/sample_predictions.png` — visual check on predictions

## Key Differences from Sigmoid

| | Sigmoid | Softmax |
|---|---|---|
| Classes | 2 (binary) | N (multi-class) |
| Output | 1 probability | N probabilities (sum to 1) |
| Loss | `BCELoss` | `CrossEntropyLoss` |
| Use case | Yes/No questions | Which digit? |
