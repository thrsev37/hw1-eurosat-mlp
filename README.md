# HW1 EuroSAT NumPy MLP

This project implements a three-layer MLP classifier for EuroSAT RGB land-cover classification using NumPy only. It does not use PyTorch, TensorFlow, JAX, or any automatic differentiation framework.

## Submission Links

- GitHub repository: https://github.com/thrsev37/hw1-eurosat-mlp
- Model weights: https://github.com/thrsev37/hw1-eurosat-mlp/raw/main/artifacts/best_model.npz
- Report: `report.pdf` or `report.html`

## Files

- `src/hw1_mlp/data.py`: data discovery, stratified train/validation/test split, image loading, resizing, flattening, and standardization.
- `src/hw1_mlp/layers.py`: `Linear`, `ReLU`, `Sigmoid`, `Tanh`, and cross-entropy loss with manual `forward` and `backward`.
- `src/hw1_mlp/model.py`: `ThreeLayerMLP` model and checkpoint save/load.
- `src/hw1_mlp/optim.py`: SGD optimizer and learning-rate decay.
- `train.py`: final training loop and best-checkpoint saving.
- `search.py`: grid search over hidden dimension, activation, learning rate, and weight decay.
- `evaluate.py`: test accuracy and confusion matrix.
- `visualize.py`: loss/accuracy curves, confusion matrix plot, first-layer weight visualization, and error examples.
- `make_report.py`: generates `report.html` and `report.pdf`.

## Environment

```bash
python3 -m pip install numpy pillow matplotlib reportlab
```

## Run

Set the data path to the provided EuroSAT folder:

```bash
DATA_DIR=/Users/cheese/Downloads/hw1/EuroSAT_RGB
```

Run hyperparameter search:

```bash
python3 search.py --data-dir "$DATA_DIR"
```

Train the final model:

```bash
python3 train.py --data-dir "$DATA_DIR" --hidden-dim 128 --activation relu --lr 0.05 --weight-decay 0.0001 --epochs 20
```

Evaluate and generate figures:

```bash
python3 evaluate.py --data-dir "$DATA_DIR"
python3 visualize.py --data-dir "$DATA_DIR"
```

Generate reports:

```bash
python3 make_report.py \
  --github-url "https://github.com/thrsev37/hw1-eurosat-mlp" \
  --weights-url "https://github.com/thrsev37/hw1-eurosat-mlp/raw/main/artifacts/best_model.npz"
```

## Outputs

- Best model checkpoint: `artifacts/best_model.npz`
- Training history: `artifacts/history.csv`
- Test metrics: `artifacts/test_metrics.json`
- Confusion matrix: `artifacts/confusion_matrix.csv`
- Figures: `artifacts/figures/`
- Report: `report.html` and `report.pdf`

Before final submission, make sure `report.pdf` contains the public GitHub repository link and the model weight download link.
