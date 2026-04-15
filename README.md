# HemoNet — Brain CT hemorrhage classification

End-to-end pipeline: dataset inspection, stratified splits, hyperparameter search, training two models, evaluation, and a FastAPI service for inference on uploaded images (including DICOM).

## Requirements

- **Python** 3.10+ (project tested with 3.10).
- **GPU** optional; CPU works with smaller batches and longer runs.

### Install dependencies

From the project root (`HemoNet-`):

```powershell
py -m pip install -r requirements.txt
```

For **CPU-only** PyTorch wheels on Windows (smaller download, no CUDA):

```powershell
py -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

On this machine, if `python`, `pip`, or `uvicorn` are not recognized in PowerShell, use the launcher and module form:

- `py script.py`
- `py -m pip install ...`
- `py -m uvicorn api:app --reload --port 8000`

---

## Libraries (from `requirements.txt`)

| Package | Role in this project |
|--------|------------------------|
| **pandas** | CSV labels and results tables |
| **numpy** | Arrays and image/tensor math |
| **matplotlib** | Plots (samples, splits, training curves, evaluation figures) |
| **scikit-learn** | `train_test_split`, metrics (confusion matrix, ROC, report) |
| **Pillow** | Load and resize images (`PIL.Image`) |
| **optuna** | Bayesian-style hyperparameter search |
| **torch** | Training, inference, tensors |
| **torchvision** | `transforms`, pretrained **ResNet50** weights |
| **fastapi** | HTTP API (`api.py`) |
| **uvicorn** | ASGI server to run the API |
| **python-multipart** | File upload handling for FastAPI |
| **pydicom** | Read `.dcm` uploads in the API |

Standard library only in several scripts: `os`, `io`, `time`, `pathlib`, `typing`, `itertools`, `random`, etc.

---

## Models (defined in `models.py`)

| Model | Description |
|-------|-------------|
| **ResNet50** | `build_resnet50()`: ImageNet-pretrained ResNet50, backbone frozen by default, binary head (sigmoid) for hemorrhage vs normal. |
| **CustomCNN** | `CustomCNN`: Smaller convolutional network trained from scratch; same input size and sigmoid output as ResNet for fair comparison. |

Weights after training are saved under `saved_models/` (e.g. `ResNet50.pth`, `CustomCNN.pth`) and loaded by `testing_models.py` and `api.py`.

---

## Data layout

Paths are centralized in `dataset_preparation.py`:

- **`ct_dataset/labels.csv`** — labels (`id`, `hemorrhage`, etc.).
- **`ct_dataset/head_ct/`** — PNG slices named to match the CSV (e.g. `000.png`).

Running **`dataset_split.py`** writes:

- `split_train.csv`, `split_val.csv`, `split_test.csv`

Ensure these exist before training, tuning, evaluation, or the API (which needs trained checkpoints in `saved_models/`).

---

## Project structure (main files)

| File | Purpose |
|------|---------|
| `dataset_preparation.py` | Dataset class, transforms, paths; step (a) inspection when run as main |
| `dataset_split.py` | Stratified train/val/test split and DataLoader smoke test |
| `models.py` | `CustomCNN`, `build_resnet50`; optional `__main__` sanity check |
| `hyperparams_tuning.py` | Grid / random / Optuna search; writes `hyperparameter_results_*.csv` |
| `training_models.py` | Train both models, curves, save `saved_models/*.pth` |
| `testing_models.py` | Test-set metrics, plots, misclassification gallery |
| `api.py` | FastAPI app: upload images → both models’ predictions |
| `requirements.txt` | Pinned minimum versions for all third-party packages |

---

## How to run the pipeline (step by step)

Run commands from the repository root. Order matters.

### 1. Dataset preparation (Step a)

Inspect data, transforms, sample figure:

```powershell
py dataset_preparation.py
```

### 2. Train / validation / test split (Step c)

Builds split CSVs and checks loaders:

```powershell
py dataset_split.py
```

### 3. Models sanity check (optional)

```powershell
py models.py
```

### 4. Hyperparameter tuning (Step e)

Uses existing `split_train.csv` / `split_val.csv`. Set `MODEL_NAME` in the file if you need **CustomCNN** results too.

```powershell
py hyperparams_tuning.py
```

### 5. Training (Step f)

Trains ResNet50 and CustomCNN; saves weights and training curve PNGs.

```powershell
py training_models.py
```

### 6. Evaluation on test set (Step g)

```powershell
py testing_models.py
```

Outputs typically include `evaluation_results/` (figures, CSVs, misclassified samples).

### 7. API server (Step h)

Requires `saved_models/ResNet50.pth` and `saved_models/CustomCNN.pth`.

```powershell
py -m uvicorn api:app --reload --port 8000
```

Then open the interactive docs at **http://127.0.0.1:8000/docs** (default FastAPI Swagger UI).

---

## License / course context

This repository is structured for an academic deep-learning project (brain CT hemorrhage classification). Adjust paths, hyperparameters, and report filenames to match your assignment brief if they differ.
