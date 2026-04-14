import time
import random
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna
from torch.utils.data import DataLoader

from dataset_preparation import (
    HeadCTDataset,
    DATA_DIR,
    IMG_SIZE,
    build_train_transforms,
    build_val_test_transforms,
)
from models import CustomCNN, build_resnet50

optuna.logging.set_verbosity(optuna.logging.WARNING)  
MAX_EPOCHS  = 30
PATIENCE    = 5
SEED        = 42
MODEL_NAME  = "ResNet50"   #to test the custom cnn, change to "CustomCNN"

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device : {device}")
print(f"Tuning model : {MODEL_NAME}\n")

train_df = pd.read_csv("split_train.csv")
val_df   = pd.read_csv("split_val.csv")

train_transforms = build_train_transforms(IMG_SIZE)
val_transforms   = build_val_test_transforms(IMG_SIZE)

def build_model(model_name):
    """Build and return a fresh model instance."""
    if model_name == "ResNet50":
        model = build_resnet50(freeze_backbone=True)
    else:
        model = CustomCNN()
    return model.to(device)


def train_and_evaluate(lr, batch_size, model_name,
                       max_epochs=MAX_EPOCHS, patience=PATIENCE):
    train_loader = DataLoader(
        HeadCTDataset(train_df, DATA_DIR, transform=train_transforms),
        batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        HeadCTDataset(val_df, DATA_DIR, transform=val_transforms),
        batch_size=batch_size, shuffle=False, num_workers=0)

    model     = build_model(model_name)
    criterion = nn.BCELoss()                        
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    best_val_loss   = float("inf")
    best_val_acc    = 0.0
    best_weights    = None
    epochs_no_improve = 0
    stopped_epoch   = max_epochs
    start_time      = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()                   
            outputs = model(images).squeeze(1)      
            loss    = criterion(outputs, labels)
            loss.backward()                         
            optimizer.step()                        

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():                       
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs  = model(images).squeeze(1)
                val_loss += criterion(outputs, labels).item()
                preds     = (outputs >= 0.5).float()
                correct  += (preds == labels).sum().item()
                total    += labels.size(0)

        val_loss /= len(val_loader)                 
        val_acc   = correct / total                 

        if val_loss < best_val_loss:
            best_val_loss       = val_loss
            best_val_acc        = val_acc
            best_weights        = {k: v.clone()
                                   for k, v in model.state_dict().items()}
            epochs_no_improve   = 0
        else:
            epochs_no_improve  += 1
            if epochs_no_improve >= patience:
                stopped_epoch = epoch
                break                               

    if best_weights:
        model.load_state_dict(best_weights)
    duration = time.time() - start_time
    return {
        "best_val_acc"  : round(best_val_acc * 100, 2),
        "best_val_loss" : round(best_val_loss, 4),
        "stopped_epoch" : stopped_epoch,
        "duration_sec"  : round(duration, 1),
    }


print("=" * 55)
print("METHOD 1 — GRID SEARCH")
print("=" * 55)

grid = {
    "lr"         : [0.01, 0.001, 0.0001],
    "batch_size" : [16, 32],
}


grid_combinations = list(itertools.product(
    grid["lr"], grid["batch_size"]))

grid_results = []
for i, (lr, bs) in enumerate(grid_combinations, 1):
    print(f"  [{i}/{len(grid_combinations)}] lr={lr}, batch={bs} ...", end=" ")
    result = train_and_evaluate(lr, bs, MODEL_NAME)
    result.update({"method": "Grid Search", "lr": lr, "batch_size": bs})
    grid_results.append(result)
    print(f"val_acc={result['best_val_acc']}%  "
          f"stopped_epoch={result['stopped_epoch']}")

best_grid = max(grid_results, key=lambda x: x["best_val_acc"])
print(f"\n  Best grid config → lr={best_grid['lr']}, "
      f"batch={best_grid['batch_size']}, "
      f"val_acc={best_grid['best_val_acc']}%\n")

print("=" * 55)
print("METHOD 2 — RANDOM SEARCH")
print("=" * 55)

LR_OPTIONS = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
BS_OPTIONS = [8, 16, 32]
N_RANDOM   = 6                                      

random.seed(SEED)
random_combinations = [
    (random.choice(LR_OPTIONS), random.choice(BS_OPTIONS))
    for _ in range(N_RANDOM)
]

random_results = []
for i, (lr, bs) in enumerate(random_combinations, 1):
    print(f"  [{i}/{N_RANDOM}] lr={lr}, batch={bs} ...", end=" ")
    result = train_and_evaluate(lr, bs, MODEL_NAME)
    result.update({"method": "Random Search", "lr": lr, "batch_size": bs})
    random_results.append(result)
    print(f"val_acc={result['best_val_acc']}%  "
          f"stopped_epoch={result['stopped_epoch']}")

best_random = max(random_results, key=lambda x: x["best_val_acc"])
print(f"\n  Best random config → lr={best_random['lr']}, "
      f"batch={best_random['batch_size']}, "
      f"val_acc={best_random['best_val_acc']}%\n")

print("METHOD 3 — BAYESIAN OPTIMIZATION (Optuna TPE)")
print("=" * 55)

optuna_results = []

def optuna_objective(trial):
    lr = trial.suggest_categorical(
        "lr", [0.01, 0.001, 0.0005, 0.0001, 0.00005])
    batch_size = trial.suggest_categorical(
        "batch_size", [8, 16, 32])

    result = train_and_evaluate(lr, batch_size, MODEL_NAME)
    result.update({"method": "Bayesian (Optuna)",
                   "lr": lr, "batch_size": batch_size})
    optuna_results.append(result)

    trial_num = len(optuna_results)
    print(f"  [{trial_num}/10] lr={lr}, batch={batch_size} ...  "
          f"val_acc={result['best_val_acc']}%  "
          f"stopped_epoch={result['stopped_epoch']}")

    return -result["best_val_acc"]


study = optuna.create_study(
    direction="minimize",                           
    sampler=optuna.samplers.TPESampler(seed=SEED)  
)
study.optimize(optuna_objective, n_trials=10)

best_optuna_trial = study.best_trial
print(f"\n  Best Optuna config → "
      f"lr={best_optuna_trial.params['lr']}, "
      f"batch={best_optuna_trial.params['batch_size']}, "
      f"val_acc={-best_optuna_trial.value}%\n")


all_results = grid_results + random_results + optuna_results
results_df  = pd.DataFrame(all_results)[[
    "method", "lr", "batch_size",
    "best_val_acc", "best_val_loss",
    "stopped_epoch", "duration_sec"
]]
results_df = results_df.sort_values("best_val_acc", ascending=False)
results_df.insert(0, "rank", range(1, len(results_df) + 1))

output_csv = f"hyperparameter_results_{MODEL_NAME}.csv"
results_df.to_csv(output_csv, index=False)

print("=" * 75)
print(f"FINAL RESULTS TABLE — {MODEL_NAME}")
print("=" * 75)
print(results_df.to_string(index=False))
print(f"\n[OK] Results saved to {output_csv}")

best_row = results_df.iloc[0]
print("\n" + "=" * 75)
print("BEST OVERALL CONFIGURATION")
print("=" * 75)
print(f"  Method      : {best_row['method']}")
print(f"  LR          : {best_row['lr']}")
print(f"  Batch size  : {int(best_row['batch_size'])}")
print(f"  Val accuracy: {best_row['best_val_acc']}%")
print(f"  Val loss    : {best_row['best_val_loss']}")
print(f"  Stopped at  : epoch {int(best_row['stopped_epoch'])}")
print(f"\n>>> Use these values in step_f_training.py")
print(f">>> Step e complete.")