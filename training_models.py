import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_preparation import (
    HeadCTDataset,
    DATA_DIR,
    IMG_SIZE,
    build_train_transforms,
    build_val_test_transforms,
)
from models import CustomCNN, build_resnet50

SEED       = 42
MAX_EPOCHS = 100
PATIENCE   = 13
RESNET_LR  = 0.01
RESNET_BS  = 16
CUSTOM_LR  = 0.001
CUSTOM_BS  = 16

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
print(f"Epochs : up to {MAX_EPOCHS} (patience={PATIENCE})\n")

train_transforms    = build_train_transforms(IMG_SIZE)
val_test_transforms = build_val_test_transforms(IMG_SIZE)

train_df = pd.read_csv("split_train.csv")
val_df   = pd.read_csv("split_val.csv")


def train_model(model, model_name, lr, batch_size, use_scheduler=False):
    print("=" * 55)
    print(f"TRAINING: {model_name}")
    print(f"  LR={lr}  |  Batch={batch_size}  |  Scheduler={'YES' if use_scheduler else 'NO'}")
    print("=" * 55)

    train_loader = DataLoader(
        HeadCTDataset(train_df, DATA_DIR, transform=train_transforms),
        batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        HeadCTDataset(val_df, DATA_DIR, transform=val_test_transforms),
        batch_size=batch_size, shuffle=False, num_workers=0)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-6)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss"  : [], "val_acc"  : [],
        "lr"        : []
    }

    best_val_loss     = float("inf")
    best_weights      = None
    epochs_no_improve = 0
    best_epoch        = 1
    start_time        = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):

        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs    = model(images).squeeze(1)
            loss       = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item()
            preds          = (outputs >= 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)

        train_loss /= len(train_loader)
        train_acc   = train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs     = model(images).squeeze(1)
                val_loss   += criterion(outputs, labels).item()
                preds       = (outputs >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        val_loss /= len(val_loader)
        val_acc   = val_correct / val_total

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(f"  Epoch {epoch:02d}/{MAX_EPOCHS} | "
              f"Train loss: {train_loss:.4f}  acc: {train_acc*100:.1f}% | "
              f"Val loss: {val_loss:.4f}  acc: {val_acc*100:.1f}% | "
              f"LR: {current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            best_weights      = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            best_epoch        = epoch
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\n  [Early stop] No improvement for {PATIENCE} epochs. Stopping at epoch {epoch}.")
                break

    if best_weights:
        model.load_state_dict(best_weights)

    duration = time.time() - start_time
    print(f"\n  Best epoch    : {best_epoch}")
    print(f"  Best val loss : {best_val_loss:.4f}")
    print(f"  Best val acc  : {history['val_acc'][best_epoch-1]*100:.1f}%")
    print(f"  Training time : {duration:.1f}s\n")

    return history, best_epoch


def plot_training_curves(history, model_name, best_epoch):
    epochs = range(1, len(history["train_loss"]) + 1)
    has_lr = len(set(history["lr"])) > 1
    ncols  = 3 if has_lr else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    fig.suptitle(f"Training curves — {model_name}", fontsize=14)

    ax1, ax2 = axes[0], axes[1]

    ax1.plot(epochs, history["train_loss"], label="Train loss", color="#D85A30", linewidth=2)
    ax1.plot(epochs, history["val_loss"],   label="Val loss",   color="#1D9E75", linewidth=2)
    ax1.axvline(x=best_epoch, color="gray", linestyle="--", linewidth=1, label=f"Best epoch ({best_epoch})")
    ax1.set_title("Model loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train accuracy", color="#D85A30", linewidth=2)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]],   label="Val accuracy",   color="#1D9E75", linewidth=2)
    ax2.axvline(x=best_epoch, color="gray", linestyle="--", linewidth=1, label=f"Best epoch ({best_epoch})")
    ax2.set_title("Model accuracy"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim([0, 105]); ax2.legend(); ax2.grid(True, alpha=0.3)

    if has_lr:
        ax3 = axes[2]
        ax3.plot(epochs, history["lr"], color="#534AB7", linewidth=2)
        ax3.set_title("Learning rate schedule")
        ax3.set_xlabel("Epoch"); ax3.set_ylabel("Learning rate")
        ax3.set_yscale("log"); ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f"training_curves_{model_name.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [OK] Curves saved to {filename}")


def save_model(model, model_name):
    os.makedirs("saved_models", exist_ok=True)
    filename = f"saved_models/{model_name.replace(' ', '_')}.pth"
    torch.save(model.state_dict(), filename)
    print(f"  [OK] Model saved to {filename}\n")


resnet_model             = build_resnet50().to(device)
resnet_history, resnet_best = train_model(
    resnet_model, "ResNet50",
    lr=RESNET_LR, batch_size=RESNET_BS,
    use_scheduler=False)
plot_training_curves(resnet_history, "ResNet50", resnet_best)
save_model(resnet_model, "ResNet50")

custom_model             = CustomCNN().to(device)
custom_history, custom_best = train_model(
    custom_model, "CustomCNN",
    lr=CUSTOM_LR, batch_size=CUSTOM_BS,
    use_scheduler=True)
plot_training_curves(custom_history, "CustomCNN", custom_best)
save_model(custom_model, "CustomCNN")

print("=" * 55)
print("TRAINING COMPLETE — SUMMARY")
print("=" * 55)
print(f"  ResNet50  → best val acc: {resnet_history['val_acc'][resnet_best-1]*100:.1f}%  at epoch {resnet_best}")
print(f"  CustomCNN → best val acc: {custom_history['val_acc'][custom_best-1]*100:.1f}%  at epoch {custom_best}")
print(f"\n  Saved: saved_models/ResNet50.pth")
print(f"  Saved: saved_models/CustomCNN.pth")