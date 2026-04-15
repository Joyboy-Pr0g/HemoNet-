import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, ConfusionMatrixDisplay
)

from dataset_preparation import (
    HeadCTDataset,
    DATA_DIR,
    IMG_SIZE,
    build_val_test_transforms,
)
from models import CustomCNN, build_resnet50

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
SEED       = 42
BATCH_SIZE = 16
THRESHOLD  = 0.5

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

os.makedirs("evaluation_results", exist_ok=True)
os.makedirs("evaluation_results/misclassified", exist_ok=True)

# ─────────────────────────────────────────────
# TTA TRANSFORMS
# 5 views per image — predictions averaged
# ─────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_tta_transforms(img_size=IMG_SIZE):
    base = [
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
    ]
    end = [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return [
        transforms.Compose(base + end),
        transforms.Compose(base + [transforms.RandomHorizontalFlip(p=1.0)] + end),
        transforms.Compose(base + [transforms.RandomRotation(degrees=(10, 10))] + end),
        transforms.Compose(base + [transforms.RandomRotation(degrees=(-10, -10))] + end),
        transforms.Compose(base + [transforms.RandomAffine(degrees=0, translate=(0.05, 0.05))] + end),
    ]

tta_transforms    = build_tta_transforms()
val_test_transforms = build_val_test_transforms(IMG_SIZE)

# ─────────────────────────────────────────────
# LOAD TEST SPLIT
# ─────────────────────────────────────────────
test_df = pd.read_csv("split_test.csv")
print(f"Test set: {len(test_df)} images")
print(f"  Hemorrhage : {(test_df['hemorrhage']==1).sum()}")
print(f"  Normal     : {(test_df['hemorrhage']==0).sum()}\n")

test_dataset = HeadCTDataset(test_df, DATA_DIR, transform=val_test_transforms)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)

# ─────────────────────────────────────────────
# LOAD TRAINED MODELS
# ─────────────────────────────────────────────
resnet_model = build_resnet50().to(device)
resnet_model.load_state_dict(
    torch.load("saved_models/ResNet50.pth", map_location=device))
resnet_model.eval()
print("[OK] ResNet50 loaded from saved_models/ResNet50.pth")

custom_model = CustomCNN().to(device)
custom_model.load_state_dict(
    torch.load("saved_models/CustomCNN.pth", map_location=device))
custom_model.eval()
print("[OK] CustomCNN loaded from saved_models/CustomCNN.pth\n")


# ─────────────────────────────────────────────
# STANDARD EVALUATION (no TTA)
# Returns: all predictions, probabilities, true labels
# ─────────────────────────────────────────────
def evaluate_standard(model, loader):
    all_probs  = []
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze(1)
            probs   = outputs.cpu().numpy()
            preds   = (outputs >= THRESHOLD).float().cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    return (np.array(all_probs),
            np.array(all_preds),
            np.array(all_labels))


# ─────────────────────────────────────────────
# TTA EVALUATION
# For each image: average predictions across 5 transforms
# ─────────────────────────────────────────────
def evaluate_tta(model, df, img_dir, threshold=THRESHOLD):
    all_probs  = []
    all_labels = []

    for idx in range(len(df)):
        row      = df.iloc[idx]
        img_path = os.path.join(img_dir, row["filename"])
        image    = Image.open(img_path).convert("RGB")
        label    = row["hemorrhage"]

        view_probs = []
        for tfm in tta_transforms:
            tensor = tfm(image).unsqueeze(0).to(device)
            with torch.no_grad():
                prob = model(tensor).squeeze().item()
            view_probs.append(prob)

        avg_prob = np.mean(view_probs)
        all_probs.append(avg_prob)
        all_labels.append(label)

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds  = (all_probs >= threshold).astype(float)
    return all_probs, all_preds, all_labels


# ─────────────────────────────────────────────
# METRICS PRINTER
# ─────────────────────────────────────────────
def print_metrics(name, preds, labels, probs=None):
    cm        = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    accuracy  = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy    : {accuracy*100:.1f}%")
    print(f"  Precision   : {precision*100:.1f}%")
    print(f"  Recall      : {recall*100:.1f}%  ← sensitivity (catches hemorrhages)")
    print(f"  Specificity : {specificity*100:.1f}%  ← catches normals")
    print(f"  F1 Score    : {f1*100:.1f}%")
    print(f"\n  Confusion Matrix:")
    print(f"               Pred Normal  Pred Hemorrhage")
    print(f"  True Normal       {tn:>3}          {fp:>3}")
    print(f"  True Hemorrhage   {fn:>3}          {tp:>3}")
    print(f"\n  TP={tp}  TN={tn}  FP={fp}  FN={fn}")

    return {"accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1, "specificity": specificity,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


# ─────────────────────────────────────────────
# CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────
def plot_confusion_matrix(labels, preds, model_name):
    cm  = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Hemorrhage"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12)
    plt.tight_layout()
    path = f"evaluation_results/confusion_matrix_{model_name.replace(' ','_')}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [OK] Confusion matrix saved to {path}")


# ─────────────────────────────────────────────
# ROC CURVE PLOT
# ─────────────────────────────────────────────
def plot_roc_curves(labels, resnet_probs, custom_probs):
    fpr_r, tpr_r, _ = roc_curve(labels, resnet_probs)
    fpr_c, tpr_c, _ = roc_curve(labels, custom_probs)
    auc_r = auc(fpr_r, tpr_r)
    auc_c = auc(fpr_c, tpr_c)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_r, tpr_r, color="#D85A30", linewidth=2,
            label=f"ResNet50  (AUC = {auc_r:.3f})")
    ax.plot(fpr_c, tpr_c, color="#1D9E75", linewidth=2,
            label=f"CustomCNN (AUC = {auc_c:.3f})")
    ax.plot([0,1],[0,1], "k--", linewidth=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve — Both Models")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = "evaluation_results/roc_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [OK] ROC curves saved to {path}")
    print(f"  ResNet50  AUC: {auc_r:.3f}")
    print(f"  CustomCNN AUC: {auc_c:.3f}")
    return auc_r, auc_c


# ─────────────────────────────────────────────
# THRESHOLD TUNING
# Tests thresholds 0.3, 0.4, 0.5, 0.6, 0.7
# Shows how accuracy and recall change
# ─────────────────────────────────────────────
def threshold_analysis(model_name, probs, labels):
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    results    = []

    print(f"\n  Threshold analysis — {model_name}")
    print(f"  {'Threshold':>10} {'Accuracy':>10} {'Recall':>10} {'Precision':>10} {'F1':>10}")

    for t in thresholds:
        preds   = (probs >= t).astype(float)
        cm      = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel()
        acc  = (tp+tn)/(tp+tn+fp+fn)
        rec  = tp/(tp+fn) if (tp+fn)>0 else 0
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        f1   = (2*prec*rec/(prec+rec)) if (prec+rec)>0 else 0
        results.append({"threshold": t, "accuracy": acc,
                        "recall": rec, "precision": prec, "f1": f1})
        print(f"  {t:>10.1f} {acc*100:>9.1f}% {rec*100:>9.1f}% "
              f"{prec*100:>9.1f}% {f1*100:>9.1f}%")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"Threshold Analysis — {model_name}", fontsize=12)

    ax1.plot(thresholds, [r["accuracy"]*100 for r in results],
             "o-", color="#534AB7", linewidth=2, label="Accuracy")
    ax1.plot(thresholds, [r["recall"]*100 for r in results],
             "o-", color="#D85A30", linewidth=2, label="Recall")
    ax1.plot(thresholds, [r["precision"]*100 for r in results],
             "o-", color="#1D9E75", linewidth=2, label="Precision")
    ax1.set_xlabel("Threshold"); ax1.set_ylabel("Score (%)")
    ax1.set_title("Metrics vs Threshold")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0.5, color="gray", linestyle="--", linewidth=1)

    ax2.plot(thresholds, [r["f1"]*100 for r in results],
             "o-", color="#BA7517", linewidth=2)
    ax2.set_xlabel("Threshold"); ax2.set_ylabel("F1 Score (%)")
    ax2.set_title("F1 Score vs Threshold")
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=0.5, color="gray", linestyle="--", linewidth=1)

    plt.tight_layout()
    path = f"evaluation_results/threshold_{model_name.replace(' ','_')}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [OK] Threshold analysis saved to {path}")
    return results


# ─────────────────────────────────────────────
# CONFIDENCE ANALYSIS
# Are wrong predictions high or low confidence?
# ─────────────────────────────────────────────
def confidence_analysis(model_name, probs, preds, labels):
    correct_mask   = (preds == labels)
    incorrect_mask = ~correct_mask

    correct_conf   = probs[correct_mask]
    incorrect_conf = probs[incorrect_mask]

    print(f"\n  Confidence Analysis — {model_name}")
    print(f"  Correct predictions   : {len(correct_conf):>3} images  "
          f"avg confidence: {np.mean(np.abs(correct_conf - 0.5))*100:.1f}% from boundary")
    print(f"  Incorrect predictions : {len(incorrect_conf):>3} images  "
          f"avg confidence: {np.mean(np.abs(incorrect_conf - 0.5))*100:.1f}% from boundary")

    if len(incorrect_conf) > 0:
        high_conf_wrong = (np.abs(incorrect_conf - 0.5) > 0.3).sum()
        print(f"  High-confidence wrong predictions (>0.3 from 0.5): {high_conf_wrong}")
        if high_conf_wrong > 0:
            print(f"  ⚠ {high_conf_wrong} dangerous misclassification(s) — model was confident but wrong")
        else:
            print(f"  ✓ All wrong predictions were low-confidence — model uncertain when wrong")


# ─────────────────────────────────────────────
# SAVE MISCLASSIFIED IMAGES
# ─────────────────────────────────────────────
def save_misclassified(model_name, df, probs, preds, labels, img_dir):
    folder = f"evaluation_results/misclassified/{model_name.replace(' ','_')}"
    os.makedirs(folder, exist_ok=True)

    count = 0
    for i in range(len(labels)):
        if preds[i] != labels[i]:
            row      = df.iloc[i]
            img_path = os.path.join(img_dir, row["filename"])
            img      = Image.open(img_path).convert("L")

            true_label = "Hemorrhage" if labels[i] == 1 else "Normal"
            pred_label = "Hemorrhage" if preds[i]  == 1 else "Normal"
            confidence = probs[i] if preds[i] == 1 else 1 - probs[i]

            fig, ax = plt.subplots(figsize=(3, 3.5))
            ax.imshow(img, cmap="gray")
            ax.set_title(
                f"True: {true_label}\nPred: {pred_label} ({confidence*100:.1f}%)",
                fontsize=9,
                color="red")
            ax.axis("off")
            plt.tight_layout()
            save_path = f"{folder}/error_{i:02d}_{true_label}_as_{pred_label}.png"
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            plt.close()
            count += 1

    print(f"  [OK] {count} misclassified images saved to {folder}/")


# ─────────────────────────────────────────────
# RUN EVALUATION — RESNET50
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("EVALUATING: ResNet50")
print("="*55)

r_probs, r_preds, r_labels = evaluate_standard(resnet_model, test_loader)
r_metrics = print_metrics("ResNet50 — Standard", r_preds, r_labels)
plot_confusion_matrix(r_labels, r_preds, "ResNet50")
threshold_analysis("ResNet50", r_probs, r_labels)
confidence_analysis("ResNet50", r_probs, r_preds, r_labels)
save_misclassified("ResNet50", test_df, r_probs, r_preds, r_labels, DATA_DIR)

print("\n  Running TTA evaluation...")
r_tta_probs, r_tta_preds, _ = evaluate_tta(resnet_model, test_df, DATA_DIR)
r_tta_metrics = print_metrics("ResNet50 — TTA (5 views)", r_tta_preds, r_labels)

# ─────────────────────────────────────────────
# RUN EVALUATION — CustomCNN
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("EVALUATING: CustomCNN")
print("="*55)

c_probs, c_preds, c_labels = evaluate_standard(custom_model, test_loader)
c_metrics = print_metrics("CustomCNN — Standard", c_preds, c_labels)
plot_confusion_matrix(c_labels, c_preds, "CustomCNN")
threshold_analysis("CustomCNN", c_probs, c_labels)
confidence_analysis("CustomCNN", c_probs, c_preds, c_labels)
save_misclassified("CustomCNN", test_df, c_probs, c_preds, c_labels, DATA_DIR)

print("\n  Running TTA evaluation...")
c_tta_probs, c_tta_preds, _ = evaluate_tta(custom_model, test_df, DATA_DIR)
c_tta_metrics = print_metrics("CustomCNN — TTA (5 views)", c_tta_preds, c_labels)

# ─────────────────────────────────────────────
# ROC CURVES — both models
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("ROC CURVES")
print("="*55)
auc_r, auc_c = plot_roc_curves(r_labels, r_probs, c_probs)

# ─────────────────────────────────────────────
# FINAL COMPARISON TABLE
# ─────────────────────────────────────────────
print("\n" + "="*65)
print("FINAL COMPARISON — TEST SET RESULTS")
print("="*65)
print(f"{'Metric':20s} {'ResNet50':>12} {'ResNet50+TTA':>14} {'CustomCNN':>12} {'CustomCNN+TTA':>15}")
print("-"*65)
metrics = ["accuracy", "precision", "recall", "f1", "specificity"]
for m in metrics:
    print(f"  {m.capitalize():18s} "
          f"{r_metrics[m]*100:>10.1f}% "
          f"{r_tta_metrics[m]*100:>12.1f}% "
          f"{c_metrics[m]*100:>10.1f}% "
          f"{c_tta_metrics[m]*100:>13.1f}%")
print(f"  {'AUC-ROC':18s} {auc_r:>10.3f} {'—':>13} {auc_c:>10.3f} {'—':>15}")

print(f"\n  Note: Test set = {len(test_df)} images "
      f"(±{1/len(test_df)*100:.1f}% per image = ±{100/len(test_df):.1f}% accuracy variance)")

results_df = pd.DataFrame({
    "Model"       : ["ResNet50", "ResNet50+TTA", "CustomCNN", "CustomCNN+TTA"],
    "Accuracy"    : [r_metrics["accuracy"], r_tta_metrics["accuracy"],
                     c_metrics["accuracy"], c_tta_metrics["accuracy"]],
    "Precision"   : [r_metrics["precision"], r_tta_metrics["precision"],
                     c_metrics["precision"], c_tta_metrics["precision"]],
    "Recall"      : [r_metrics["recall"], r_tta_metrics["recall"],
                     c_metrics["recall"], c_tta_metrics["recall"]],
    "F1"          : [r_metrics["f1"], r_tta_metrics["f1"],
                     c_metrics["f1"], c_tta_metrics["f1"]],
    "Specificity" : [r_metrics["specificity"], r_tta_metrics["specificity"],
                     c_metrics["specificity"], c_tta_metrics["specificity"]],
    "AUC"         : [auc_r, None, auc_c, None],
})
results_df.to_csv("evaluation_results/test_results.csv", index=False)
print(f"\n  [OK] Results saved to evaluation_results/test_results.csv")
print(f"  [OK] All plots saved to evaluation_results/")
print(f"\n>>> Step g complete. Run interface script next (Step h).")