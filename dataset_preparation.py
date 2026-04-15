import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision import transforms

DATA_DIR = "ct_dataset/head_ct"
LABELS_CSV = "ct_dataset/labels.csv"
IMG_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transforms(img_size=IMG_SIZE):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_val_test_transforms(img_size=IMG_SIZE):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class HeadCTDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row["hemorrhage"], dtype=torch.float32)
        return image, label


def visualize_samples(df, img_dir, n=6, random_state=42):
    sample = df.sample(n=n, random_state=random_state).reset_index(drop=True)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    for i, ax in enumerate(axes):
        row = sample.iloc[i]
        img_path = os.path.join(img_dir, row["filename"])
        img = Image.open(img_path).convert("L")  # grayscale for display
        label = "Hemorrhage" if row["hemorrhage"] == 1 else "Normal"
        ax.imshow(img, cmap="gray")
        ax.set_title(
            label, fontsize=10, color="red" if row["hemorrhage"] == 1 else "green"
        )
        ax.axis("off")
    plt.suptitle("Sample CT Images from Dataset", fontsize=13)
    plt.tight_layout()
    plt.savefig("sample_images.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[OK] Sample visualization saved to sample_images.png")


if __name__ == "__main__":
    BATCH_SIZE = 32
    SEED = 42

    df = pd.read_csv(LABELS_CSV)
    df.columns = df.columns.str.strip()
    df["id"] = df["id"].astype(str).str.zfill(3)
    df["filename"] = df["id"] + ".png"

    print("=== Dataset Overview ===")
    print(f"Total images : {len(df)}")
    print(f"Hemorrhage   : {(df['hemorrhage'] == 1).sum()}")
    print(f"Normal       : {(df['hemorrhage'] == 0).sum()}")
    print(f"\nFirst 5 rows:\n{df.head()}")

    missing = []
    for fname in df["filename"]:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            missing.append(fname)

    if missing:
        print(f"\n[WARNING] {len(missing)} missing files: {missing[:5]}")
    else:
        print(f"\n[OK] All {len(df)} image files found.")

    train_transforms = build_train_transforms(IMG_SIZE)
    val_test_transforms = build_val_test_transforms(IMG_SIZE)

    print("\n[OK] Transforms defined.")
    print(
        "  Train  : Resize → Grayscale→RGB → Flip → Rotate → Jitter → Tensor → Normalize"
    )
    print("  Val/Test: Resize → Grayscale→RGB → Tensor → Normalize")

    visualize_samples(df, DATA_DIR, random_state=SEED)

    sample_row = df.iloc[0]
    sample_path = os.path.join(DATA_DIR, sample_row["filename"])
    sample_img = Image.open(sample_path).convert("RGB")
    sample_tensor = val_test_transforms(sample_img)

    print("\n=== Tensor Sanity Check ===")
    print(f"Shape  : {sample_tensor.shape}")
    print(f"Min    : {sample_tensor.min():.4f}")
    print(f"Max    : {sample_tensor.max():.4f}")
    print(f"Mean   : {sample_tensor.mean():.4f}")
    print("Preprocessing pipeline working correctly.")
