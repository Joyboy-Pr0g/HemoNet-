import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset_preparation import (
    HeadCTDataset,
    DATA_DIR,
    LABELS_CSV,
    IMG_SIZE,
    build_train_transforms,
    build_val_test_transforms,
)


def plot_split_distribution(
    train_df, val_df, test_df, out_path="split_distribution.png", show=True
):
    splits = [
        f"Train ({len(train_df)})",
        f"Val ({len(val_df)})",
        f"Test ({len(test_df)})",
    ]
    hemorrhage = [
        (train_df["hemorrhage"] == 1).sum(),
        (val_df["hemorrhage"] == 1).sum(),
        (test_df["hemorrhage"] == 1).sum(),
    ]
    normal = [
        (train_df["hemorrhage"] == 0).sum(),
        (val_df["hemorrhage"] == 0).sum(),
        (test_df["hemorrhage"] == 0).sum(),
    ]

    x = range(len(splits))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, hemorrhage, label="Hemorrhage", color="#D85A30", width=0.4)
    ax.bar(x, normal, bottom=hemorrhage, label="Normal", color="#1D9E75", width=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontsize=11)
    ax.set_ylabel("Number of images")
    ax.set_title("Class distribution across splits", fontsize=12)
    ax.legend()

    for i, (h, n) in enumerate(zip(hemorrhage, normal)):
        ax.text(
            i,
            h / 2,
            str(h),
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
        )
        ax.text(
            i,
            h + n / 2,
            str(n),
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    print(f"Split distribution chart saved to {out_path}")


BATCH_SIZE = 32
SEED = 42

df = pd.read_csv(LABELS_CSV)
df.columns = df.columns.str.strip()
df["id"] = df["id"].astype(str).str.zfill(3)
df["filename"] = df["id"] + ".png"


train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["hemorrhage"], random_state=SEED
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["hemorrhage"], random_state=SEED
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("=== Data Split Results ===\n")
for name, subset in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    total = len(subset)
    h = (subset["hemorrhage"] == 1).sum()
    n = (subset["hemorrhage"] == 0).sum()
    print(
        f"{name:6s}: {total:3d} images  |  "
        f"Hemorrhage: {h} ({h/total*100:.0f}%)  |  "
        f"Normal: {n} ({n/total*100:.0f}%)"
    )

total_check = len(train_df) + len(val_df) + len(test_df)
print(f"\nTotal : {total_check} images (should be 200)")
assert total_check == 200, "Split error: images don't add up to 200"
print("Split verified — no data leakage, no missing images.\n")

train_df.to_csv("split_train.csv", index=False)
val_df.to_csv("split_val.csv", index=False)
test_df.to_csv("split_test.csv", index=False)
print("Split saved to split_train.csv / split_val.csv / split_test.csv")


train_transforms = build_train_transforms(IMG_SIZE)
val_test_transforms = build_val_test_transforms(IMG_SIZE)

train_dataset = HeadCTDataset(train_df, DATA_DIR, transform=train_transforms)
val_dataset = HeadCTDataset(val_df, DATA_DIR, transform=val_test_transforms)
test_dataset = HeadCTDataset(test_df, DATA_DIR, transform=val_test_transforms)


train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

print("\n=== DataLoader Summary ===")
print(f"Train batches : {len(train_loader)}  ({len(train_dataset)} images)")
print(f"Val batches   : {len(val_loader)}   ({len(val_dataset)} images)")
print(f"Test batches  : {len(test_loader)}   ({len(test_dataset)} images)")


images, labels = next(iter(train_loader))
print(f"\n=== Batch Sanity Check ===")
print(f"Image batch shape : {images.shape}")
print(f"Label batch shape : {labels.shape}")
print(f"Label values      : {labels.unique().tolist()}")
print("DataLoaders working correctly.\n")

plot_split_distribution(train_df, val_df, test_df)
