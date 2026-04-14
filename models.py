import torch
import torch.nn as nn
from torchvision import models

def build_resnet50(freeze_backbone=True):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features          
    model.fc = nn.Sequential(
        nn.Linear(in_features, 1),
        nn.Sigmoid()
    )

    return model


class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3,  out_channels=32,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)   
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)   
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)   
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block1(x)              
        x = self.block2(x)              
        x = self.block3(x)              

        x = self.global_avg_pool(x)     
        x = x.view(x.size(0), -1)       

        out = self.classifier(x)        
        return out


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    resnet_model     = build_resnet50(freeze_backbone=True).to(device)
    total_r          = sum(p.numel() for p in resnet_model.parameters())
    trainable_r      = sum(p.numel() for p in resnet_model.parameters()
                           if p.requires_grad)

    print("=" * 50)
    print("MODEL 1 — ResNet50 (pretrained, transfer learning)")
    print("=" * 50)
    print(f"Total parameters    : {total_r:,}")
    print(f"Trainable parameters: {trainable_r:,}  ← only FC layer")
    print(f"Frozen parameters   : {total_r - trainable_r:,}")

    dummy    = torch.randn(4, 3, 224, 224).to(device)
    out      = resnet_model(dummy)
    print(f"Output shape        : {out.shape}")
    print(f"[OK] ResNet50 forward pass working.\n")

    custom_model     = CustomCNN().to(device)
    total_c          = sum(p.numel() for p in custom_model.parameters())
    trainable_c      = sum(p.numel() for p in custom_model.parameters()
                           if p.requires_grad)

    print("=" * 50)
    print("MODEL 2 — CustomCNN v2 (3-block, from scratch)")
    print("=" * 50)
    print(f"Total parameters    : {total_c:,}")
    print(f"Trainable parameters: {trainable_c:,}  ← all layers")

    out2 = custom_model(dummy)
    print(f"Output shape        : {out2.shape}")
    print(f"[OK] CustomCNN forward pass working.\n")

    print("=" * 50)
    print("COMPARISON SUMMARY")
    print("=" * 50)
    print(f"{'':30s} {'ResNet50':>14s} {'CustomCNN v2':>14s}")
    print(f"{'Total parameters':30s} {total_r:>14,} {total_c:>14,}")
    print(f"{'Trainable parameters':30s} {trainable_r:>14,} {trainable_c:>14,}")
    print(f"{'Pretrained weights':30s} {'Yes (ImageNet)':>14s} {'No':>14s}")
    print(f"{'Conv blocks':30s} {'5 stages':>14s} {'3 blocks':>14s}")
    print(f"{'Input size':30s} {'224×224':>14s} {'224×224':>14s}")
    print(f"{'Output':30s} {'1 (sigmoid)':>14s} {'1 (sigmoid)':>14s}")