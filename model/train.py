import os, sys, json, copy, time, random, shutil, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).parent.parent

# Fix random seeds for reproducibility — eigula set na korle data split r training er shomoy randomization hobe, jar fole protibar run korle different results asbe.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class BanknoteCNN(nn.Module):
    #Same architecture as model/model.py — duplicated here so train.py is self-contained.
    def __init__(self, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,  32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(128 * 4, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ei function ta data split er jono, data_root/{fake,real}/ theke train/val/test te copy korbe
def prepare_splits(data_root: Path, out_root: Path):
    """
    Copy images from data_root/{fake,real}/ into train/val/test splits
    under out_root/. Skips files that already exist.
    """
    for split in ("train", "val", "test"):
        for cls in ("fake", "real"):
            (out_root / split / cls).mkdir(parents=True, exist_ok=True)

    for cls in ("fake", "real"):
        # Collect all images for this class
        imgs = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            imgs += sorted((data_root / cls).rglob(ext))

        random.shuffle(imgs)
        n          = len(imgs)
        train_end  = int(n * 0.70)
        val_end    = int(n * 0.85)  # 70–85% → val, 85–100% → test

        for i, img in enumerate(imgs):
            if   i < train_end: split = "train"
            elif i < val_end:   split = "val"
            else:               split = "test"

            dest = out_root / split / cls / img.name
            if not dest.exists():
                shutil.copy2(img, dest)

        print(f"  {cls}: {train_end} train  |  {val_end - train_end} val  |  {n - val_end} test")



def train(data_dir: Path, save_dir: Path, epochs: int, lr: float, batch_size: int):
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 48
    print(f"Device: {device}")

    # Augmented transform for training; plain resize+normalize for validation
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    train_ds = datasets.ImageFolder(str(data_dir / "train"), train_tf)
    val_ds   = datasets.ImageFolder(str(data_dir / "val"),   val_tf)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    

    class_names = train_ds.classes
    print(f"Classes: {class_names}  |  Train: {len(train_ds)}  |  Val: {len(val_ds)}")
    
    # hyperparameters and model setup eikhnae
    model     = BanknoteCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc  = 0.0
    best_weights  = copy.deepcopy(model.state_dict())
    history       = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Training pass
        model.train()
        train_loss, train_correct = 0.0, 0
        for X, y in train_dl:
            X, y = X.to(device), y.to(device)  # Move batch to GPU if available
            optimizer.zero_grad()
            logits = model(X)
            loss   = criterion(logits, y) # Calculate loss
            loss.backward()  # Backpropagation
            optimizer.step() # Update weights
            train_loss    += loss.item() * X.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()

        train_loss /= len(train_ds)
        train_acc   = train_correct / len(train_ds)
        
        
        # ei part ta validation er jono
        # Validation pass
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for X, y in val_dl:
                X, y  = X.to(device), y.to(device)
                logits = model(X)
                loss   = criterion(logits, y)
                val_loss    += loss.item() * X.size(0)
                val_correct += (logits.argmax(1) == y).sum().item()

        val_loss /= len(val_ds)
        val_acc   = val_correct / len(val_ds)
        scheduler.step()

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_acc"].append(round(val_acc, 4))

        print(f"Epoch {epoch:3d}/{epochs}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  ({time.time()-t0:.0f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, save_dir / "best_model.pth")
            print(f"  ✓ New best saved (val_acc={best_val_acc:.4f})")

    # Save metadata alongside the weights
    model.load_state_dict(best_weights)
    meta = {
        "class_names":   class_names,
        "num_classes":   2,
        "input_size":    img_size,
        "best_val_acc":  round(best_val_acc, 4),
        "epochs":        epochs,
        "history":       history,
        "architecture":  "BanknoteCNN",
    }
    with open(save_dir / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ Training complete  |  Best val acc: {best_val_acc:.4f}  |  Saved to: {save_dir}/")
    print("  Next step: python model/test.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",    default=str(ROOT / "data"))
    p.add_argument("--prepared_dir", default=str(ROOT / "data_prepared"))
    p.add_argument("--save_dir",     default=str(ROOT / "model" / "saved"))
    p.add_argument("--epochs",  type=int,   default=15)
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--batch",   type=int,   default=64)
    args = p.parse_args()

    print("── Preparing dataset splits ──")
    prepare_splits(Path(args.data_root), Path(args.prepared_dir))
    print("── Training ──")
    train(Path(args.prepared_dir), Path(args.save_dir), args.epochs, args.lr, args.batch)
