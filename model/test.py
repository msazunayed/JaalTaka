"""
Test script — evaluates the saved model on the held-out test set.

Usage (run after training):
    python model/test.py

Prints overall accuracy, per-class accuracy, and a confusion matrix.
"""

import sys, json
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

ROOT     = Path(__file__).parent.parent
PREPARED = ROOT / "data_prepared"
SAVED    = Path(__file__).parent / "saved"

sys.path.insert(0, str(Path(__file__).parent))
from model import BanknoteCNN


def test():
    # Load metadata saved during training
    meta_path = SAVED / "model_meta.json"
    if not meta_path.exists():
        print("✗ model_meta.json not found. Run train.py first.")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    class_names = meta["class_names"]
    img_size    = meta.get("input_size", 48)

    # Verify the test split exists
    test_dir = PREPARED / "test"
    if not test_dir.exists():
        print("✗ Test set not found at data_prepared/test/")
        print("  Delete data_prepared/ and re-run train.py to regenerate the 70/15/15 split.")
        return

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = BanknoteCNN(num_classes=meta["num_classes"])
    model.load_state_dict(
        torch.load(SAVED / "best_model.pth", map_location=device, weights_only=True)
    )
    model.to(device).eval()

    # Load test data
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    test_ds = datasets.ImageFolder(str(test_dir), transform)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    print(f"Test set: {len(test_ds)} images  |  Classes: {test_ds.classes}")

    # Run evaluation and build confusion matrix
    n_classes = len(class_names)
    confusion = [[0] * n_classes for _ in range(n_classes)]
    correct   = 0

    with torch.no_grad():
        for X, y in test_dl:
            X, y  = X.to(device), y.to(device)
            preds = model(X).argmax(dim=1)
            correct += (preds == y).sum().item()
            for actual, pred in zip(y.tolist(), preds.tolist()):
                confusion[actual][pred] += 1

    total    = len(test_ds)
    accuracy = correct / total * 100

    # Print results
    print(f"\n{'─' * 40}")
    print(f"  Test Accuracy : {accuracy:.2f}%  ({correct}/{total})")
    print(f"{'─' * 40}")

    print("\n  Per-class accuracy:")
    for i, cls in enumerate(class_names):
        row_total = sum(confusion[i])
        cls_acc   = confusion[i][i] / row_total * 100 if row_total > 0 else 0
        print(f"    {cls:<6}  {cls_acc:.2f}%  ({confusion[i][i]}/{row_total})")

    print("\n  Confusion matrix (rows=actual, cols=predicted):")
    pad    = max(len(c) for c in class_names) + 2
    header = " " * (pad + 2) + "  ".join(f"pred_{c}" for c in class_names)
    print(f"  {header}")
    for i, cls in enumerate(class_names):
        row = "  ".join(
            f"{confusion[i][j]:>{len('pred_' + class_names[j])}}"
            for j in range(n_classes)
        )
        print(f"  actual_{cls:<{pad}}{row}")

    # Quick sanity check: val vs test accuracy
    print(f"\n  Val acc (from training) : {meta['best_val_acc'] * 100:.2f}%")
    print(f"  Test acc                : {accuracy:.2f}%")
    diff = accuracy - meta["best_val_acc"] * 100
    if abs(diff) < 5:
        print("  ✓ Val and test are close — model generalizes well")
    elif diff < -5:
        print("  ⚠ Test is lower than val — possible overfitting")
    else:
        print("  ✓ Test is higher than val — good generalization")


if __name__ == "__main__":
    test()
