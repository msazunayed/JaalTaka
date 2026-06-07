# JaalTaka 🔍

> **জাল টাকা** (JaalTaka) means *Counterfeit Money* in Bengali.

JaalTaka is a deep learning–powered web application that detects whether a banknote image is **real or fake**. Upload a photo of a banknote, and the app uses a trained Convolutional Neural Network (CNN) to classify it with a confidence score — in real time, directly in the browser.

---

## What the Project Does

- Accepts an image of a banknote (JPG, PNG, WEBP, BMP, TIFF)
- Runs inference through a custom-built `BanknoteCNN` model
- Returns a prediction: **real** or **fake**, with a confidence percentage and per-class probabilities
- Displays the uploaded image thumbnail alongside the result in a clean web UI

The model was trained on a dataset of 600 banknote images (300 real, 300 fake) across 50 unique note types, achieving **~88% validation accuracy** over 5 epochs.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web Framework | FastAPI |
| ML Framework | PyTorch + TorchVision |
| Image Processing | Pillow (PIL) |
| Frontend | Jinja2 + HTML/CSS/JS |
| Server | Uvicorn (ASGI) |
| Model | Custom `BanknoteCNN` (3-block CNN) |

---

## Project Structure

```
JaalTaka/
│
├── main.py                   # FastAPI app — routes, startup, file upload handling
├── requirements.txt          # Python dependencies
│
├── model/
│   ├── __init__.py
│   ├── model.py              # BanknoteCNN architecture definition
│   ├── predict.py            # Inference logic (BanknotePredictor singleton)
│   ├── train.py              # Training script with dataset splitting
│   ├── test.py               # Evaluation on the held-out test set
│   └── saved/
│       ├── best_model.pth    # Pre-trained model weights
│       └── model_meta.json   # Training metadata (accuracy, history, config)
│
├── templates/
│   └── index.html            # Frontend — single-page upload & result UI
│
└── data/
    ├── real/                 # Real banknote images (note_XXX/note_XXX_Y.jpg)
    └── fake/                 # Fake banknote images (note_XXX/note_XXX_Y.jpg)
```

---

## Setup & Installation

### Prerequisites

- Python 3.9 or later
- `pip`
- (Optional) A CUDA-capable GPU for faster inference

### 1. Clone or extract the project

```bash
# If cloning from git:
git clone <repo-url>
cd JaalTaka

# Or if you have the zip:
unzip JaalTaka.zip
cd JaalTaka
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs: `fastapi`, `uvicorn`, `python-multipart`, `jinja2`, `torch`, `torchvision`, and `pillow`.

> **Note:** The PyTorch install from `requirements.txt` fetches the CPU build by default. For GPU support, install PyTorch manually first following the instructions at [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Running the App

The pre-trained model weights (`model/saved/best_model.pth`) are included, so you can run the app immediately without training.

```bash
uvicorn main:app --reload
```

Then open your browser at: **http://127.0.0.1:8000**

You should see the upload interface. Drop or select a banknote image and click **Detect** to get a prediction.

### Other useful flags

```bash
# Run on a specific port
uvicorn main:app --reload --port 8080

# Make accessible on your local network
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serve the web UI |
| `POST` | `/api/predict` | Upload an image, get prediction JSON |
| `GET` | `/api/health` | Health check — returns `{"status": "ok"}` |
| `GET` | `/api/model-info` | Returns model metadata (architecture, accuracy, etc.) |

### Example: `/api/predict` response

```json
{
  "label": "fake",
  "confidence": 0.9231,
  "probabilities": {
    "fake": 0.9231,
    "real": 0.0769
  },
  "image_b64": "<base64-encoded thumbnail>"
}
```

---

## Training the Model (Optional)

If you want to retrain the model on your own dataset, follow these steps.

### 1. Prepare your data

Place your banknote images into the `data/` directory:

```
data/
├── real/   ← images of genuine banknotes (.jpg / .jpeg / .png)
└── fake/   ← images of counterfeit banknotes (.jpg / .jpeg / .png)
```

Each folder should contain images either directly or inside per-note subdirectories (e.g. `note_001/note_001_1.jpg`).

### 2. Run the training script

```bash
python model/train.py
```

The script will:
- Auto-split images **70% train / 15% val / 15% test** into `data_prepared/`
- Train `BanknoteCNN` with Adam optimizer + cosine annealing scheduler
- Apply data augmentation (random flips, rotation, color jitter) on the training set
- Save the best weights to `model/saved/best_model.pth`
- Save metadata (accuracy history, config) to `model/saved/model_meta.json`

#### Optional training arguments

```bash
python model/train.py \
  --data_root    data/           \   # source images
  --prepared_dir data_prepared/  \   # split output directory
  --save_dir     model/saved/    \   # weights output directory
  --epochs       15              \
  --lr           0.001           \
  --batch        64
```

### 3. Evaluate on the test set

```bash
python model/test.py
```

---

## Model Architecture

`BanknoteCNN` is a lightweight 3-block CNN designed for 48×48 RGB banknote images.

```
Input: (batch, 3, 48, 48)
│
├── Block 1: Conv(3→32, 3×3) + BatchNorm + ReLU + MaxPool  →  (batch, 32, 24, 24)
├── Block 2: Conv(32→64, 3×3) + BatchNorm + ReLU + MaxPool →  (batch, 64, 12, 12)
├── Block 3: Conv(64→128, 3×3) + BatchNorm + ReLU + AdaptiveAvgPool(2×2)
│                                                           →  (batch, 128, 2, 2)
│
└── Classifier: Flatten → Linear(512→256) → ReLU → Dropout(0.4) → Linear(256→2)
                                                               ↓
                                                   Output logits: [fake, real]
```

**Pre-trained model stats:**

| Metric | Value |
|---|---|
| Architecture | BanknoteCNN |
| Input size | 48 × 48 px |
| Training epochs | 5 |
| Best validation accuracy | **88.33%** |
| Classes | `fake`, `real` |

---

## Supported Image Formats

`.jpg` · `.jpeg` · `.png` · `.bmp` · `.webp` · `.tiff`

EXIF rotation is automatically corrected before inference.

Output Result: 

https://github.com/user-attachments/assets/8155d4ff-6e24-4ed6-8b1d-6a9d26a85031



---

## License

This project is for educational and research purposes.
