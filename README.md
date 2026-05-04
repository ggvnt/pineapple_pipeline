# 🌿 Pineapple Plant Analysis System

A production-ready, three-task deep learning pipeline that analyses a **single RGB photo** of a pineapple plant and simultaneously predicts:

| Task | Output | Method |
|---|---|---|
| **Health** | healthy / nitrogen_deficiency / water_stress | 3-class softmax |
| **Growth stage** | Month 1–12 | 12-class softmax |
| **Width** | estimated cm | regression (Softplus) |

---

## Architecture

```
EfficientNet-B0 (ImageNet pretrained)
        │
   BatchNorm + Dropout(0.3)
        │
   ┌────┼────┐
   ▼    ▼    ▼
health  month  width
head    head   MLP → Softplus → cm
(×3)   (×12)
```

**Loss:** `0.4 × CrossEntropy(health) + 0.4 × CrossEntropy(month) + 0.2 × HuberLoss(width)`  
Width loss is masked for samples without ruler-calibrated ground truth.

---

## Dataset Structure

```
data/
├── M1/
│   ├── healthy/             *.jpg / *.png
│   ├── nitrogen_deficiency/
│   └── water_stress/
├── M2/ ...
└── M12/
    ├── healthy/
    ├── nitrogen_deficiency/
    └── water_stress/
```

### Optional width CSV

If you have ruler-measured widths, provide a CSV with two columns:

```
path,width_cm
/data/M1/healthy/img001.jpg,8.3
/data/M1/healthy/img002.jpg,7.9
...
```

Samples not in the CSV receive the domain-knowledge expected width as a soft target.

---

## Expected Plant Widths (domain knowledge)

| Month | Expected cm |
|-------|-------------|
| M1 | 8 |
| M2 | 12 |
| M3 | 17 |
| M4 | 23 |
| M5 | 29 |
| M6 | 35 |
| M7 | 41 |
| M8 | 46 |
| M9 | 51 |
| M10 | 55 |
| M11 | 59 |
| M12 | 62 |

**Stunting flag:** raised when `predicted_width < 0.80 × expected_width`.

---

## Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Train
```bash
python main.py train \
    --data_root   /path/to/data \
    --width_csv   /path/to/width_labels.csv \   # optional
    --output_dir  runs/exp01/output \
    --checkpoint_dir runs/exp01/checkpoints \
    --epochs 80 \
    --batch_size 32 \
    --device auto
```

### 3. Infer (single image)
```bash
python main.py infer \
    --image      photo.jpg \
    --checkpoint runs/exp01/checkpoints/best_model.pt \
    --save_dir   runs/exp01/inference
```

### 4. Python API
```python
from pineapple import predict

result = predict(
    image_path = "photo.jpg",
    model_path = "runs/exp01/checkpoints/best_model.pt",
    save_dir   = "output/",   # writes gradcam_overlay.jpg + prediction.json
)

print(result["health"])             # "nitrogen_deficiency"
print(result["health_confidence"])  # 0.923
print(result["month"])              # "M4"
print(result["width_cm"])           # 21.7
print(result["is_stunted"])         # True
print(result["advice"]["recovery_steps"])
```

### 5. Export to ONNX
```bash
python main.py export \
    --checkpoint runs/exp01/checkpoints/best_model.pt \
    --out        model.onnx \
    --format     onnx \
    --opset      17
```

---

## Output Files

| File | Description |
|---|---|
| `checkpoints/best_model.pt` | Best validation-loss checkpoint |
| `checkpoints/last_model.pt` | Final epoch checkpoint |
| `output/training_history.json` | Loss + metrics per epoch |
| `output/training_curves.png` | Loss / accuracy / width-MAE plots |
| `output/test_metrics.json` | Final test-set performance |
| `output/confusion_matrices.png` | Row-normalised heatmaps (health + month) |
| `output/sample_predictions.csv` | Per-image predictions on test set |
| `output/val_epoch_NNN.json` | Per-epoch validation metrics |

---

## Training Features

- **Mixed precision (AMP)** — automatic on CUDA; ~2× speedup
- **Gradient clipping** — `max_norm=1.0` for stability
- **Cosine annealing LR** — smooth learning rate decay
- **Backbone warm-up** — freeze EfficientNet for first N epochs; train heads only, then unfreeze all
- **Weighted random sampler** — joint (health, month) stratification to balance rare classes
- **Class-weighted CE** — inverse-frequency weighting for both classification heads
- **Label smoothing** — optional (default 0.05)
- **RandAugment + RandomErasing** — strong augmentation mode

---

## Colab

Open `notebooks/train_colab.ipynb` in Google Colab (GPU: T4).  
All you need to set: `DATASET_ROOT`, `WIDTH_CSV`, `RUN_DIR` in Cell 4.

---

## Module Layout

```
pineapple_pipeline/
├── main.py                  # CLI: train / infer / export
├── requirements.txt
├── notebooks/
│   └── train_colab.ipynb    # end-to-end Colab notebook
└── pineapple/
    ├── __init__.py          # exposes predict()
    ├── constants.py         # class names, expected widths
    ├── model.py             # MultiTaskNet (EfficientNet-B0)
    ├── data.py              # dataset scanning, splits, DataLoaders
    ├── losses.py            # MultiTaskLoss, class weights, metrics
    ├── train.py             # full training loop
    ├── infer.py             # predict() public API
    ├── export.py            # ONNX / TorchScript export
    ├── gradcam.py           # Grad-CAM visualisation
    ├── ruler.py             # Hough-based ruler calibration
    ├── advice.py            # farmer advice generation
    └── utils.py             # seed, device, early stopping, plots
```
