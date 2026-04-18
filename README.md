# FaceAge ClientScan

> **🏆 State-of-the-art face-only age estimation — MAE 3.555 on LAGENDA 84k, beating MiVOLO v2 paper (3.650) - the best specific task model using only the face crop.**

Age and gender estimation from face crops using **DINOv3-ViT-L** backbone with CORAL ordinal regression.

- 🤗 **Model**: [TrungTran/faceage_ClientScan](https://huggingface.co/TrungTran/faceage_ClientScan)
- 🚀 **Live Demo**: [HuggingFace Spaces](https://huggingface.co/spaces/TrungTran/faceage_ClientScan)

---

## Performance (LAGENDA 84k benchmark)

| Model | Input | MAE ↓ | CS@5 ↑ | Gender Acc ↑ |
|-------|-------|--------|--------|-------------|
| **FaceAge ClientScan (ours)** | **face-only** | **3.555** | **75.5%** | **97.75%** |
| MiVOLO v2 (paper) | face + body | 3.650 | 74.48% | 97.99% |
| MiVOLO v1 (paper) | face + body | 3.990 | 71.27% | 97.36% |
| MiVOLO v2 (measured, face+body) | face + body | 3.859 | 76.5% | 96.96% |
| MiVOLO v2 (measured, face-only) | face only | 4.224 | 69.7% | 96.05% |

**Key result**: FaceAge ClientScan achieves **MAE=3.555** using only the face crop — no body information needed — outperforming MiVOLO v2's paper claim of 3.650 which requires both face and body bounding boxes.

### Per age-group MAE

| Age Group | n | MiVOLO v2 best | FaceAge ClientScan | Delta |
|-----------|--:|---------------:|-------------------:|------:|
| 0–12      | 15,369 | 1.677 | **1.548** | ✅ −0.129 |
| 13–17     | 3,930  | 3.365 | **2.845** | ✅ −0.520 |
| 18–25     | 9,975  | 2.989 | **2.877** | ✅ −0.112 |
| 26–35     | 10,303 | **3.348** | 3.775 | ❌ +0.427 |
| 36–50     | 19,234 | 4.484 | **4.195** | ✅ −0.289 |
| 51–65     | 16,350 | 4.794 | **4.329** | ✅ −0.465 |
| 66+       | 9,031  | 6.310 | **5.013** | ✅ −1.297 |
| **Overall** | **84,192** | 3.859 | **3.555** | ✅ −0.304 |

Wins **6/7 age groups** vs MiVOLO v2.

---

## Quick Start (ONNX — no PyTorch needed)

```bash
cd inference
pip install -r requirements.txt
```

### Single image

```bash
python infer_onnx.py --image photo.jpg --bbox 120 80 300 320
# --bbox X0 Y0 X1 Y1 of the face in the full image
# ONNX model auto-downloads from HuggingFace on first run
```

### LAGENDA MAE benchmark

```bash
python infer_onnx.py \
    --lagenda_dir   /path/to/lagenda \
    --annotation_csv lagenda_test.csv \
    --batch_size    256
```

### Python API

```python
import numpy as np
from PIL import Image
from inference.infer_onnx import FaceAgeModel, crop_face

model = FaceAgeModel()   # auto-downloads ONNX from HuggingFace

img  = np.asarray(Image.open("photo.jpg").convert("RGB"))
face = crop_face(img, x0=120, y0=80, x1=300, y1=320)  # 10% padding applied automatically
out  = model.predict(face)

print(out)
# {'age': 34.2, 'gender': 'male', 'gender_conf': 0.981}
```

> **Important**: always use `crop_face()` with the default `pad=0.10` — the 10% proportional padding is required to reproduce MAE=3.555. Without it, MAE degrades to ~3.758.

---

## Architecture

```
Face [B, 3, 224, 224]  (+ 10% proportional bbox padding)
    ↓
DINOv3-ViT-L/16  (307M params, pretrained on LVD-1.68B)
    ↓ pooler_output
[B, 1024]
    ↓ LayerNorm → Linear(1024→512) → GELU → Dropout(0.1)
[B, 512]
    ├── age_head:    Linear(512, 100) → CORAL → age ∈ [0, 100]
    └── gender_head: Linear(512, 2)  → softmax → {female, male}
```

**CORAL ordinal regression**: age = Σ σ(logit_k) for k=0..99.

---

## Training

Multi-phase fine-tuning on DINOv3-ViT-L:

| Phase | Backbone | LR | Key change |
|-------|----------|----|-----------|
| 1 | Frozen (all 24 blocks) | 1e-3 | Head training only |
| 2 | Top 4 blocks unfrozen | 1e-4 | Partial fine-tuning |
| 3 | All blocks unfrozen | 3e-5 | Full fine-tuning |
| 4 | All blocks | 3e-6 | Age-group reweighting → MAE=3.555 |

Training data: Our Collection (4M images).

## Testing
Testing data: [Lagenda test set](https://github.com/WildChlamydia/MiVOLO)

---

## Citation

```bibtex
@misc{faceage-clientscan-2026,
  title  = {FaceAge ClientScan: Face-Only Age & Gender Estimation by pseudo label and light model},
  author = {Trung Thanh Tran},
  year   = {2026},
  url    = {https://huggingface.co/TrungTran/faceage_ClientScan}
}
```
