"""
FaceAge ClientScan — ONNX inference (no PyTorch required).

HuggingFace model : https://huggingface.co/TrungTran/faceage_ClientScan
Live demo         : https://huggingface.co/spaces/TrungTran/faceage_ClientScan

Performance on LAGENDA 84k:
  MAE=3.555  CS@5=75.5%  Gender=97.75%
  → beats MiVOLO v2 paper (3.650) using face-only (no body bbox needed)

Install:
    pip install onnxruntime numpy pillow huggingface_hub

Usage — single image:
    python infer_onnx.py --image photo.jpg --bbox 120 80 300 320

Usage — LAGENDA MAE benchmark:
    python infer_onnx.py \
        --lagenda_dir   /path/to/lagenda \
        --annotation_csv lagenda_test.csv \
        --batch_size    256
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MEAN     = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD      = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_IMG_SIZE = 224
_FACE_PAD = 0.10   # 10% proportional padding — required for MAE=3.555


# ---------------------------------------------------------------------------
# Face crop helper
# ---------------------------------------------------------------------------

def crop_face(image_rgb: np.ndarray,
              x0: float, y0: float, x1: float, y1: float,
              pad: float = _FACE_PAD) -> np.ndarray:
    """Crop face bbox with proportional padding.

    pad=0.10 adds 10% of bbox width/height on each side.
    This is required to reproduce the benchmark MAE=3.555.
    Without padding, MAE degrades to ~3.758.
    """
    h, w = image_rgb.shape[:2]
    pw, ph = (x1 - x0) * pad, (y1 - y0) * pad
    x0 = max(0, int(x0 - pw));  y0 = max(0, int(y0 - ph))
    x1 = min(w, int(x1 + pw));  y1 = min(h, int(y1 + ph))
    return image_rgb[y0:y1, x0:x1]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(img_rgb: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 RGB → [1, 3, 224, 224] float32, ImageNet normalised."""
    pil = Image.fromarray(img_rgb).resize((_IMG_SIZE, _IMG_SIZE), Image.BICUBIC)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[np.newaxis])


def preprocess_batch(imgs: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([preprocess(img) for img in imgs], axis=0)


# ---------------------------------------------------------------------------
# CORAL age decode + gender softmax
# ---------------------------------------------------------------------------

def decode_age(logits: np.ndarray) -> np.ndarray:
    """CORAL ordinal regression: age = Σ sigmoid(logits).  [B,100] → [B]."""
    logits = np.clip(logits, -88.0, 88.0)
    return (1.0 / (1.0 + np.exp(-logits))).sum(axis=-1)


def decode_gender(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Softmax → (class_idx, confidence).  0=female, 1=male."""
    ex    = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = ex / ex.sum(axis=-1, keepdims=True)
    idx   = probs.argmax(axis=-1)
    conf  = probs[np.arange(len(idx)), idx]
    return idx, conf


# ---------------------------------------------------------------------------
# ONNX session
# ---------------------------------------------------------------------------

class FaceAgeModel:
    """FaceAge ClientScan ONNX model wrapper."""

    HF_REPO    = "TrungTran/faceage_ClientScan"
    ONNX_FILE  = "faceage_dino_fp32.onnx"

    def __init__(self, onnx_path: str | None = None, device: str = "cpu"):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("pip install onnxruntime  (or onnxruntime-gpu)")

        if onnx_path is None:
            onnx_path = self._download()

        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if device == "cuda" else ["CPUExecutionProvider"])

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 0

        self.sess    = ort.InferenceSession(onnx_path, sess_options=opts,
                                            providers=providers)
        self.in_name = self.sess.get_inputs()[0].name
        self.fp16    = "float16" in self.sess.get_inputs()[0].type

        print(f"[FaceAge] Model   : {Path(onnx_path).name}")
        print(f"[FaceAge] Provider: {self.sess.get_providers()[0]}")

    def _download(self) -> str:
        from huggingface_hub import hf_hub_download
        print(f"[FaceAge] Downloading from {self.HF_REPO} …")
        return hf_hub_download(repo_id=self.HF_REPO, filename=self.ONNX_FILE)

    def predict(self, face_rgb: np.ndarray) -> dict:
        """Predict age and gender from a single face crop (HxWx3 uint8 RGB).

        The face crop should already have 10% proportional padding applied.
        Use crop_face() to prepare the crop from a full image + bbox.
        """
        x = preprocess(face_rgb)
        if self.fp16:
            x = x.astype(np.float16)
        age_logits, gender_logits = self.sess.run(None, {self.in_name: x})
        age          = float(decode_age(age_logits)[0])
        gidx, gconf  = decode_gender(gender_logits)
        return {
            "age":         round(age, 1),
            "gender":      "male" if int(gidx[0]) == 1 else "female",
            "gender_conf": round(float(gconf[0]), 3),
        }

    def run_batch(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.fp16:
            x = x.astype(np.float16)
        return self.sess.run(None, {self.in_name: x})


# ---------------------------------------------------------------------------
# Single image inference
# ---------------------------------------------------------------------------

def infer_single(model: FaceAgeModel, image_path: str,
                 bbox: list[int] | None) -> dict:
    img = np.asarray(Image.open(image_path).convert("RGB"))

    if bbox:
        x0, y0, x1, y1 = bbox
        face = crop_face(img, x0, y0, x1, y1)
    else:
        face = img  # assume already a face crop

    t0  = time.perf_counter()
    out = model.predict(face)
    ms  = (time.perf_counter() - t0) * 1000

    print(f"\nAge        : {out['age']}")
    print(f"Gender     : {out['gender']}  (conf={out['gender_conf']:.2f})")
    print(f"Inference  : {ms:.1f} ms")
    return out


# ---------------------------------------------------------------------------
# LAGENDA benchmark
# ---------------------------------------------------------------------------

_GENDER_MAP = {"m": 1, "male": 1, "1": 1, "f": 0, "female": 0, "0": 0}


def benchmark(model: FaceAgeModel, lagenda_dir: str, annotation_csv: str,
              batch_size: int, out_json: str | None) -> dict:
    import pandas as pd
    from tqdm import tqdm

    df      = pd.read_csv(annotation_csv)
    df      = df[df["age"] >= 0].reset_index(drop=True)
    img_dir = Path(lagenda_dir)

    pred_ages, gt_ages, pred_genders, gt_genders = [], [], [], []
    batch_imgs, batch_rows = [], []
    missing = 0

    def flush():
        if not batch_imgs:
            return
        x              = preprocess_batch(batch_imgs)
        age_logits, gl = model.run_batch(x)
        ages           = decode_age(age_logits).tolist()
        gidx, _        = decode_gender(gl)
        for row, age, gi in zip(batch_rows, ages, gidx.tolist()):
            pred_ages.append(age)
            gt_ages.append(float(row["age"]))
            pred_genders.append(int(gi))
            gt_genders.append(_GENDER_MAP.get(
                str(row.get("gender", "")).strip().lower(), -1))
        batch_imgs.clear()
        batch_rows.clear()

    t0 = time.perf_counter()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Benchmarking"):
        p = img_dir / row["img_name"]
        if not p.exists():
            p = img_dir / "images" / row["img_name"]
        if not p.exists():
            missing += 1
            continue
        try:
            img  = np.asarray(Image.open(p).convert("RGB"))
            face = crop_face(img,
                             row["face_x0"], row["face_y0"],
                             row["face_x1"], row["face_y1"])
            batch_imgs.append(face)
            batch_rows.append(row)
        except Exception:
            missing += 1
            continue
        if len(batch_imgs) >= batch_size:
            flush()
    flush()
    elapsed = time.perf_counter() - t0

    pa  = np.array(pred_ages)
    ga  = np.array(gt_ages)
    err = np.abs(pa - ga)

    metrics = {
        "MAE":   round(float(err.mean()), 4),
        "RMSE":  round(float(np.sqrt((err**2).mean())), 4),
        "CS5":   round(float((err <= 5).mean() * 100), 2),
        "CS3":   round(float((err <= 3).mean() * 100), 2),
        "MedAE": round(float(np.median(err)), 4),
        "Bias":  round(float((pa - ga).mean()), 4),
        "n":     len(pa),
        "fps":   round(len(pa) / elapsed, 1),
    }

    gm, pm = np.array(gt_genders), np.array(pred_genders)
    valid  = gm >= 0
    if valid.sum() > 0:
        metrics["gender_acc"] = round(
            float((pm[valid] == gm[valid]).mean() * 100), 2)

    groups = [(0,12),(13,17),(18,25),(26,35),(36,50),(51,65),(66,100)]
    metrics["per_group"] = {
        f"{lo}-{hi}": {"mae": round(float(err[(ga>=lo)&(ga<=hi)].mean()), 4),
                       "n":   int(((ga>=lo)&(ga<=hi)).sum())}
        for lo, hi in groups if ((ga>=lo)&(ga<=hi)).sum() > 0
    }

    print("\n" + "=" * 60)
    print("FaceAge ClientScan — LAGENDA Benchmark")
    print("=" * 60)
    print(f"  MAE    : {metrics['MAE']}")
    print(f"  RMSE   : {metrics['RMSE']}")
    print(f"  CS@5   : {metrics['CS5']}%")
    print(f"  CS@3   : {metrics['CS3']}%")
    print(f"  MedAE  : {metrics['MedAE']}")
    print(f"  Bias   : {metrics['Bias']}")
    if "gender_acc" in metrics:
        print(f"  Gender : {metrics['gender_acc']}%")
    print(f"  FPS    : {metrics['fps']}  ({elapsed:.1f}s, n={metrics['n']:,})")
    if missing:
        print(f"  Missing: {missing}")
    print("\nPer age group:")
    for grp, m in metrics["per_group"].items():
        print(f"  {grp:8s}: MAE={m['mae']:.3f}  n={m['n']:,}")

    print("\n" + "-" * 60)
    print(f"{'Model':<36} {'MAE':>6} {'CS@5':>6} {'Gender':>8}")
    print("-" * 60)
    print(f"{'FaceAge ClientScan (ours)':<36} {metrics['MAE']:>6.3f} "
          f"{metrics['CS5']:>5.1f}% "
          f"{metrics.get('gender_acc', 0):>7.2f}%")
    print(f"{'MiVOLO v2 paper [face+body]':<36} {'3.650':>6} {'74.48':>6}% {'97.99':>7}%  ← paper")
    delta = metrics["MAE"] - 3.650
    print(f"\nDelta vs MiVOLO v2 paper: {delta:+.3f}  "
          f"→ {'✅ BEAT' if delta < 0 else '❌ BEHIND'}")
    print("=" * 60)

    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved → {out_json}")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="FaceAge ClientScan — Age & Gender from face crop (ONNX)")
    p.add_argument("--onnx",    default=None,
                   help="Path to .onnx file (auto-downloads from HF if omitted)")
    p.add_argument("--device",  default="cpu", help="cpu or cuda")

    g = p.add_argument_group("Single image")
    g.add_argument("--image", help="Image path")
    g.add_argument("--bbox",  type=int, nargs=4, metavar=("X0","Y0","X1","Y1"),
                   help="Face bbox in full image (optional — skips padding if omitted)")

    b = p.add_argument_group("LAGENDA benchmark")
    b.add_argument("--lagenda_dir",    help="LAGENDA dataset root")
    b.add_argument("--annotation_csv", default="lagenda_test.csv")
    b.add_argument("--batch_size",     type=int, default=256)
    b.add_argument("--out_json",       default=None)

    args = p.parse_args()

    if not args.image and not args.lagenda_dir:
        p.error("Provide --image or --lagenda_dir")

    model = FaceAgeModel(onnx_path=args.onnx, device=args.device)

    if args.image:
        infer_single(model, args.image, args.bbox)

    if args.lagenda_dir:
        benchmark(model, args.lagenda_dir, args.annotation_csv,
                  args.batch_size, args.out_json)


if __name__ == "__main__":
    main()
