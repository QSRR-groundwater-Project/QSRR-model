# C_external_validation_AD_ANN.py
# =============================================================================
# ANN-only pipeline | Step C
# External validation + AD:
#   Train space: 1492
#   External set: 390
#
# INPUT:
#   outputs_A_rdkit_build/train1492_rdkit_raw.xlsx
#   outputs_A_rdkit_build/external390_rdkit_raw.xlsx
#   outputs_B_ANN_1492/ (preprocessor + ANN model)
#
# OUTPUT:
#   outputs_C_external390_eval_ANN/
#     - metrics.txt
#     - AD_summary.txt
#     - predictions.csv
#
# Run:
#   python scripts/C_external_validation_AD_ANN_1492_to_390.py
# =============================================================================

import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.interpolate import interp1d, UnivariateSpline


class BinningCalibrator:
    """Equal-frequency binning calibrator with linear interpolation."""
    def __init__(self, n_bins=10):
        self.n_bins = n_bins
        self.bin_edges_ = None
        self.bin_centers_pred_ = None
        self.bin_means_true_ = None

    def fit(self, y_pred, y_true):
        quantiles = np.linspace(0, 100, self.n_bins + 1)
        self.bin_edges_ = np.unique(np.percentile(y_pred, quantiles))
        centers, means = [], []
        for i in range(len(self.bin_edges_) - 1):
            lower = self.bin_edges_[i]
            upper = self.bin_edges_[i + 1]
            if i == len(self.bin_edges_) - 2:
                mask = (y_pred >= lower) & (y_pred <= upper)
            else:
                mask = (y_pred >= lower) & (y_pred < upper)
            if np.sum(mask):
                centers.append(np.median(y_pred[mask]))
                means.append(np.mean(y_true[mask]))
        self.bin_centers_pred_ = np.array(centers)
        self.bin_means_true_ = np.array(means)
        return self

    def predict(self, y_pred):
        if len(self.bin_centers_pred_) < 2:
            return np.full_like(y_pred, np.mean(self.bin_means_true_))
        f = interp1d(
            self.bin_centers_pred_,
            self.bin_means_true_,
            kind="linear",
            bounds_error=False,
            fill_value=(self.bin_means_true_[0], self.bin_means_true_[-1]),
        )
        return f(y_pred)


class SplineCalibrator:
    """Compatibility class for loading pickled spline calibrators."""
    def __init__(self, s=0.5):
        self.s = s
        self.spline = None

    def fit(self, y_pred, y_true):
        eps = 1e-9 * (np.std(y_pred) if np.std(y_pred) > 0 else 1.0)
        y_pred_jitter = y_pred + np.random.normal(0, eps, size=len(y_pred))
        order = np.argsort(y_pred_jitter)
        self.spline = UnivariateSpline(
            y_pred_jitter[order], y_true[order], s=self.s, ext="const"
        )
        return self

    def predict(self, y_pred):
        return self.spline(y_pred)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPE_DIR = BASE_DIR

TRAIN_PATH = os.path.join(PIPE_DIR, "train1492_rdkit_raw.xlsx")
EXT_PATH = os.path.join(PIPE_DIR, "external390_rdkit_raw.xlsx")

IMPUTER_PATH = os.path.join(PIPE_DIR, "preprocess_imputer.pkl")
SCALER_PATH = os.path.join(PIPE_DIR, "preprocess_scaler.pkl")
FEAT_PATH = os.path.join(PIPE_DIR, "feature_columns.txt")
ANN_PATH = os.path.join(PIPE_DIR, "ann_model.keras")
ENSEMBLE_PATH = os.path.join(PIPE_DIR,"ensemble_calibrator.pkl")

OUT_DIR = os.path.join(PIPE_DIR, "outputs_C_external390_eval_ANN")
os.makedirs(OUT_DIR, exist_ok=True)

for p in [TRAIN_PATH, EXT_PATH, IMPUTER_PATH, SCALER_PATH, FEAT_PATH, ANN_PATH, ENSEMBLE_PATH]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing: {p}")

imputer = joblib.load(IMPUTER_PATH)
scaler = joblib.load(SCALER_PATH)
with open(FEAT_PATH, "r", encoding="utf-8") as f:
    feature_cols = [line.strip() for line in f.readlines() if line.strip()]

ann = tf.keras.models.load_model(ANN_PATH)
ensemble = joblib.load(ENSEMBLE_PATH)

calibrators = ensemble["calibrators"]
weights = ensemble["weights"]

df_train = pd.read_excel(TRAIN_PATH)
df_ext = pd.read_excel(EXT_PATH)

X_train_raw = df_train[feature_cols].apply(pd.to_numeric, errors="coerce").values
X_ext_raw = df_ext[feature_cols].apply(pd.to_numeric, errors="coerce").values
y_ext = df_ext["target"].astype(float).values

X_train_raw = np.where(np.isfinite(X_train_raw), X_train_raw, np.nan)
X_ext_raw = np.where(np.isfinite(X_ext_raw), X_ext_raw, np.nan)

X_train = scaler.transform(imputer.transform(X_train_raw))
X_ext = scaler.transform(imputer.transform(X_ext_raw))

# AD
centroid = np.mean(X_train, axis=0)
dist_train = np.linalg.norm(X_train - centroid, axis=1)
dist_ext = np.linalg.norm(X_ext - centroid, axis=1)
threshold = np.quantile(dist_train, 0.95)
in_domain = dist_ext <= threshold

pred_ann = ann.predict(X_ext, verbose=0).reshape(-1)

def ensemble_predict(pred):
    pred_final = np.zeros_like(pred)

    for name,cal in calibrators.items():
        if "Linear" in name:
            p = cal.predict(pred.reshape (-1,1))
        else:
            p = cal.predict(pred)
        pred_final +=weights [name] * p
    return pred_final

pred_cal = ensemble_predict(pred_ann)

def metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae

r2_raw, rmse_raw, mae_raw = metrics(y_ext, pred_ann)
r2_all, rmse_all, mae_all = metrics(y_ext, pred_cal)
r2_in, rmse_in, mae_in = metrics(y_ext[in_domain], pred_cal[in_domain])

lines = []
lines.append("===== External Validation (ANN only; 1492 -> 390) + AD =====")
lines.append(f"External n={len(y_ext)} | in-AD n={int(np.sum(in_domain))} ({np.mean(in_domain)*100:.2f}%)")
lines.append(f"AD threshold (95% train dist) = {threshold:.4f}")
lines.append("")
lines.append(f"[Raw ANN] ALL: R2={r2_raw:.4f}, RMSE={rmse_raw:.2f}, MAE={mae_raw:.2f}")
lines.append(f"[Calibrated Ensemble] ALL: R2={r2_all:.4f}, RMSE={rmse_all:.2f}, MAE={mae_all:.2f} | in-AD: R2={r2_in:.4f}, RMSE={rmse_in:.2f}, MAE={mae_in:.2f}")


txt = "\n".join(lines) + "\n"
print("\n" + txt)

with open(os.path.join(OUT_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
    f.write(txt)

with open(os.path.join(OUT_DIR, "AD_summary.txt"), "w", encoding="utf-8") as f:
    f.write(lines[1] + "\n" + lines[2] + "\n")

pd.DataFrame({
    "y_true": y_ext,
    "pred_ann": pred_ann,
    "pred_calibrated": pred_cal,
    "in_domain": in_domain.astype(int),
    "dist_to_centroid": dist_ext
}).to_csv(os.path.join(OUT_DIR, "predictions.csv"), index=False)

print("✅ Step C DONE. Outputs in:", OUT_DIR)
