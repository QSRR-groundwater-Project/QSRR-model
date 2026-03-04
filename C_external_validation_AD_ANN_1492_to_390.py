# C_external_validation_AD_ANN_1492_to_390.py
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPE_DIR = os.path.dirname(BASE_DIR)

TRAIN_PATH = os.path.join(PIPE_DIR, "outputs_A_rdkit_build", "train1492_rdkit_raw.xlsx")
EXT_PATH = os.path.join(PIPE_DIR, "outputs_A_rdkit_build", "external390_rdkit_raw.xlsx")

ART_DIR = os.path.join(PIPE_DIR, "outputs_B_ANN_1492")
OUT_DIR = os.path.join(PIPE_DIR, "outputs_C_external390_eval_ANN")
os.makedirs(OUT_DIR, exist_ok=True)

IMPUTER_PATH = os.path.join(ART_DIR, "preprocess_imputer.pkl")
SCALER_PATH = os.path.join(ART_DIR, "preprocess_scaler.pkl")
FEAT_PATH = os.path.join(ART_DIR, "feature_columns.txt")
ANN_PATH = os.path.join(ART_DIR, "ann_model.keras")

for p in [TRAIN_PATH, EXT_PATH, IMPUTER_PATH, SCALER_PATH, FEAT_PATH, ANN_PATH]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing: {p}")

imputer = joblib.load(IMPUTER_PATH)
scaler = joblib.load(SCALER_PATH)
with open(FEAT_PATH, "r", encoding="utf-8") as f:
    feature_cols = [line.strip() for line in f.readlines() if line.strip()]

ann = tf.keras.models.load_model(ANN_PATH)

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

def metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae

r2_all, rmse_all, mae_all = metrics(y_ext, pred_ann)
r2_in, rmse_in, mae_in = metrics(y_ext[in_domain], pred_ann[in_domain])

lines = []
lines.append("===== External Validation (ANN only; 1492 -> 390) + AD =====")
lines.append(f"External n={len(y_ext)} | in-AD n={int(np.sum(in_domain))} ({np.mean(in_domain)*100:.2f}%)")
lines.append(f"AD threshold (95% train dist) = {threshold:.4f}")
lines.append("")
lines.append(f"[ANN] ALL: R2={r2_all:.4f}, RMSE={rmse_all:.2f}, MAE={mae_all:.2f} | in-AD: R2={r2_in:.4f}, RMSE={rmse_in:.2f}, MAE={mae_in:.2f}")

txt = "\n".join(lines) + "\n"
print("\n" + txt)

with open(os.path.join(OUT_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
    f.write(txt)

with open(os.path.join(OUT_DIR, "AD_summary.txt"), "w", encoding="utf-8") as f:
    f.write(lines[1] + "\n" + lines[2] + "\n")

pd.DataFrame({
    "y_true": y_ext,
    "pred_ann": pred_ann,
    "in_domain": in_domain.astype(int),
    "dist_to_centroid": dist_ext
}).to_csv(os.path.join(OUT_DIR, "predictions.csv"), index=False)

print("✅ Step C DONE. Outputs in:", OUT_DIR)
