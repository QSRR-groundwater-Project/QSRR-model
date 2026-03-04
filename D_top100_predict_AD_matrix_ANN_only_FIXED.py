# D_top100_predict_AD_matrix_ANN_only_FIXED.py
# =============================================================================
# FIXED version of Step D (ANN-only)
# - Fixes numpy string concatenation bug when building "Quadrant" column
#
# Run:
#   python scripts/D_top100_predict_AD_matrix_ANN_only_FIXED.py
# =============================================================================

import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPE_DIR = os.path.dirname(BASE_DIR)

TOP_PATH = os.path.join(PIPE_DIR, "outputs_A_rdkit_build", "top100_rdkit_raw.xlsx")
TRAIN_PATH = os.path.join(PIPE_DIR, "outputs_A_rdkit_build", "train1492_rdkit_raw.xlsx")

ART_DIR = os.path.join(PIPE_DIR, "outputs_B_ANN_1492")
IMPUTER_PATH = os.path.join(ART_DIR, "preprocess_imputer.pkl")
SCALER_PATH = os.path.join(ART_DIR, "preprocess_scaler.pkl")
FEAT_PATH = os.path.join(ART_DIR, "feature_columns.txt")
ANN_PATH = os.path.join(ART_DIR, "ann_model.keras")

OUT_DIR = os.path.join(PIPE_DIR, "outputs_D_top100_screening_ANN")
os.makedirs(OUT_DIR, exist_ok=True)

for p in [TOP_PATH, TRAIN_PATH, IMPUTER_PATH, SCALER_PATH, FEAT_PATH, ANN_PATH]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing: {p}")

imputer = joblib.load(IMPUTER_PATH)
scaler = joblib.load(SCALER_PATH)
with open(FEAT_PATH, "r", encoding="utf-8") as f:
    feature_cols = [line.strip() for line in f.readlines() if line.strip()]

ann = tf.keras.models.load_model(ANN_PATH)

df_top = pd.read_excel(TOP_PATH)
df_train = pd.read_excel(TRAIN_PATH)

meta_cols = [c for c in ["SMILES", "Detection_frequency_records", "Rank", "Individual compound", "StdInChIKey", "CAS No."] if c in df_top.columns]

X_train_raw = df_train[feature_cols].apply(pd.to_numeric, errors="coerce").values
X_top_raw = df_top[feature_cols].apply(pd.to_numeric, errors="coerce").values
X_train_raw = np.where(np.isfinite(X_train_raw), X_train_raw, np.nan)
X_top_raw = np.where(np.isfinite(X_top_raw), X_top_raw, np.nan)

X_train = scaler.transform(imputer.transform(X_train_raw))
X_top = scaler.transform(imputer.transform(X_top_raw))

# AD
centroid = np.mean(X_train, axis=0)
dist_train = np.linalg.norm(X_train - centroid, axis=1)
dist_top = np.linalg.norm(X_top - centroid, axis=1)
threshold = np.quantile(dist_train, 0.95)
in_domain = dist_top <= threshold

pred_ann = ann.predict(X_top, verbose=0).reshape(-1)

out = df_top[meta_cols].copy() if meta_cols else pd.DataFrame(index=df_top.index)
out["AD_in_domain"] = in_domain.astype(int)
out["AD_distance_to_centroid"] = dist_top
out["Pred_RI_ANN"] = pred_ann

out_path = os.path.join(OUT_DIR, "top100_predictions_with_AD.xlsx")
out.to_excel(out_path, index=False)

# Mobility × Occurrence matrix (median cutoffs)
if "Detection_frequency_records" in out.columns:
    RI_med = float(np.nanmedian(out["Pred_RI_ANN"].values))
    F_med = float(np.nanmedian(out["Detection_frequency_records"].values))

    mobility = np.where(out["Pred_RI_ANN"] < RI_med, "mobile", "retarded")
    occurrence = np.where(out["Detection_frequency_records"] >= F_med, "high_occ", "low_occ")

    out["Mobility_class"] = mobility
    out["Occurrence_class"] = occurrence

    # ✅ FIX: safe string concatenation
    out["Quadrant"] = pd.Series(mobility).astype(str) + " + " + pd.Series(occurrence).astype(str)

    matrix_path = os.path.join(OUT_DIR, "mobility_occurrence_matrix.xlsx")
    out.to_excel(matrix_path, index=False)

    vc = out["Quadrant"].value_counts()
    summary_lines = [
        "===== Mobility × Occurrence matrix summary (ANN only) =====",
        f"RI_med (ANN) = {RI_med:.2f}",
        f"F_med = {F_med:.2f}",
        "",
        "Quadrant counts:"
    ]
    for k, v in vc.items():
        summary_lines.append(f"{k}: {v}")

    with open(os.path.join(OUT_DIR, "quadrant_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

print("✅ Step D DONE. Outputs in:", OUT_DIR)