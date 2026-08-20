Script_D_top100_predict_AD_matrix.py
=============================================================================
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
import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline, interp1d
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression

class BinningCalibrator:
    """Equal‑frequency binning calibrator with linear interpolation."""
    def __init__(self, n_bins=10):
        self.n_bins = n_bins
        self.bin_edges_ = None
        self.bin_centers_pred_ = None
        self.bin_means_true_ = None

    def fit(self, y_pred, y_true):
        quantiles = np.linspace(0, 100, self.n_bins + 1)
        self.bin_edges_ = np.unique(np.percentile(y_pred, quantiles))
        bin_centers_pred = []
        bin_means_true = []
        for i in range(len(self.bin_edges_) - 1):
            lower = self.bin_edges_[i]
            upper = self.bin_edges_[i + 1]
            mask = (y_pred >= lower) & (y_pred < upper)
            if i == len(self.bin_edges_) - 2:   # include upper bound for the last bin
                mask = (y_pred >= lower) & (y_pred <= upper)
            if np.sum(mask) > 0:
                bin_centers_pred.append(np.median(y_pred[mask]))
                bin_means_true.append(np.mean(y_true[mask]))
        self.bin_centers_pred_ = np.array(bin_centers_pred)
        self.bin_means_true_ = np.array(bin_means_true)
        return self

    def predict(self, y_pred):
        if len(self.bin_centers_pred_) < 2:
            return np.full_like(y_pred, np.mean(self.bin_means_true_))
        f = interp1d(self.bin_centers_pred_, self.bin_means_true_,
                     kind='linear', bounds_error=False,
                     fill_value=(self.bin_means_true_[0], self.bin_means_true_[-1]))
        return f(y_pred)

class SplineCalibrator:
    """Smoothing spline calibrator using ext='const' and jitter for ties."""
    def __init__(self, s=0.5):
        self.s = s
        self.spline = None

    def fit(self, y_pred, y_true):
        eps = 1e-9 * (np.std(y_pred) if np.std(y_pred) > 0 else 1.0)
        y_pred_jitter = y_pred + np.random.normal(0, eps, size=len(y_pred))
        order = np.argsort(y_pred_jitter)
        self.spline = UnivariateSpline(
            y_pred_jitter[order], y_true[order],
            s=self.s, ext='const'
        )
        return self

    def predict(self, y_pred):
        return self.spline(y_pred)


def ensemble_predict(pred, calibrators, weights):
    pred_ensemble = np.zeros_like(pred)
    for name, cal in calibrators.items():
        if "Linear" in name:
            p = cal.predict(pred.reshape(-1, 1))
        else:
            p = cal.predict(pred)
        pred_ensemble += weights[name] * p
    return pred_ensemble



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPE_DIR = BASE_DIR

# Auto-detect model artifact directory
candidate_dirs = [
    os.path.join(PIPE_DIR, "outputs_B_ANN_1492"),
    os.path.join(PIPE_DIR, "outputs_B_ANN"),
    PIPE_DIR,
]
ART_DIR = None
for d in candidate_dirs:
    if os.path.exists(os.path.join(d, "ann_model.keras")):
        ART_DIR = d
        break
if ART_DIR is None:
    ART_DIR = candidate_dirs[0]


TOP_PATH = os.path.join(PIPE_DIR, "top100_rdkit_raw.xlsx")
TRAIN_PATH = os.path.join(PIPE_DIR, "train1492_rdkit_raw.xlsx")

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

USE_CALIBRATION = True
CALIBRATOR_PATH = os.path.join(
    ART_DIR,
    "ensemble_calibrator.pkl"
    )
ensemble_package = joblib.load(CALIBRATOR_PATH)
calibrators = ensemble_package["calibrators"]
weights = ensemble_package["weights"]

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

if USE_CALIBRATION:
    pred_used = ensemble_predict(pred_ann, calibrators, weights)
else:
    pred_used = pred_ann

out["Pred_RI_ANN"] = pred_used
out["Pred_RI_RAW"] = pred_ann

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
    # ==========================================================
    # Figure: Mobility × Occurrence Matrix
    # ==========================================================
    fig, ax = plt.subplots(figsize=(8,6))
    color_map={"mobile + high_occ":"#d73027","mobile + low_occ":"#fc8d59","retarded + high_occ":"#4575b4","retarded + low_occ":"#91bfdb"}
    for quad,color in color_map.items():
        df_plot=out[out["Quadrant"]==quad]
        if len(df_plot)==0: continue
        ax.scatter(df_plot["Detection_frequency_records"],df_plot["Pred_RI_ANN"],s=70,color=color,edgecolors="black",linewidth=0.5,alpha=0.85,label=quad.replace("_"," "))
    ax.axhline(RI_med,color="black",linestyle="--",linewidth=1.2)
    ax.axvline(F_med,color="black",linestyle="--",linewidth=1.2)
    xmin,xmax=ax.get_xlim(); ymin,ymax=ax.get_ylim(); dx=xmax-xmin; dy=ymax-ymin
  
   
    ax.set_xlabel("Detection frequency (records)")
    ax.set_ylabel("Predicted RI")
    ax.legend(frameon=False,fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,"Figure_Mobility_Occurrence_Matrix.png"),dpi=300,bbox_inches="tight")
    plt.close()




# ==========================================================
# Figure: Priority Score Correlation Heat Map
# ==========================================================

# Mobility score
ri_min = out["Pred_RI_ANN"].min()
ri_max = out["Pred_RI_ANN"].max()

out["Mobility_score"] = (
    ri_max - out["Pred_RI_ANN"]
) / (ri_max - ri_min + 1e-12)

# Occurrence score
occ_min = out["Detection_frequency_records"].min()
occ_max = out["Detection_frequency_records"].max()

out["Occurrence_score"] = (
    out["Detection_frequency_records"] - occ_min
) / (occ_max - occ_min + 1e-12)

# AD score
dist_min = out["AD_distance_to_centroid"].min()
dist_max = out["AD_distance_to_centroid"].max()

out["AD_score"] = (
    dist_max - out["AD_distance_to_centroid"]
) / (dist_max - dist_min + 1e-12)

# ---------- Priority score ----------

out["Priority_score"] = (
      0.45 * out["Mobility_score"]
    + 0.45 * out["Occurrence_score"]
    + 0.10 * out["AD_score"]
)

# ---------- Correlation matrix ----------

corr_df = out[
    [
        "Mobility_score",
        "Occurrence_score",
        "AD_score",
        "Priority_score"
    ]
]

corr = corr_df.corr(method="pearson")

corr.to_excel(
    os.path.join(
        OUT_DIR,
        "PriorityScore_Correlation_Matrix.xlsx"
    )
)

# ---------- Heat map ----------

fig, ax = plt.subplots(figsize=(6,5.5))

im = ax.imshow(
    corr,
    cmap="RdBu_r",
    vmin=-1,
    vmax=1
)

labels = [
    "Mobility\nScore",
    "Occurrence\nScore",
    "AD\nScore",
    "Priority\nScore"
]

ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))

ax.set_xticklabels(labels, fontsize=11)
ax.set_yticklabels(labels, fontsize=11)

plt.setp(
    ax.get_xticklabels(),
    rotation=35,
    ha="right"
)

# Correlation coefficients
for i in range(corr.shape[0]):
    for j in range(corr.shape[1]):

        value = corr.iloc[i, j]

        ax.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=10,
            color="white" if abs(value) > 0.60 else "black"
        )

cbar = plt.colorbar(im)

cbar.set_label(
    "Pearson correlation",
    fontsize=11
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "Figure_PriorityScore_Correlation_Heatmap.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("✅ Step D DONE. Outputs in:", OUT_DIR)
