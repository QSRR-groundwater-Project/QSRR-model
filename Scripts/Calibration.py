# Calibration.py
# ============================================================
# Calibration curve (reliability diagram) and Brier Skill Score
# ============================================================

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.interpolate import UnivariateSpline, interp1d

# ============================================================
# File paths
# ============================================================

DATA_PATH = "train1492_rdkit_raw.xlsx"
MODEL_PATH = "ann_model.keras"
IMPUTER_PATH = "preprocess_imputer.pkl"
SCALER_PATH = "preprocess_scaler.pkl"
SPLIT_PATH = "split_indices.csv"

# ============================================================
# Load data
# ============================================================

df = pd.read_excel(DATA_PATH)
feature_cols = [c for c in df.columns if c not in ["SMILES", "target"]]
X = df[feature_cols].apply(pd.to_numeric, errors="coerce").values
X = np.where(np.isfinite(X), X, np.nan)
y = df["target"].astype(float).values

# ============================================================
# Restore the split indices
# ============================================================

split = pd.read_csv(SPLIT_PATH)
idx_train = split.loc[split["split"] == "train", "row_index"].values
idx_val   = split.loc[split["split"] == "val",   "row_index"].values
idx_test  = split.loc[split["split"] == "test",  "row_index"].values

print("=" * 70)
print(f"Train={len(idx_train)}  Val={len(idx_val)}  Test={len(idx_test)}")
print("=" * 70)

# ============================================================
# Load preprocessing objects
# ============================================================

imputer = joblib.load(IMPUTER_PATH)
scaler = joblib.load(SCALER_PATH)

X_train = scaler.transform(imputer.transform(X[idx_train]))
X_val   = scaler.transform(imputer.transform(X[idx_val]))
X_test  = scaler.transform(imputer.transform(X[idx_test]))

# ============================================================
# Load the pre‑trained ANN
# ============================================================

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# ============================================================
# Raw predictions
# ============================================================

pred_train = model.predict(X_train, verbose=0).reshape(-1)
pred_val   = model.predict(X_val,   verbose=0).reshape(-1)
pred_test  = model.predict(X_test,  verbose=0).reshape(-1)

# ============================================================
# Metric functions
# ============================================================

def calc_metrics(y_true, pred):
    r2 = r2_score(y_true, pred)
    rmse = np.sqrt(mean_squared_error(y_true, pred))
    mae = mean_absolute_error(y_true, pred)
    return r2, rmse, mae

def print_metrics(title, y_true, pred):
    r2, rmse, mae = calc_metrics(y_true, pred)
    print(f"\n{title}")
    print(f"R2   = {r2:.4f}")
    print(f"RMSE = {rmse:.2f}")
    print(f"MAE  = {mae:.2f}")
    return r2, rmse, mae

print("\n================ BEFORE CALIBRATION ================")
train_before = print_metrics("TRAIN", y[idx_train], pred_train)
val_before   = print_metrics("VALIDATION", y[idx_val], pred_val)
test_before  = print_metrics("TEST", y[idx_test], pred_test)

# ============================================================
# Calibrator definitions (unchanged)
# ============================================================

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

# ============================================================
# Candidate calibrators
# ============================================================

calibrator_candidates = {}

calibrator_candidates["Linear"] = LinearRegression()
calibrator_candidates["Isotonic"] = IsotonicRegression(out_of_bounds="clip")

for bins in [5, 7, 10, 15, 20]:
    calibrator_candidates[f"Binning_{bins}"] = BinningCalibrator(n_bins=bins)

# ============================================================
# 5‑fold cross‑validation on the validation set to select the best calibrator
# ============================================================

print("\n=== Cross‑validation selection (5‑fold CV on validation set) ===")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse_dict = {}

for name, cal in calibrator_candidates.items():
    cv_rmse_list = []
    for train_idx, val_idx in kf.split(pred_val):
        pred_tr = pred_val[train_idx]
        y_tr = y[idx_val][train_idx]
        pred_va = pred_val[val_idx]
        y_va = y[idx_val][val_idx]

        if "Linear" in name:
            cal_copy = LinearRegression()
            cal_copy.fit(pred_tr.reshape(-1, 1), y_tr)
            y_pred_cv = cal_copy.predict(pred_va.reshape(-1, 1))
        elif "Isotonic" in name:
            cal_copy = IsotonicRegression(out_of_bounds="clip")
            cal_copy.fit(pred_tr, y_tr)
            y_pred_cv = cal_copy.predict(pred_va)
        elif "Binning" in name:
            n_bins = int(name.split('_')[1])
            cal_copy = BinningCalibrator(n_bins=n_bins)
            cal_copy.fit(pred_tr, y_tr)
            y_pred_cv = cal_copy.predict(pred_va)
        elif "Spline" in name:
            s_val = float(name.split('=')[1])
            cal_copy = SplineCalibrator(s=s_val)
            cal_copy.fit(pred_tr, y_tr)
            y_pred_cv = cal_copy.predict(pred_va)
        else:
            raise ValueError(f"Unknown calibrator: {name}")

        _, rmse_cv, _ = calc_metrics(y_va, y_pred_cv)
        cv_rmse_list.append(rmse_cv)

    mean_rmse = np.mean(cv_rmse_list)
    cv_rmse_dict[name] = mean_rmse
    print(f"  {name:20s}  CV RMSE = {mean_rmse:.2f}")

best_cv_rmse = min(cv_rmse_dict.values())
candidates_within_5pct = [n for n, rmse in cv_rmse_dict.items() if rmse <= best_cv_rmse * 1.05]
preference = ["Linear", "Isotonic"] + [n for n in cv_rmse_dict.keys() if n.startswith("Binning")] + [n for n in cv_rmse_dict.keys() if n.startswith("Spline")]
best_name = None
for name in preference:
    if name in candidates_within_5pct:
        best_name = name
        break
if best_name is None:
    best_name = min(cv_rmse_dict, key=cv_rmse_dict.get)

print(f"\nSelected best calibrator (with 5%% tolerance for simpler models): {best_name}")

# ============================================================
# Fit the best single calibrator on the full validation set
# ============================================================

def fit_calibrator(name, cal, pred_val, y_val):
    if "Linear" in name:
        cal.fit(pred_val.reshape(-1, 1), y_val)
    elif "Isotonic" in name:
        cal.fit(pred_val, y_val)
    elif "Binning" in name:
        cal.fit(pred_val, y_val)
    elif "Spline" in name:
        cal.fit(pred_val, y_val)
    else:
        raise ValueError(f"Unknown calibrator: {name}")
    return cal

best_calibrator = fit_calibrator(best_name, calibrator_candidates[best_name], pred_val, y[idx_val])
joblib.dump(best_calibrator, "prediction_calibrator.pkl")
print("Saved best single calibrator to prediction_calibrator.pkl")

# ============================================================
# Build an ensemble calibrator (weighted average by inverse CV RMSE)
# ============================================================

print("\n=== Building Ensemble Calibrator (weighted by 1/CV_RMSE) ===")
all_calibrators = {}
for name, cal in calibrator_candidates.items():
    if "Linear" in name:
        cal_copy = LinearRegression()
        cal_copy.fit(pred_val.reshape(-1, 1), y[idx_val])
    elif "Isotonic" in name:
        cal_copy = IsotonicRegression(out_of_bounds="clip")
        cal_copy.fit(pred_val, y[idx_val])
    elif "Binning" in name:
        n_bins = int(name.split('_')[1])
        cal_copy = BinningCalibrator(n_bins=n_bins)
        cal_copy.fit(pred_val, y[idx_val])
    elif "Spline" in name:
        s_val = float(name.split('=')[1])
        cal_copy = SplineCalibrator(s=s_val)
        cal_copy.fit(pred_val, y[idx_val])
    else:
        continue
    all_calibrators[name] = cal_copy

weights = {}
for name in all_calibrators.keys():
    weights[name] = 1.0 / cv_rmse_dict[name]
total_weight = sum(weights.values())
weights = {k: v/total_weight for k, v in weights.items()}
print("Ensemble weights:", {k: f"{v:.3f}" for k, v in weights.items()})

def ensemble_predict(pred, calibrators, weights):
    pred_ensemble = np.zeros_like(pred)
    for name, cal in calibrators.items():
        if "Linear" in name:
            p = cal.predict(pred.reshape(-1, 1))
        else:
            p = cal.predict(pred)
        pred_ensemble += weights[name] * p
    return pred_ensemble

pred_train_ensemble = ensemble_predict(pred_train, all_calibrators, weights)
pred_val_ensemble   = ensemble_predict(pred_val,   all_calibrators, weights)
pred_test_ensemble  = ensemble_predict(pred_test,  all_calibrators, weights)

# Predictions from the single best calibrator (for comparison)
def apply_single_calibrator(name, cal, pred):
    if "Linear" in name:
        return cal.predict(pred.reshape(-1, 1))
    else:
        return cal.predict(pred)

pred_train_single = apply_single_calibrator(best_name, best_calibrator, pred_train)
pred_val_single   = apply_single_calibrator(best_name, best_calibrator, pred_val)
pred_test_single  = apply_single_calibrator(best_name, best_calibrator, pred_test)

print("\n================ AFTER CALIBRATION (Single Best) ================")
train_after_single = print_metrics("TRAIN (single)", y[idx_train], pred_train_single)
val_after_single   = print_metrics("VALIDATION (single)", y[idx_val], pred_val_single)
test_after_single  = print_metrics("TEST (single)", y[idx_test], pred_test_single)

print("\n================ AFTER CALIBRATION (Ensemble) ================")
train_after_ensemble = print_metrics("TRAIN (ensemble)", y[idx_train], pred_train_ensemble)
val_after_ensemble   = print_metrics("VALIDATION (ensemble)", y[idx_val], pred_val_ensemble)
test_after_ensemble  = print_metrics("TEST (ensemble)", y[idx_test], pred_test_ensemble)

# Save the full ensemble calibrator for reuse in Step C and D
ensemble_package = {
    "calibrators": all_calibrators,
    "weights": weights
}
joblib.dump(ensemble_package, "ensemble_calibrator.pkl")
print("Saved full ensemble calibrator to ensemble_calibrator.pkl")

# ============================================================
# Additional: Calibration curve and Brier Skill Score (on test set)
# ============================================================

def calibration_curve_regression(y_true, y_pred, n_bins=10, title="Calibration curve"):
    """
    Compute and plot a reliability diagram for regression.
    Returns bin centres, bin means, and the Expected Calibration Error (ECE).
    """
    # Sort by prediction and divide into equal‑frequency bins
    order = np.argsort(y_pred)
    y_pred_sorted = y_pred[order]
    y_true_sorted = y_true[order]
    bin_edges = np.percentile(y_pred, np.linspace(0, 100, n_bins+1))
    bin_edges = np.unique(bin_edges)  # merge duplicate edges

    bin_centres = []
    bin_means = []
    for i in range(len(bin_edges)-1):
        lower = bin_edges[i]
        upper = bin_edges[i+1]
        if i == len(bin_edges)-2:
            mask = (y_pred_sorted >= lower) & (y_pred_sorted <= upper)
        else:
            mask = (y_pred_sorted >= lower) & (y_pred_sorted < upper)
        if np.sum(mask) > 0:
            bin_centres.append(np.mean(y_pred_sorted[mask]))
            bin_means.append(np.mean(y_true_sorted[mask]))
    bin_centres = np.array(bin_centres)
    bin_means = np.array(bin_means)

    # Expected Calibration Error (weighted by bin size)
    ece = 0.0
    total = len(y_pred)
    for i in range(len(bin_edges)-1):
        lower = bin_edges[i]
        upper = bin_edges[i+1]
        if i == len(bin_edges)-2:
            mask = (y_pred >= lower) & (y_pred <= upper)
        else:
            mask = (y_pred >= lower) & (y_pred < upper)
        if np.sum(mask) > 0:
            bin_pred_mean = np.mean(y_pred[mask])
            bin_true_mean = np.mean(y_true[mask])
            ece += (np.sum(mask) / total) * np.abs(bin_pred_mean - bin_true_mean)

    # Plot
    plt.figure(figsize=(6,6))
    plt.scatter(bin_centres, bin_means, color='blue', label='Binned observations')
    plt.plot([min(y_pred), max(y_pred)], [min(y_pred), max(y_pred)], 'r--', label='Perfect calibration')
    plt.xlabel('Mean predicted value per bin')
    plt.ylabel('Mean observed value per bin')
    plt.title(f"{title} (ECE = {ece:.3f})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return bin_centres, bin_means, ece

def brier_skill_score(y_true, y_pred, reference=None):
    """
    Compute the Brier Skill Score for regression, defined as:
    BSS = 1 - MSE(pred) / MSE(ref)
    where ref is the mean of y_true if not provided.
    """
    mse_pred = mean_squared_error(y_true, y_pred)
    if reference is None:
        ref_pred = np.full_like(y_true, np.mean(y_true))
    else:
        ref_pred = reference
    mse_ref = mean_squared_error(y_true, ref_pred)
    bss = 1 - (mse_pred / mse_ref) if mse_ref > 0 else np.nan
    return bss, mse_pred, mse_ref

# Compute calibration metrics for the ensemble predictions on the test set
bin_centres, bin_means, ece = calibration_curve_regression(
    y[idx_test], pred_test_ensemble, n_bins=10,
    title="Ensemble Calibration Curve (Test Set)"
)
plt.savefig("calibration_curve_ensemble.png", dpi=150)
plt.close()
print("\nCalibration curve saved as 'calibration_curve_ensemble.png'")
print(f"Expected Calibration Error (ECE) = {ece:.4f}")

# Brier Skill Score (using the mean of observed values as reference)
bss, mse_ens, mse_ref = brier_skill_score(y[idx_test], pred_test_ensemble)
print(f"Ensemble MSE = {mse_ens:.2f}, Reference MSE (mean) = {mse_ref:.2f}")
print(f"Brier Skill Score (BSS) = {bss:.4f}")

# For comparison, also compute for the raw ANN predictions
_, _, ece_raw = calibration_curve_regression(
    y[idx_test], pred_test, n_bins=10,
    title="Raw ANN Calibration Curve (Test Set)"
)
plt.savefig("calibration_curve_raw.png", dpi=150)
plt.close()
print("Calibration curve for raw predictions saved as 'calibration_curve_raw.png'")
print(f"Raw ECE = {ece_raw:.4f}")

bss_raw, mse_raw, _ = brier_skill_score(y[idx_test], pred_test)
print(f"Raw MSE = {mse_raw:.2f}, Raw BSS = {bss_raw:.4f}")

# ============================================================
# Save outputs (using ensemble predictions)
# ============================================================

pred_df = pd.DataFrame({
    "row_index": idx_test,
    "Observed": y[idx_test],
    "ANN_Prediction": pred_test,
    "Calibrated_Prediction": pred_test_ensemble,
})
pred_df.to_excel("calibrated_predictions_ensemble.xlsx", index=False)

with open("calibration_metrics_improved.txt", "w", encoding="utf-8") as f:
    f.write(f"Selected Single Calibrator: {best_name}\n")
    f.write("\nCross-Validation RMSE (5-fold) on validation set:\n")
    for name, rmse in cv_rmse_dict.items():
        f.write(f"  {name:20s}: {rmse:.2f}\n")
    f.write("\n=============== BEFORE CALIBRATION ===============\n")
    f.write(f"TRAIN:      R2={train_before[0]:.4f}  RMSE={train_before[1]:.2f}  MAE={train_before[2]:.2f}\n")
    f.write(f"VALIDATION: R2={val_before[0]:.4f}  RMSE={val_before[1]:.2f}  MAE={val_before[2]:.2f}\n")
    f.write(f"TEST:       R2={test_before[0]:.4f}  RMSE={test_before[1]:.2f}  MAE={test_before[2]:.2f}\n\n")
    f.write("=============== AFTER CALIBRATION (Single Best) ===============\n")
    f.write(f"TRAIN:      R2={train_after_single[0]:.4f}  RMSE={train_after_single[1]:.2f}  MAE={train_after_single[2]:.2f}\n")
    f.write(f"VALIDATION: R2={val_after_single[0]:.4f}  RMSE={val_after_single[1]:.2f}  MAE={val_after_single[2]:.2f}\n")
    f.write(f"TEST:       R2={test_after_single[0]:.4f}  RMSE={test_after_single[1]:.2f}  MAE={test_after_single[2]:.2f}\n\n")
    f.write("=============== AFTER CALIBRATION (Ensemble) ===============\n")
    f.write(f"TRAIN:      R2={train_after_ensemble[0]:.4f}  RMSE={train_after_ensemble[1]:.2f}  MAE={train_after_ensemble[2]:.2f}\n")
    f.write(f"VALIDATION: R2={val_after_ensemble[0]:.4f}  RMSE={val_after_ensemble[1]:.2f}  MAE={val_after_ensemble[2]:.2f}\n")
    f.write(f"TEST:       R2={test_after_ensemble[0]:.4f}  RMSE={test_after_ensemble[1]:.2f}  MAE={test_after_ensemble[2]:.2f}\n\n")
    f.write("=============== CALIBRATION METRICS (Test Set) ===============\n")
    f.write(f"Raw ANN:   MSE = {mse_raw:.2f}, BSS = {bss_raw:.4f}, ECE = {ece_raw:.4f}\n")
    f.write(f"Ensemble:  MSE = {mse_ens:.2f}, BSS = {bss:.4f}, ECE = {ece:.4f}\n\n")
    if "Linear" in best_name:
        f.write(f"Single calibrator equation: y = {best_calibrator.coef_[0]:.6f} * pred + {best_calibrator.intercept_:.6f}\n")
    else:
        f.write(f"Single calibrator model: {best_name}\n")
    f.write("\nEnsemble weights:\n")
    for name, w in weights.items():
        f.write(f"  {name:20s}: {w:.4f}\n")

print("\nDone.")
print("Saved files:")
print("  prediction_calibrator.pkl (single best)")
print("  calibrated_predictions_ensemble.xlsx")
print("  calibration_metrics_improved.txt")
print("  calibration_curve_raw.png")
print("  calibration_curve_ensemble.png")
