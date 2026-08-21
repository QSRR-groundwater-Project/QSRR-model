# train_MLR.py
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.15

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPE_DIR = os.path.dirname(BASE_DIR)

DATA_PATH = "/Users/mac/Desktop/outputs_A_rdkit_build/train1492_rdkit_raw.xlsx"

OUT_DIR = os.path.join(
    PIPE_DIR,
    "outputs_B_MLR_1492"
)

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_excel (DATA_PATH)
assert "target" in df.columns

feature_cols = [c for c in df.columns if c not in ["target", "SMILES"]]

with open(os.path.join(OUT_DIR, "feature_columns.txt"), "w", encoding="utf-8") as f:
    f.write("\\n".join(feature_cols))

X = df[feature_cols].apply(pd.to_numeric, errors="coerce").values
y = df["target"].astype(float).values
X = np.where(np.isfinite(X), X, np.nan)

idx_all = np.arange(len(df))

idx_trainval, idx_test = train_test_split(
    idx_all, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

idx_train, idx_val = train_test_split(
    idx_trainval, test_size=VAL_SIZE, random_state=RANDOM_STATE
)

split = np.array(["trainval"] * len(df), dtype=object)
split[idx_train] = "train"
split[idx_val] = "val"
split[idx_test] = "test"

pd.DataFrame({"row_index": idx_all, "split": split}).to_csv(
    os.path.join(OUT_DIR, "split_indices.csv"), index=False
)

X_train_raw = X[idx_train]
X_val_raw = X[idx_val]
X_test_raw = X[idx_test]

y_train = y[idx_train]
y_val = y[idx_val]
y_test = y[idx_test]

imputer = SimpleImputer(strategy="median")

X_train_imp = imputer.fit_transform(X_train_raw)
X_val_imp = imputer.transform(X_val_raw)
X_test_imp = imputer.transform(X_test_raw)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train_imp)
X_val = scaler.transform(X_val_imp)
X_test = scaler.transform(X_test_imp)

joblib.dump(imputer, os.path.join(OUT_DIR, "preprocess_imputer.pkl"))
joblib.dump(scaler, os.path.join(OUT_DIR, "preprocess_scaler.pkl"))

selector = SelectKBest(score_func=f_regression, k=120)
X_train = selector.fit_transform(X_train, y_train)
X_val = selector.transform(X_val)    
X_test = selector.transform(X_test)  

joblib.dump(selector, os.path.join(OUT_DIR,"feature_selector.pkl"))

selected_mask = selector.get_support()
selected_feature_names = [feature_cols[i] for i, selected in enumerate(selected_mask) if selected]
feature_cols = selected_feature_names

print(f"Feature count reduced to: {X_train.shape[1]}")
        

mlr = RidgeCV(alphas=np.logspace(-2, 4, 50), scoring='neg_mean_squared_error')
mlr.fit(X_train, y_train)

joblib.dump(mlr, os.path.join(OUT_DIR, "mlr_model.pkl"))

def metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae

pred_test = mlr.predict(X_test)
r2, rmse, mae = metrics(y_test, pred_test)

coef_df = pd.DataFrame({
    "Feature": feature_cols,
    "Coefficient": mlr.coef_
})

coef_df = coef_df.sort_values(
    by="Coefficient",
    key=np.abs,
    ascending=False
)

coef_df.to_excel(
    os.path.join(OUT_DIR, "MLR_coefficients.xlsx"),
    index=False
)

lines = []
lines.append("===== Internal Hold-out Test (1492 split; MLR only; no leakage) =====")
lines.append(f"n_total={len(df)} | train={len(idx_train)} | val={len(idx_val)} | test={len(idx_test)}")
lines.append(f"[MLR] R2={r2:.4f} | RMSE={rmse:.2f} | MAE={mae:.2f}")

txt = "\\n".join(lines) + "\\n"
print(txt)

with open(os.path.join(OUT_DIR, "internal_metrics.txt"), "w", encoding="utf-8") as f:
    f.write(txt)

print("Done.")
