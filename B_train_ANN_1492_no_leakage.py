# B_train_ANN_1492_no_leakage.py
# =============================================================================
# ANN-only pipeline | Step B
# Train ANN on 1492 dataset ONLY (strict no leakage).
#
# INPUT:
#   outputs_A_rdkit_build/train1492_rdkit_raw.xlsx
#
# OUTPUT:
#   outputs_B_ANN_1492/
#     - preprocess_imputer.pkl
#     - preprocess_scaler.pkl
#     - feature_columns.txt
#     - split_indices.csv
#     - internal_metrics.txt
#     - ann_model.keras
#
# Run:
#   python scripts/B_train_ANN_1492_no_leakage.py
# =============================================================================

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras import layers, callbacks, models

RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.15

ANN_EPOCHS = 3000
ANN_BATCH = 32
ANN_PATIENCE = 150
ANN_LR = 3e-4

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPE_DIR = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(PIPE_DIR, "outputs_A_rdkit_build", "train1492_rdkit_raw.xlsx")
OUT_DIR = os.path.join(PIPE_DIR, "outputs_B_ANN_1492")
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

df = pd.read_excel(DATA_PATH)
assert "target" in df.columns

feature_cols = [c for c in df.columns if c not in ["target", "SMILES"]]
with open(os.path.join(OUT_DIR, "feature_columns.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(feature_cols))

X = df[feature_cols].apply(pd.to_numeric, errors="coerce").values
y = df["target"].astype(float).values
X = np.where(np.isfinite(X), X, np.nan)

idx_all = np.arange(len(df))
idx_trainval, idx_test = train_test_split(idx_all, test_size=TEST_SIZE, random_state=RANDOM_STATE)
idx_train, idx_val = train_test_split(idx_trainval, test_size=VAL_SIZE, random_state=RANDOM_STATE)

split = np.array(["trainval"] * len(df), dtype=object)
split[idx_train] = "train"
split[idx_val] = "val"
split[idx_test] = "test"
pd.DataFrame({"row_index": idx_all, "split": split}).to_csv(
    os.path.join(OUT_DIR, "split_indices.csv"), index=False
)

X_train_raw, y_train = X[idx_train], y[idx_train]
X_val_raw, y_val = X[idx_val], y[idx_val]
X_test_raw, y_test = X[idx_test], y[idx_test]

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

def metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae

def build_ann(d):
    m = models.Sequential([
        layers.Input(shape=(d,)),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.20),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.10),
        layers.Dense(1)
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=ANN_LR), loss="mse")
    return m

ann = build_ann(X_train.shape[1])

cb = [
    callbacks.EarlyStopping(monitor="val_loss", patience=ANN_PATIENCE, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=50, min_lr=1e-6, verbose=0)
]

ann.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=ANN_EPOCHS,
    batch_size=ANN_BATCH,
    verbose=0,
    callbacks=cb
)

ann.save(os.path.join(OUT_DIR, "ann_model.keras"))

pred_test = ann.predict(X_test, verbose=0).reshape(-1)
r2, rmse, mae = metrics(y_test, pred_test)

lines = []
lines.append("===== Internal Hold-out Test (1492 split; ANN only; no leakage) =====")
lines.append(f"n_total={len(df)} | train={len(idx_train)} | val={len(idx_val)} | test={len(idx_test)}")
lines.append(f"[ANN] R2={r2:.4f} | RMSE={rmse:.2f} | MAE={mae:.2f}")
txt = "\n".join(lines) + "\n"
print("\n" + txt)

with open(os.path.join(OUT_DIR, "internal_metrics.txt"), "w", encoding="utf-8") as f:
    f.write(txt)

print("✅ Step B DONE. Outputs in:", OUT_DIR)
