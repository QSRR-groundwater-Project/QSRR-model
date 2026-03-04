# A_build_rdkit_datasets_1492_390_top100_with_smiles_cleanup_KEEPFREQ.py
# =============================================================================
# FIXED Step A (ANN-only pipeline):
# Build consistent RDKit descriptor datasets with SMILES cleanup/canonicalization.
#
# ✅ NEW: keep Top100 metadata columns such as:
#    - Detection_frequency_records
#    - Rank
#    - Individual compound
#    - StdInChIKey
#    - CAS No.
#
# INPUT (place inside ./data/):
#   - ibio_a_12354757_sm0003.xls   (1492, SMILES+RI)
#   - ibio_a_12354757_sm0004.xls   (390, SMILES+RI)
#   - S6_predict.xlsx              (Top100; SMILES + Detection_frequency_records)
#
# OUTPUT:
#   ./outputs_A_rdkit_build/
#     - train1492_rdkit_raw.xlsx
#     - external390_rdkit_raw.xlsx
#     - top100_rdkit_raw.xlsx   ✅ includes Detection_frequency_records + meta cols
#
# Run:
#   python scripts/A_build_rdkit_datasets_1492_390_top100_with_smiles_cleanup_KEEPFREQ.py
# =============================================================================

import os
import re
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPE_DIR = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(PIPE_DIR, "data")
OUT_DIR = os.path.join(PIPE_DIR, "outputs_A_rdkit_build")
os.makedirs(OUT_DIR, exist_ok=True)

SRC_1492 = os.path.join(DATA_DIR, "ibio_a_12354757_sm0003.xls")
SRC_390  = os.path.join(DATA_DIR, "ibio_a_12354757_sm0004.xls")
SRC_TOP  = os.path.join(DATA_DIR, "S6_predict.xlsx")

OUT_1492 = os.path.join(OUT_DIR, "train1492_rdkit_raw.xlsx")
OUT_390  = os.path.join(OUT_DIR, "external390_rdkit_raw.xlsx")
OUT_TOP  = os.path.join(OUT_DIR, "top100_rdkit_raw.xlsx")

def norm_cols(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def find_col(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        for k, v in cols.items():
            if k == cand.lower():
                return v
    for cand in candidates:
        for c in df.columns:
            if cand.lower() in c.lower():
                return c
    return None

def clean_smiles(s):
    if pd.isna(s):
        return None
    s = str(s).strip()
    s = s.replace("\u200b", "").replace("\ufeff", "")
    s = re.sub(r"\s+", "", s)
    if s in ["", "NA", "N/A", "nan", "None", "null", "-"]:
        return None
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None

def compute_rdkit_desc(smiles_list):
    names = [n for n, _ in Descriptors._descList]
    funcs = [f for _, f in Descriptors._descList]
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        vals = []
        for fn in funcs:
            try:
                vals.append(float(fn(mol)) if mol is not None else np.nan)
            except Exception:
                vals.append(np.nan)
        rows.append(vals)
    return pd.DataFrame(rows, columns=names)

def build_1492_or_390(path):
    df = pd.read_excel(path)
    df = norm_cols(df)

    smi_col = find_col(df, ["SMILES", "smiles"])
    if smi_col is None:
        raise ValueError(f"Cannot find SMILES column in {path}. Columns={list(df.columns)}")

    ri_col = find_col(df, ["RI", "ri", "RetentionIndex", "Retention Index", "RI_amide"])
    if ri_col is None:
        raise ValueError(f"Cannot find RI column in {path}. Columns={list(df.columns)}")

    df["SMILES_clean"] = [clean_smiles(x) for x in df[smi_col].tolist()]
    invalid = df[df["SMILES_clean"].isna()].copy()
    valid = df[df["SMILES_clean"].notna()].copy()

    invalid_out = os.path.join(OUT_DIR, f"smiles_invalid_rows_{os.path.basename(path)}.csv")
    invalid.to_csv(invalid_out, index=False)

    X_desc = compute_rdkit_desc(valid["SMILES_clean"].tolist())
    out = pd.concat([valid[["SMILES_clean"]].rename(columns={"SMILES_clean":"SMILES"}), X_desc], axis=1)
    out["target"] = pd.to_numeric(valid[ri_col], errors="coerce")

    return out, (len(df), len(out), len(invalid), os.path.basename(invalid_out))

def build_top100(path):
    df = pd.read_excel(path)
    df = norm_cols(df)

    smi_col = find_col(df, ["SMILES", "smiles"])
    if smi_col is None:
        raise ValueError(f"Cannot find SMILES column in {path}. Columns={list(df.columns)}")

    # Keep these if exist
    keep_cols = [
        "Rank",
        "Individual compound",
        "Detection_frequency_records",
        "CAS No.",
        "StdInChIKey"
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    df["SMILES_clean"] = [clean_smiles(x) for x in df[smi_col].tolist()]
    invalid = df[df["SMILES_clean"].isna()].copy()
    valid = df[df["SMILES_clean"].notna()].copy()

    invalid_out = os.path.join(OUT_DIR, f"smiles_invalid_rows_{os.path.basename(path)}.csv")
    invalid.to_csv(invalid_out, index=False)

    X_desc = compute_rdkit_desc(valid["SMILES_clean"].tolist())

    meta = valid[keep_cols].copy() if keep_cols else pd.DataFrame(index=valid.index)
    meta["SMILES"] = valid["SMILES_clean"].astype(str).values

    out = pd.concat([meta.reset_index(drop=True), X_desc.reset_index(drop=True)], axis=1)
    return out, (len(df), len(out), len(invalid), os.path.basename(invalid_out), keep_cols)

print("✅ Step A FIX (KEEPFREQ): Building RDKit datasets with SMILES cleanup...")

d1492, s1492 = build_1492_or_390(SRC_1492)
d390,  s390  = build_1492_or_390(SRC_390)
dtop,  stop  = build_top100(SRC_TOP)

d1492.to_excel(OUT_1492, index=False)
d390.to_excel(OUT_390, index=False)
dtop.to_excel(OUT_TOP, index=False)

lines = []
lines.append("===== SMILES cleaning summary (KEEPFREQ) =====")
lines.append(f"{os.path.basename(SRC_1492)}: raw={s1492[0]}, valid={s1492[1]}, invalid={s1492[2]} | log={s1492[3]}")
lines.append(f"{os.path.basename(SRC_390)}: raw={s390[0]}, valid={s390[1]}, invalid={s390[2]} | log={s390[3]}")
lines.append(f"{os.path.basename(SRC_TOP)}: raw={stop[0]}, valid={stop[1]}, invalid={stop[2]} | log={stop[3]}")
lines.append(f"Top100 kept meta columns: {stop[4]}")
txt = "\n".join(lines) + "\n"

print("\n" + txt)
with open(os.path.join(OUT_DIR, "smiles_cleaning_summary_KEEPFREQ.txt"), "w", encoding="utf-8") as f:
    f.write(txt)

print("✅ Step A FIX DONE. Outputs in:", OUT_DIR)