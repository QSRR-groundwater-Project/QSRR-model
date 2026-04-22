# QSRR-Based Groundwater Contaminant Screening Pipeline
This repository implements a reproducible QSRR (Quantitative Structure–Retention Relationship) pipeline for predicting retention indices (RI) and prioritizing groundwater contaminants.

The workflow integrates:

SMILES standardization and RDKit descriptor generation
Artificial Neural Network (ANN) modeling (leakage-free)
External validation with applicability domain (AD) analysis
Screening and prioritization using a mobility–occurrence framework

This pipeline is designed to support data-driven environmental prioritization of micropollutants, particularly in groundwater systems.





The pipeline consists of four main steps:

Step A → Step B → Step C → Step D
Data Build → Model Training → External Validation → Screening



Step A — RDKit Descriptor Dataset Construction

Script:
A_build_rdkit_datasets_1492_390_top100_with_smiles_cleanup_KEEPFREQ.py

Outputs:
train1492_rdkit_raw.xlsx
external390_rdkit_raw.xlsx
top100_rdkit_raw.xlsx
SMILES cleaning logs


Step B — ANN Model Training (No Leakage)

Script:
B_train_ANN_1492_no_leakage.py

Outputs:
ann_model.keras
preprocess_imputer.pkl
preprocess_scaler.pkl
feature_columns.txt
split_indices.csv
internal_metrics.txt


Step C — External Validation + Applicability Domain

Script:
C_external_validation_AD_ANN_1492_to_390.py

Outputs:
metrics.txt
AD_summary.txt
predictions.csv


Step D — Screening & Prioritization (Top100)

Script:
D_top100_predict_AD_matrix_ANN_only_FIXED.py

top100_predictions_with_AD.xlsx
mobility_occurrence_matrix.xlsx
quadrant_summary.txt

How to Run ?


Step 1: Install dependencies
pip install numpy pandas scikit-learn tensorflow rdkit joblib


Step 2: Run pipeline
python scripts/A_build_rdkit_datasets_*.py
python scripts/B_train_ANN_1492_no_leakage.py
python scripts/C_external_validation_AD_ANN_1492_to_390.py
python scripts/D_top100_predict_AD_matrix_ANN_only_FIXED.py

