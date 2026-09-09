QSRR-Based Groundwater Micropollutant Screening Pipeline

Our PhD project : Computational Models for Predicting Micropollutant Transfer and Ecotoxicological Risk in Water

A reproducible Quantitative Structure–Retention Relationship (QSRR) workflow for predicting chromatographic retention indices (RI) and prioritizing groundwater contaminants using mobility, occurrence, and toxicity information.

This repository accompanies our research on computational screening of groundwater micropollutants and provides datasets, source code, methodologies, and supplementary materials required to reproduce the complete workflow.





Repository Structure: 
QSRR-model

Data

Training dataset (1492 compounds);
Validation dataset (390 compounds);
Screening application dataset (Top100); 
Raw groundwater micropollutant monitoring dataset. 

Data reference : 

Training dataset (1492 compounds) + Validation dataset (390 compounds) : Hall, L.M., et al., Optimizing artificial neural network models for metabolomics and systems biology: an example using HPLC retention index data. Bioanalysis, 2015. 7(8): p. 939-955. https://www.tandfonline.com/doi/full/10.4155/bio.15.1  src="https://github.com/user-attachments/assets/d8a62714-91d8-4496-a2eb-534c7d5e52f6" />


Screening application dataset (Top100) : System, N.D.; Available from: https://www.norman-network.com/nds/empodat/.




Scripts

Script_A_build_rdkit_datasets.py;
Script_B_train_ANN.py;
Script_Calibration.py;
Script_C_external_validation_AD_ANN.py;
Script_D_top100_predict_AD_matrix.py. 

 Methodologies
 
 Dataset details and reference.pdf; 
 Methodologies for modeling (code).pdf. 
 
 Multicriteria risk systems (This section is part of the ACS ES&T water manuscript). 
 
 Risk scoring framework;
 Mobility–occurrence matrix;
 Toxicity table;
 Supporting tables.










Pipeline Description


Step A — RDKit Descriptor Dataset Construction

Script

Scripts/Script_A_build_rdkit_datasets.py


Input
Training dataset (1492 compounds)
Validation dataset (390 compounds)
Top100 screening dataset
SMILES structures
Experimental retention indices (RI)


Output
RDKit descriptor datasets
Canonicalized SMILES
Descriptor matrices
Data cleaning logs


Step B — ANN Model Development

Script

Scripts/Script_B_train_ANN.py
Purpose

Develop a leakage-free Artificial Neural Network (ANN) model for RI prediction.

Output
Trained ANN model
Feature scaler
Missing-value imputer
Selected feature list
Internal validation metrics


The calibration workflow is:

ANN training → save ANN → calibration code loads ANN → generate predictions → fit calibration using validation set → apply calibration to test set.


Step C — External Validation and Applicability Domain Analysis

Script

Scripts/Script_C_external_validation_AD_ANN.py
Purpose

Evaluate model performance using an independent external dataset and determine prediction reliability through Applicability Domain (AD) analysis.

Output
External prediction results
Performance metrics
AD summary
Prediction report



Step D — Groundwater Screening and Prioritization

Script

Scripts/Script_D_top100_predict_AD_matrix.py
Purpose

Predict RI values for groundwater contaminants and prioritize compounds using a mobility–occurrence framework combined with toxicity information.

Output
Predicted retention indices
Mobility–occurrence matrix
Risk prioritization tables
Summary reports



Data

The Data directory contains:

Training dataset (1492 compounds)
External validation dataset (390 compounds)
Top100 groundwater screening dataset
Raw groundwater monitoring occurrence dataset




Methodologies

The Methodologies directory provides detailed documentation describing

dataset preparation,
descriptor generation,
machine learning workflow,
model development,
validation procedures.




Multicriteria Risk Systems

This directory contains the supplementary materials supporting the proposed prioritization framework, including

mobility–occurrence matrix,
toxicity information,
multicriteria risk scoring framework,
supplementary tables used in the manuscript.



How to Run ?


Step 1: Install dependencies
pip install numpy pandas scikit-learn tensorflow rdkit joblib


Step 2: Run pipeline


python Scripts/Script_A_build_rdkit_datasets.py

python Scripts/Script_B_train_ANN.py

python Scripts/Script_Calibration.py

python Scripts/Script_C_external_validation_AD_ANN.py

python Scripts/Script_D_top100_predict_AD_matrix.py

