import numpy as np
import pandas as pd

def compute_risk_score(
    df,
    w_M=0.4,
    w_O=0.3,
    w_T=0.3,
    drop_missing_toxicity=True,
    RI_min=None,
    RI_max=None
):
    """
    Multicriteria risk scoring with normalization and missing data handling.

    Required input columns:
    - Pred_RI_ANN  (predicted RI)
    - frequency
    - LC50         (experimental LC50, can contain NaN)
    - AD_in_domain (0 or 1)
    """

    df = df.copy()

    # Remove missing LC50 values
    if drop_missing_toxicity:
        df = df.dropna(subset=["LC50"])
    else:
        print("Warning: Missing LC50 values retained.")

    # RI normalization (Min–Max)
    if RI_min is None:
        RI_min = df["Pred_RI_ANN"].min()
    if RI_max is None:
        RI_max = df["Pred_RI_ANN"].max()

    df["RI_norm"] = (df["Pred_RI_ANN"] - RI_min) / (RI_max - RI_min)

    # LC50 normalization
    LC50_min = df["LC50"].min()
    LC50_max = df["LC50"].max()
    df["LC50_norm"] = (df["LC50"] - LC50_min) / (LC50_max - LC50_min)

    # Occurrence normalization
    f_max = df["frequency"].max()
    df["occ_norm"] = np.log1p(df["frequency"]) / np.log1p(f_max)

    # Risk score calculation
    df["risk_score"] = (
        w_M * (1 - df["RI_norm"]) * df["AD_in_domain"] +
        w_O * df["occ_norm"] +
        w_T * (1 - df["LC50_norm"])
    )

    # Classification
    def classify(score):
        if score > 0.75:
            return "High priority"
        elif score >= 0.5:
            return "Medium priority"
        else:
            return "Low priority"

    df["priority_class"] = df["risk_score"].apply(classify)

    return df


if __name__ == "__main__":
    df = pd.read_csv("input_data.csv")

    results = compute_risk_score(
        df,
        w_M=0.33,
        w_O=0.33,
        w_T=0.33,
        drop_missing_toxicity=True
    )

    results.to_csv("risk_scoring_results.csv", index=False)

    print("Risk scoring completed.")
