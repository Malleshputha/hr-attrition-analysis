"""
HR Attrition Analysis — IBM HR Analytics Dataset
EDA + Logistic Regression to predict employee attrition
Author: Mallesh Putha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────
DATA_PATH   = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
OUTPUT_PATH = "data/processed/"
RANDOM_STATE = 42

plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


# ── Step 1: Load & explore ────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")
    print(f"\nAttrition rate: {df['Attrition'].value_counts(normalize=True)['Yes']*100:.1f}%")
    print(f"\nNull values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    return df


# ── Step 2: Clean ─────────────────────────────────────────────
def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Drop single-value columns (no predictive value)
    drop_cols = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Binary encode target
    df["Attrition_binary"] = (df["Attrition"] == "Yes").astype(int)
    return df


# ── Step 3: EDA ───────────────────────────────────────────────
def run_eda(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("HR Attrition EDA — Key Patterns", fontsize=16, fontweight="bold")

    # 1. Attrition by Department
    dept_attr = df.groupby("Department")["Attrition_binary"].mean() * 100
    axes[0,0].bar(dept_attr.index, dept_attr.values, color=["#1D9E75","#7F77DD","#EF9F27"])
    axes[0,0].set_title("Attrition Rate by Department (%)")
    axes[0,0].set_ylabel("Attrition %")
    for i, v in enumerate(dept_attr.values):
        axes[0,0].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)

    # 2. Attrition by Age group
    df["age_group"] = pd.cut(df["Age"], bins=[18,25,35,45,55,65],
                              labels=["18-25","26-35","36-45","46-55","56+"])
    age_attr = df.groupby("age_group")["Attrition_binary"].mean() * 100
    axes[0,1].plot(age_attr.index.astype(str), age_attr.values, marker="o", color="#1D9E75", linewidth=2)
    axes[0,1].set_title("Attrition Rate by Age Group (%)")
    axes[0,1].set_ylabel("Attrition %")
    axes[0,1].fill_between(range(len(age_attr)), age_attr.values, alpha=0.15, color="#1D9E75")

    # 3. Monthly income vs Attrition
    axes[0,2].boxplot(
        [df[df["Attrition"]=="No"]["MonthlyIncome"],
         df[df["Attrition"]=="Yes"]["MonthlyIncome"]],
        labels=["Stayed","Left"], patch_artist=True
    )
    axes[0,2].set_title("Monthly Income vs Attrition")
    axes[0,2].set_ylabel("Monthly Income ($)")

    # 4. Years at company
    axes[1,0].hist(df[df["Attrition"]=="No"]["YearsAtCompany"],
                   bins=20, alpha=0.6, label="Stayed", color="#1D9E75")
    axes[1,0].hist(df[df["Attrition"]=="Yes"]["YearsAtCompany"],
                   bins=20, alpha=0.6, label="Left", color="#D85A30")
    axes[1,0].set_title("Years at Company Distribution")
    axes[1,0].legend()

    # 5. Job satisfaction
    sat_attr = df.groupby("JobSatisfaction")["Attrition_binary"].mean() * 100
    axes[1,1].bar(sat_attr.index, sat_attr.values, color="#7F77DD")
    axes[1,1].set_title("Attrition by Job Satisfaction (1=Low, 4=High)")
    axes[1,1].set_xlabel("Job Satisfaction Score")
    axes[1,1].set_ylabel("Attrition %")

    # 6. Overtime impact
    ot_attr = df.groupby("OverTime")["Attrition_binary"].mean() * 100
    axes[1,2].bar(ot_attr.index, ot_attr.values, color=["#1D9E75","#D85A30"])
    axes[1,2].set_title("Attrition: Overtime vs No Overtime (%)")
    axes[1,2].set_ylabel("Attrition %")
    for i, v in enumerate(ot_attr.values):
        axes[1,2].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}eda_plots.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("EDA plots saved!")


# ── Step 4: Feature engineering ───────────────────────────────
def prepare_features(df: pd.DataFrame):
    df = df.copy()

    # Encode categoricals
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ["Attrition"]]

    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    feature_cols = [c for c in df.columns
                    if c not in ["Attrition", "Attrition_binary", "age_group"]]

    X = df[feature_cols]
    y = df["Attrition_binary"]
    return X, y


# ── Step 5: Train model ───────────────────────────────────────
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n── Model Results ──────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=["Stayed","Left"]))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.3f}")

    # Feature importance
    importance = pd.DataFrame({
        "feature":   X.columns,
        "coefficient": abs(model.coef_[0])
    }).sort_values("coefficient", ascending=False).head(10)

    print("\n── Top 10 Attrition Predictors ────────────────")
    print(importance.to_string(index=False))

    # Save importance
    importance.to_csv(f"{OUTPUT_PATH}feature_importance.csv", index=False)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="#1D9E75", linewidth=2,
             label=f"ROC AUC = {roc_auc_score(y_test, y_prob):.3f}")
    plt.plot([0,1], [0,1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Attrition Prediction Model")
    plt.legend()
    plt.savefig(f"{OUTPUT_PATH}roc_curve.png", dpi=150, bbox_inches="tight")
    plt.show()

    return model, scaler


# ── Step 6: Business insights ──────────────────────────────────
def generate_insights(df: pd.DataFrame):
    print("\n── Business Insights ──────────────────────────")

    # High risk segments
    high_risk = df[df["Attrition_binary"] == 1].groupby("Department").agg(
        attrition_count=("Attrition_binary","sum"),
        avg_income=("MonthlyIncome","mean"),
        avg_satisfaction=("JobSatisfaction","mean"),
        avg_tenure=("YearsAtCompany","mean")
    ).round(2)
    print("\nHigh-risk departments:")
    print(high_risk)
    high_risk.to_csv(f"{OUTPUT_PATH}high_risk_segments.csv")

    # Overtime impact
    ot = df.groupby("OverTime")["Attrition_binary"].mean() * 100
    print(f"\nOvertime attrition rate: {ot.get('Yes', 0):.1f}%")
    print(f"No overtime attrition:   {ot.get('No', 0):.1f}%")
    print(f"Overtime multiplier:     {ot.get('Yes',0)/ot.get('No',1):.1f}x higher risk")


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    df    = load_data(DATA_PATH)
    df    = clean(df)
    run_eda(df)
    X, y  = prepare_features(df)
    model, scaler = train_model(X, y)
    generate_insights(df)
    print("\n✅ Analysis complete! Check data/processed/ for outputs.")
