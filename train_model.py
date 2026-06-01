"""
============================================================
  SYNENT TECHNOLOGIES - DATA SCIENCE INTERNSHIP
  Task 9: End-to-End Data Science Project
  Name   : Palakurthy Shiva Sai Goud
  Project: Loan Approval Prediction
============================================================

Workflow:
  1. Data collection
  2. Data cleaning
  3. EDA and visualization
  4. Model building
  5. Deployment artifact for Streamlit
"""

import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

DATA_PATH = "data/loan_approval_dataset.csv"
CHART_DIR = "data/charts"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "loan_approval_model.pkl")
RESULTS_PATH = os.path.join(MODEL_DIR, "model_results.csv")
CLEAN_DATA_PATH = "data/loan_approval_cleaned.csv"
TARGET_COLUMN = "loan_status"

os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10


def save_chart(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def clean_dataset(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.drop_duplicates()

    if "loan_id" in df.columns:
        df = df.drop(columns=["loan_id"])

    text_cols = df.select_dtypes(include=["object"]).columns
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()

    for col in df.columns:
        if col != TARGET_COLUMN and col not in text_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Approved": 1, "Rejected": 0})
    df = df.dropna(subset=[TARGET_COLUMN])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    if {"income_annum", "loan_amount"}.issubset(df.columns):
        df["loan_to_income_ratio"] = df["loan_amount"] / df["income_annum"].replace(0, np.nan)

    asset_cols = [
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value",
    ]
    if all(col in df.columns for col in asset_cols):
        df["total_assets_value"] = df[asset_cols].sum(axis=1)
        df["loan_to_asset_ratio"] = df["loan_amount"] / df["total_assets_value"].replace(0, np.nan)

    return df


def make_eda_charts(df):
    print("\n>> Creating EDA charts...")

    status_counts = df[TARGET_COLUMN].map({1: "Approved", 0: "Rejected"}).value_counts()
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x=status_counts.index, y=status_counts.values, palette=["#4C78A8", "#F58518"])
    for container in ax.containers:
        ax.bar_label(container, fmt="%d")
    plt.title("Loan Approval Status Distribution")
    plt.xlabel("Loan Status")
    plt.ylabel("Applications")
    save_chart(os.path.join(CHART_DIR, "01_loan_status_distribution.png"))

    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="cibil_score", hue=TARGET_COLUMN, bins=30, kde=True, palette=["#F58518", "#4C78A8"])
    plt.title("CIBIL Score Distribution by Loan Status")
    plt.xlabel("CIBIL Score")
    plt.legend(title="Loan Status", labels=["Approved", "Rejected"])
    save_chart(os.path.join(CHART_DIR, "02_cibil_score_distribution.png"))

    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x=TARGET_COLUMN, y="loan_amount", palette=["#F58518", "#4C78A8"])
    plt.title("Loan Amount by Approval Status")
    plt.xlabel("Loan Status")
    plt.ylabel("Loan Amount")
    plt.xticks([0, 1], ["Rejected", "Approved"])
    save_chart(os.path.join(CHART_DIR, "03_loan_amount_by_status.png"))

    approval_by_education = df.groupby("education")[TARGET_COLUMN].mean().sort_values(ascending=False) * 100
    plt.figure(figsize=(7, 4))
    ax = sns.barplot(x=approval_by_education.index, y=approval_by_education.values, palette="muted")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%")
    plt.title("Approval Rate by Education")
    plt.xlabel("Education")
    plt.ylabel("Approval Rate (%)")
    save_chart(os.path.join(CHART_DIR, "04_approval_rate_by_education.png"))

    corr_cols = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns
    plt.figure(figsize=(10, 7))
    sns.heatmap(df[corr_cols].corr(), cmap="coolwarm", center=0, annot=False, linewidths=0.4)
    plt.title("Numeric Feature Correlation Heatmap")
    save_chart(os.path.join(CHART_DIR, "05_correlation_heatmap.png"))


def get_feature_names(preprocessor, numeric_features, categorical_features):
    names = list(numeric_features)
    if categorical_features:
        encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        names.extend(encoder.get_feature_names_out(categorical_features).tolist())
    return names


def main():
    print("=" * 60)
    print("   TASK 9 - END-TO-END DATA SCIENCE PROJECT")
    print("   Loan Approval Prediction")
    print("=" * 60)

    print(f"\n>> Loading dataset: {DATA_PATH}")
    df_raw = pd.read_csv(DATA_PATH)
    print(f"   Raw dataset: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")

    print("\n>> Cleaning dataset...")
    df = clean_dataset(df_raw)
    df.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"   Clean dataset saved: {CLEAN_DATA_PATH}")
    print(f"   Clean shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"   Missing values after cleaning: {int(df.isna().sum().sum())}")

    make_eda_charts(df)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=42, max_depth=10),
    }

    print("\n>> Training models...")
    results = []
    trained = {}
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
            "F1 Score": f1_score(y_test, preds),
        }
        results.append(metrics)
        trained[name] = pipeline
        print(f"   {name}: accuracy={metrics['Accuracy']:.3f}, f1={metrics['F1 Score']:.3f}")

    results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
    results_df.to_csv(RESULTS_PATH, index=False)
    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained[best_model_name]
    best_preds = best_model.predict(X_test)

    print(f"\n   Best model: {best_model_name}")
    print("\nClassification Report:")
    print(classification_report(y_test, best_preds, target_names=["Rejected", "Approved"]))

    plt.figure(figsize=(7, 4))
    ax = sns.barplot(data=results_df, x="Model", y="Accuracy", palette="muted")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f")
    plt.ylim(0, 1)
    plt.title("Model Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    save_chart(os.path.join(CHART_DIR, "06_model_accuracy_comparison.png"))

    cm = confusion_matrix(y_test, best_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Rejected", "Approved"], yticklabels=["Rejected", "Approved"])
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    save_chart(os.path.join(CHART_DIR, "07_confusion_matrix.png"))

    if hasattr(best_model.named_steps["model"], "feature_importances_"):
        feature_names = get_feature_names(best_model.named_steps["preprocessor"], numeric_features, categorical_features)
        importances = pd.Series(best_model.named_steps["model"].feature_importances_, index=feature_names)
        top_importances = importances.sort_values(ascending=False).head(10)
        plt.figure(figsize=(9, 5))
        sns.barplot(x=top_importances.values, y=top_importances.index, palette="muted")
        plt.title("Top 10 Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        save_chart(os.path.join(CHART_DIR, "08_feature_importance.png"))

    model_package = {
        "model": best_model,
        "model_name": best_model_name,
        "feature_columns": X.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "results": results_df,
    }
    joblib.dump(model_package, MODEL_PATH)
    print(f"\n   Model saved: {MODEL_PATH}")
    print(f"   Results saved: {RESULTS_PATH}")
    print("   Charts saved in: data/charts/")
    print("\nTask 9 training complete.")


if __name__ == "__main__":
    main()
