# src\model.py

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

def build_model(cat_cols: list[str], num_cols: list[str]) -> Pipeline:
    preprocess = ColumnTransformer(transformers=[
        ("cat",
         OneHotEncoder(handle_unknown="ignore"),
         cat_cols),
         ("num", "passthrough", num_cols)
    ])

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model)
    ])
    return pipe

def train_logreg(long_df: pd.DataFrame):
    ml_df = long_df[long_df["sir"].isin(["S", "R"])].copy()
    ml_df["y"] = ml_df["susceptible"].astype(int)

    feature_cols = ["bacteria", "antibiotic"]
    optional_cols = ["age", "gender", "Hospital_before", "Diabetes", "Hypertension"]
    for c in optional_cols:
        if c in ml_df.columns:
            feature_cols.append(c)

    X = ml_df[feature_cols].copy()
    y = ml_df["y"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2,
        random_state=42,
        stratify=y
    )

    cat_cols = X_train.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    pipe = build_model(cat_cols, num_cols)
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    report = classification_report(y_test, pred, digits=3)

    metrics = {"roc_auc": auc, "report": report, "feature_cols": feature_cols}
    return pipe, metrics

