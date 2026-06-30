import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix


FEATURES = [
    "age",
    "income",
    "loan_amount",
    "debt_to_income",
    "credit_history_years",
    "delinquencies",
    "utilization",
]


def train_credit_model(df, model_type="logistic"):
    X = df[FEATURES]
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    if model_type == "logistic":
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        )
    else:
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            random_state=42,
            class_weight="balanced",
        )

    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    metrics = {
        "ROC AUC": roc_auc_score(y_test, prob),
        "Accuracy": accuracy_score(y_test, pred),
        "Confusion Matrix": confusion_matrix(y_test, pred),
        "Classification Report": classification_report(y_test, pred),
    }

    return model, metrics, X_test, y_test, prob


def risk_category(pd_value):
    if pd_value < 0.03:
        return "Low Risk"
    elif pd_value < 0.10:
        return "Medium Risk"
    else:
        return "High Risk"


def expected_loss(pd_value, lgd=0.45, ead=10000):
    return pd_value * lgd * ead
