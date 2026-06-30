from data import make_credit_data
from model import train_credit_model, risk_category, expected_loss


def main():
    df = make_credit_data()
    model, metrics, X_test, y_test, prob = train_credit_model(df, "logistic")

    print("\nCredit Risk Model")
    print("-----------------")
    print(f"ROC AUC: {metrics['ROC AUC']:.4f}")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print("\nConfusion Matrix")
    print(metrics["Confusion Matrix"])

    sample_pd = prob[0]
    print(f"\nSample Probability of Default: {sample_pd:.2%}")
    print(f"Risk Category: {risk_category(sample_pd)}")
    print(f"Expected Loss on $10,000 exposure: ${expected_loss(sample_pd):,.2f}")

    df.to_csv("project_5_credit_risk_model/synthetic_credit_data.csv", index=False)
    print("\nSaved: project_5_credit_risk_model/synthetic_credit_data.csv")


if __name__ == "__main__":
    main()
