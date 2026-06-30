import numpy as np
import pandas as pd


def make_credit_data(n=5000, seed=42):
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 70, n)
    income = rng.normal(65000, 25000, n).clip(15000, 250000)
    loan_amount = rng.normal(18000, 9000, n).clip(1000, 80000)
    debt_to_income = (loan_amount / income + rng.normal(0.15, 0.08, n)).clip(0.01, 1.5)
    credit_history_years = rng.integers(0, 30, n)
    delinquencies = rng.poisson(0.4, n)
    utilization = rng.beta(2, 5, n)

    logit = (
        -3.0
        + 2.2 * debt_to_income
        + 1.8 * utilization
        + 0.35 * delinquencies
        - 0.035 * credit_history_years
        - 0.000008 * income
        + 0.01 * (35 - age)
    )

    pd_score = 1 / (1 + np.exp(-logit))
    default = rng.binomial(1, pd_score)

    return pd.DataFrame(
        {
            "age": age,
            "income": income,
            "loan_amount": loan_amount,
            "debt_to_income": debt_to_income,
            "credit_history_years": credit_history_years,
            "delinquencies": delinquencies,
            "utilization": utilization,
            "default": default,
        }
    )
