import streamlit as st
import pandas as pd
import plotly.express as px

from data import make_credit_data
from model import train_credit_model, risk_category, expected_loss, FEATURES


st.set_page_config(page_title="Credit Risk Model", layout="wide")
st.title("Project 5: Credit Risk Model")

df = make_credit_data()
model_type = st.sidebar.selectbox("Model", ["logistic", "random_forest"])
model, metrics, X_test, y_test, prob = train_credit_model(df, model_type)

cols = st.columns(3)
cols[0].metric("ROC AUC", f"{metrics['ROC AUC']:.3f}")
cols[1].metric("Accuracy", f"{metrics['Accuracy']:.3f}")
cols[2].metric("Default Rate", f"{df['default'].mean():.2%}")

st.subheader("PD Distribution")
pd_df = pd.DataFrame({"Probability of Default": prob})
st.plotly_chart(px.histogram(pd_df, x="Probability of Default", nbins=50), use_container_width=True)

st.subheader("Score a New Applicant")

c1, c2, c3 = st.columns(3)
age = c1.number_input("Age", value=35)
income = c1.number_input("Income", value=65000.0)
loan_amount = c2.number_input("Loan Amount", value=18000.0)
debt_to_income = c2.number_input("Debt to Income", value=0.35)
credit_history_years = c3.number_input("Credit History Years", value=7)
delinquencies = c3.number_input("Delinquencies", value=0)
utilization = st.slider("Credit Utilization", 0.0, 1.0, 0.35)

new_applicant = pd.DataFrame(
    [
        {
            "age": age,
            "income": income,
            "loan_amount": loan_amount,
            "debt_to_income": debt_to_income,
            "credit_history_years": credit_history_years,
            "delinquencies": delinquencies,
            "utilization": utilization,
        }
    ]
)

pd_value = model.predict_proba(new_applicant[FEATURES])[:, 1][0]
category = risk_category(pd_value)
el = expected_loss(pd_value, lgd=0.45, ead=loan_amount)

st.metric("Probability of Default", f"{pd_value:.2%}")
st.metric("Risk Category", category)
st.metric("Expected Loss", f"${el:,.2f}")

st.subheader("Training Data Sample")
st.dataframe(df.head(100))
