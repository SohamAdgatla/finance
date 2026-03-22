"""
Synthetic Test Data Generator

Generates realistic personal finance data for testing the application
without requiring a real database or user data.
"""

import random
from typing import List, Dict, Any
import pandas as pd
import numpy as np


def generate_synthetic_data(
    n_users: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic user financial profiles for testing.

    Creates a mix of healthy, at-risk, and critical profiles with
    realistic distributions for income, expenses, savings, and debt.

    Args:
        n_users: Number of synthetic user profiles to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: user_id, monthly_income, monthly_expenses,
        current_savings, monthly_debt_payments, total_debt, credit_card_spending,
        total_credit_limit, consecutive_deficit_months, spending_history
    """
    random.seed(seed)
    np.random.seed(seed)

    data = []
    for i in range(n_users):
        # Create variety: 40% healthy, 35% at-risk, 25% critical
        profile_type = random.choices(
            ["healthy", "at_risk", "critical"],
            weights=[0.4, 0.35, 0.25],
        )[0]

        if profile_type == "healthy":
            income = random.uniform(4500, 12000)
            expenses = income * random.uniform(0.5, 0.75)
            savings = expenses * random.uniform(3, 12)
            debt_pct = random.uniform(0.05, 0.15)
            debt_payments = income * debt_pct
            cc_util = random.uniform(0.05, 0.25)
            consecutive_deficit = 0
        elif profile_type == "at_risk":
            income = random.uniform(3500, 9000)
            expenses = income * random.uniform(0.85, 1.05)
            savings = expenses * random.uniform(0.5, 2.5)
            debt_pct = random.uniform(0.2, 0.4)
            debt_payments = income * debt_pct
            cc_util = random.uniform(0.25, 0.5)
            consecutive_deficit = random.randint(0, 2)
        else:  # critical
            income = random.uniform(2500, 6000)
            expenses = income * random.uniform(1.05, 1.4)
            savings = expenses * random.uniform(0, 1)
            debt_pct = random.uniform(0.35, 0.55)
            debt_payments = income * debt_pct
            cc_util = random.uniform(0.4, 0.9)
            consecutive_deficit = random.randint(2, 5)

        # Credit limit: typically 1-3x monthly income
        credit_limit = income * random.uniform(0.5, 2.5)
        cc_spending = credit_limit * cc_util
        total_debt = debt_payments * random.uniform(12, 60)  # Amortized over 1-5 years

        # Generate 6 months of spending history
        spending_history = [
            expenses * random.uniform(0.9, 1.15)
            for _ in range(6)
        ]
        if consecutive_deficit > 0:
            for j in range(min(consecutive_deficit, 6)):
                spending_history[5 - j] = income * random.uniform(1.05, 1.3)

        data.append({
            "user_id": f"user_{i+1:03d}",
            "monthly_income": round(income, 2),
            "monthly_expenses": round(expenses, 2),
            "current_savings": round(savings, 2),
            "monthly_debt_payments": round(debt_payments, 2),
            "total_debt": round(total_debt, 2),
            "credit_card_spending": round(cc_spending, 2),
            "total_credit_limit": round(credit_limit, 2),
            "consecutive_deficit_months": consecutive_deficit,
            "spending_history": spending_history,
            "profile_type": profile_type,
        })

    return pd.DataFrame(data)


def get_sample_user(df: pd.DataFrame, profile_type: str = None) -> Dict[str, Any]:
    """
    Extract a single sample user from synthetic data.

    Args:
        df: DataFrame from generate_synthetic_data
        profile_type: Optional filter ("healthy", "at_risk", "critical")

    Returns:
        Dictionary with user financial data
    """
    if profile_type:
        subset = df[df["profile_type"] == profile_type]
        if subset.empty:
            subset = df
    else:
        subset = df

    row = subset.sample(1).iloc[0]
    return {
        "user_id": row["user_id"],
        "monthly_income": row["monthly_income"],
        "monthly_expenses": row["monthly_expenses"],
        "current_savings": row["current_savings"],
        "monthly_debt_payments": row["monthly_debt_payments"],
        "total_debt": row["total_debt"],
        "credit_card_spending": row["credit_card_spending"],
        "total_credit_limit": row["total_credit_limit"],
        "consecutive_deficit_months": row["consecutive_deficit_months"],
        "spending_history": row["spending_history"],
        "profile_type": row["profile_type"],
    }


# -----------------------------------------------------------------------------
# Quick-run data for Streamlit demo (no DB required)
# -----------------------------------------------------------------------------
DEMO_USERS = [
    {
        "monthly_income": 6500,
        "monthly_expenses": 5200,
        "current_savings": 8000,
        "monthly_debt_payments": 1800,
        "total_debt": 45000,
        "credit_card_spending": 3500,
        "total_credit_limit": 10000,
    },
    {
        "monthly_income": 4200,
        "monthly_expenses": 4100,
        "current_savings": 15000,
        "monthly_debt_payments": 500,
        "total_debt": 12000,
        "credit_card_spending": 800,
        "total_credit_limit": 5000,
    },
    {
        "monthly_income": 8500,
        "monthly_expenses": 7200,
        "current_savings": 25000,
        "monthly_debt_payments": 1200,
        "total_debt": 28000,
        "credit_card_spending": 2000,
        "total_credit_limit": 15000,
    },
]
