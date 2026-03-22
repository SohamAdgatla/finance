"""
Financial Health Score Calculator

Computes a 0-100 score based on:
- Emergency fund adequacy
- Debt-to-income ratio
- Credit utilization
- Spending vs income trend
- Savings rate
"""

from typing import List

from .rule_engine import RedFlag, RuleEngine


def calculate_financial_health_score(
    monthly_income: float,
    monthly_expenses: float,
    monthly_debt_payments: float,
    current_savings: float,
    credit_card_spending: float = 0,
    total_credit_limit: float = 1,  # Avoid division by zero
    red_flags: List[RedFlag] = None,
) -> float:
    """
    Calculate a 0-100 financial health score.

    Score components (each 0-100, weighted):
    - Emergency Fund (25%): 3+ months = 100, 0 months = 0
    - DTI (25%): 0-36% = 100, >50% = 0
    - Credit Utilization (20%): 0-30% = 100, >80% = 0
    - Expense-to-Income (20%): <70% = 100, >110% = 0
    - Savings Rate (10%): Based on (income - expenses) / income

    Args:
        monthly_income: Gross monthly income
        monthly_expenses: Total monthly expenses
        monthly_debt_payments: Monthly debt obligations
        current_savings: Liquid savings
        credit_card_spending: Credit card balance (optional)
        total_credit_limit: Total credit limit (optional)
        red_flags: Pre-computed red flags (optional, computed if not provided)

    Returns:
        Score between 0 and 100
    """
    if red_flags is None:
        red_flags = RuleEngine.run_all_checks(
            monthly_income,
            monthly_expenses,
            monthly_debt_payments,
            current_savings,
            credit_card_spending,
            total_credit_limit,
        )

    # Component 1: Emergency Fund (25% weight)
    months_of_expenses = current_savings / monthly_expenses if monthly_expenses > 0 else 0
    if months_of_expenses >= 6:
        ef_score = 100
    elif months_of_expenses >= 3:
        ef_score = 80 + (months_of_expenses - 3) / 3 * 20
    elif months_of_expenses >= 1:
        ef_score = 40 + (months_of_expenses - 1) / 2 * 40
    else:
        ef_score = max(0, months_of_expenses * 40)
    ef_score = min(100, ef_score)

    # Component 2: DTI (25% weight)
    dti_pct = (monthly_debt_payments / monthly_income * 100) if monthly_income > 0 else 0
    if dti_pct <= 20:
        dti_score = 100
    elif dti_pct <= 36:
        dti_score = 100 - (dti_pct - 20) * 2.5
    elif dti_pct <= 50:
        dti_score = 60 - (dti_pct - 36) * 4.3
    else:
        dti_score = max(0, 20 - (dti_pct - 50))
    dti_score = min(100, max(0, dti_score))

    # Component 3: Credit Utilization (20% weight)
    if total_credit_limit <= 0:
        cu_score = 100
    else:
        util_pct = (credit_card_spending / total_credit_limit) * 100
        if util_pct <= 30:
            cu_score = 100
        elif util_pct <= 50:
            cu_score = 100 - (util_pct - 30)
        elif util_pct <= 80:
            cu_score = 80 - (util_pct - 50)
        else:
            cu_score = max(0, 50 - (util_pct - 80))
        cu_score = min(100, max(0, cu_score))

    # Component 4: Expense-to-Income (20% weight)
    exp_to_inc = monthly_expenses / monthly_income if monthly_income > 0 else 1
    if exp_to_inc <= 0.7:
        eti_score = 100
    elif exp_to_inc <= 0.9:
        eti_score = 100 - (exp_to_inc - 0.7) * 250
    elif exp_to_inc <= 1.0:
        eti_score = 50 - (exp_to_inc - 0.9) * 500
    else:
        eti_score = max(0, 0 - (exp_to_inc - 1) * 100)
    eti_score = min(100, max(0, eti_score))

    # Component 5: Savings Rate (10% weight)
    surplus = monthly_income - monthly_expenses if monthly_income > 0 else 0
    savings_rate = surplus / monthly_income if monthly_income > 0 else 0
    if savings_rate >= 0.2:
        sr_score = 100
    elif savings_rate >= 0.1:
        sr_score = 50 + savings_rate * 500
    elif savings_rate >= 0:
        sr_score = savings_rate * 500
    else:
        sr_score = 0
    sr_score = min(100, max(0, sr_score))

    # Weighted average
    weights = [0.25, 0.25, 0.20, 0.20, 0.10]
    components = [ef_score, dti_score, cu_score, eti_score, sr_score]
    raw_score = sum(w * c for w, c in zip(weights, components))

    # Penalty for each red flag
    penalty_per_flag = 5
    penalty = len(red_flags) * penalty_per_flag
    final_score = max(0, min(100, raw_score - penalty))

    return round(final_score, 1)
