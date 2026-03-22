"""
Rule-Based Financial Mistake Detection Engine

Implements three core financial health rules:
1. Emergency Fund: Alerts if savings < 3x monthly expenses
2. Debt-to-Income (DTI): Alerts if monthly debt payments > 36% of gross income
3. Credit Utilization: Alerts if spending on credit cards > 30% of total limit
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RedFlag:
    """Represents a detected financial red flag."""
    rule_id: str
    title: str
    description: str
    severity: str  # "high", "medium", "low"
    current_value: float
    threshold_value: float
    unit: str = ""


# -----------------------------------------------------------------------------
# Financial Constants (Industry Standards)
# -----------------------------------------------------------------------------
EMERGENCY_FUND_MULTIPLIER = 3  # Savings should be >= 3 months of expenses
DTI_THRESHOLD_PCT = 36  # Debt payments should be <= 36% of gross income
CREDIT_UTILIZATION_THRESHOLD_PCT = 30  # Credit usage should be <= 30% of limit


class RuleEngine:
    """
    Rule-based engine for detecting common personal finance mistakes.
    """

    @staticmethod
    def check_emergency_fund(
        monthly_expenses: float,
        current_savings: float
    ) -> Optional[RedFlag]:
        """
        Rule: Emergency fund should cover at least 3 months of expenses.
        
        Args:
            monthly_expenses: Total monthly expenditure
            current_savings: Total liquid savings
            
        Returns:
            RedFlag if insufficient, None if healthy
        """
        required_savings = monthly_expenses * EMERGENCY_FUND_MULTIPLIER
        
        if current_savings < required_savings and monthly_expenses > 0:
            shortfall = required_savings - current_savings
            return RedFlag(
                rule_id="EMERGENCY_FUND",
                title="Insufficient Emergency Fund",
                description=f"Your savings (${current_savings:,.2f}) are below the "
                           f"recommended 3-month cushion (${required_savings:,.2f}). "
                           f"Shortfall: ${shortfall:,.2f}",
                severity="high",
                current_value=current_savings,
                threshold_value=required_savings,
                unit="USD"
            )
        return None

    @staticmethod
    def check_debt_to_income(
        monthly_gross_income: float,
        monthly_debt_payments: float
    ) -> Optional[RedFlag]:
        """
        Rule: Debt-to-Income ratio should not exceed 36%.
        
        Args:
            monthly_gross_income: Total pre-tax monthly income
            monthly_debt_payments: Total monthly debt obligations
            
        Returns:
            RedFlag if DTI exceeds threshold, None if healthy
        """
        if monthly_gross_income <= 0:
            return None
            
        dti_ratio = (monthly_debt_payments / monthly_gross_income) * 100
        
        if dti_ratio > DTI_THRESHOLD_PCT:
            return RedFlag(
                rule_id="DEBT_TO_INCOME",
                title="High Debt-to-Income Ratio",
                description=f"Your DTI ratio ({dti_ratio:.1f}%) exceeds the "
                           f"recommended 36% threshold. Debt payments (${monthly_debt_payments:,.2f}) "
                           f"are too high relative to income (${monthly_gross_income:,.2f})",
                severity="high",
                current_value=dti_ratio,
                threshold_value=DTI_THRESHOLD_PCT,
                unit="%"
            )
        return None

    @staticmethod
    def check_credit_utilization(
        credit_card_spending: float,
        total_credit_limit: float
    ) -> Optional[RedFlag]:
        """
        Rule: Credit card spending should not exceed 30% of total limit.
        
        Args:
            credit_card_spending: Current credit card balance/usage
            total_credit_limit: Total combined credit limit across cards
            
        Returns:
            RedFlag if utilization exceeds threshold, None if healthy
        """
        if total_credit_limit <= 0:
            return None
            
        utilization_pct = (credit_card_spending / total_credit_limit) * 100
        
        if utilization_pct > CREDIT_UTILIZATION_THRESHOLD_PCT:
            return RedFlag(
                rule_id="CREDIT_UTILIZATION",
                title="High Credit Card Utilization",
                description=f"Your credit utilization ({utilization_pct:.1f}%) exceeds "
                           f"the recommended 30%. Spending ${credit_card_spending:,.2f} "
                           f"of ${total_credit_limit:,.2f} limit may hurt your credit score",
                severity="medium",
                current_value=utilization_pct,
                threshold_value=CREDIT_UTILIZATION_THRESHOLD_PCT,
                unit="%"
            )
        return None

    @classmethod
    def run_all_checks(
        cls,
        monthly_income: float,
        monthly_expenses: float,
        monthly_debt_payments: float,
        current_savings: float,
        credit_card_spending: float = 0,
        total_credit_limit: float = 0
    ) -> List[RedFlag]:
        """
        Execute all rule-based checks and return aggregated red flags.
        
        Args:
            monthly_income: Gross monthly income
            monthly_expenses: Total monthly expenses
            monthly_debt_payments: Monthly debt obligations
            current_savings: Liquid savings balance
            credit_card_spending: Current credit card balance (optional)
            total_credit_limit: Total credit limit (optional)
            
        Returns:
            List of RedFlag objects for any violated rules
        """
        red_flags: List[RedFlag] = []
        
        # Emergency Fund check
        ef_flag = cls.check_emergency_fund(monthly_expenses, current_savings)
        if ef_flag:
            red_flags.append(ef_flag)
        
        # DTI check
        dti_flag = cls.check_debt_to_income(monthly_income, monthly_debt_payments)
        if dti_flag:
            red_flags.append(dti_flag)
        
        # Credit Utilization (only if credit data provided)
        if total_credit_limit > 0:
            cu_flag = cls.check_credit_utilization(
                credit_card_spending, total_credit_limit
            )
            if cu_flag:
                red_flags.append(cu_flag)
        
        return red_flags
