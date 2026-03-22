"""
Corrective Suggestions Module

NLP-driven (dictionary-mapping) function that takes detected mistakes
and generates 3-step actionable plans. Can be extended with LLM integration.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .rule_engine import RedFlag


@dataclass
class ActionablePlan:
    """A 3-step corrective action plan."""
    mistake_id: str
    mistake_title: str
    steps: List[str]
    priority: str


# -----------------------------------------------------------------------------
# Dictionary Mapping: Mistake ID -> 3-Step Actionable Plan
# -----------------------------------------------------------------------------
SUGGESTION_MAP: Dict[str, Dict[str, Any]] = {
    "EMERGENCY_FUND": {
        "title": "Insufficient Emergency Fund",
        "priority": "high",
        "steps": [
            "Step 1: Set up automatic transfers. Allocate 10-20% of each paycheck to a high-yield savings account before spending. Even $50-100/month builds the fund over time.",
            "Step 2: Reduce non-essential spending. Cancel unused subscriptions (gym, streaming, software) and redirect those funds to savings. Use the 48-hour rule for discretionary purchases.",
            "Step 3: Create a separate emergency-only account. Keep it at a different bank to reduce temptation. Aim for 3 months of expenses first, then 6 months.",
        ],
    },
    "DEBT_TO_INCOME": {
        "title": "High Debt-to-Income Ratio",
        "priority": "high",
        "steps": [
            "Step 1: List all debts with interest rates. Prioritize either the avalanche method (highest interest first) or snowball method (smallest balance first) for psychological wins.",
            "Step 2: Contact creditors for lower rates or hardship plans. Many offer 0% balance transfers or reduced payment plans. Refinancing high-interest loans can cut DTI significantly.",
            "Step 3: Increase income temporarily. Side gigs, overtime, or selling unused items can accelerate debt payoff. Avoid new debt until ratio is below 36%.",
        ],
    },
    "CREDIT_UTILIZATION": {
        "title": "High Credit Card Utilization",
        "priority": "medium",
        "steps": [
            "Step 1: Pay down balances before the statement closes. Utilization is calculated per statement—paying mid-cycle can lower the reported balance and boost your score.",
            "Step 2: Request a credit limit increase. If you have good payment history, a higher limit (without new spending) lowers utilization. Use the 'soft inquiry' option when available.",
            "Step 3: Consider a balance transfer to a 0% APR card. Move high-utilization balances to spread payments without interest. Cancel old cards only after payoff to preserve credit history length.",
        ],
    },
    "ML_AT_RISK": {
        "title": "At-Risk Spending Patterns",
        "priority": "medium",
        "steps": [
            "Step 1: Track every expense for 30 days. Use a budgeting app or spreadsheet to identify leaks—dining out, subscriptions, and impulse purchases often add up without notice.",
            "Step 2: Create a 50/30/20 budget. Allocate 50% to needs, 30% to wants, 20% to savings and debt. Adjust ratios if needed, but ensure savings is non-negotiable.",
            "Step 3: Build a buffer. Aim for one month of expenses in a checking buffer before the next paycheck. This prevents the 'paycheck-to-paycheck' trap and reduces stress.",
        ],
    },
    "ML_CRITICAL": {
        "title": "Critical Financial Stress",
        "priority": "critical",
        "steps": [
            "Step 1: Immediate triage. List all income and essential expenses (rent, utilities, food, minimum debt payments). Pause all non-essential spending until cash flow is positive.",
            "Step 2: Seek professional help. A certified credit counselor (NFCC.org) can negotiate with creditors and create a debt management plan. Bankruptcy may be an option—consult an attorney.",
            "Step 3: Establish a survival budget. Focus on needs only. Consider temporary income boosts (gig work, selling assets) and negotiate payment plans with landlords or service providers.",
        ],
    },
}


class SuggestionEngine:
    """
    Generates 3-step actionable plans for detected financial mistakes.
    Uses dictionary mapping; can be extended with LLM API calls.
    """

    @staticmethod
    def get_suggestions_for_red_flag(red_flag: RedFlag) -> ActionablePlan:
        """
        Generate corrective suggestions for a single red flag.
        
        Args:
            red_flag: Detected RedFlag from rule engine
            
        Returns:
            ActionablePlan with 3 steps
        """
        config = SUGGESTION_MAP.get(
            red_flag.rule_id,
            {
                "title": red_flag.title,
                "priority": "medium",
                "steps": [
                    "Step 1: Review your spending patterns and identify the root cause.",
                    "Step 2: Create a detailed budget aligned with your income.",
                    "Step 3: Set up automated savings and track progress monthly.",
                ],
            },
        )
        return ActionablePlan(
            mistake_id=red_flag.rule_id,
            mistake_title=config["title"],
            steps=config["steps"],
            priority=config["priority"],
        )

    @staticmethod
    def get_suggestions_for_risk_level(risk_level: str) -> ActionablePlan:
        """
        Generate suggestions based on ML risk classification.
        
        Args:
            risk_level: "Safe", "At-Risk", or "Critical"
            
        Returns:
            ActionablePlan for At-Risk or Critical; None for Safe
        """
        if risk_level == "Safe":
            return ActionablePlan(
                mistake_id="ML_SAFE",
                mistake_title="Healthy Financial Patterns",
                steps=[
                    "Step 1: Maintain your current habits. Consistency is key to long-term wealth.",
                    "Step 2: Consider increasing savings rate. Aim for 15-20% of income if not already.",
                    "Step 3: Diversify investments. Explore index funds or retirement accounts for growth.",
                ],
                priority="low",
            )

        key = "ML_AT_RISK" if risk_level == "At-Risk" else "ML_CRITICAL"
        config = SUGGESTION_MAP.get(key, SUGGESTION_MAP["ML_AT_RISK"])
        return ActionablePlan(
            mistake_id=key,
            mistake_title=config["title"],
            steps=config["steps"],
            priority=config["priority"],
        )

    @classmethod
    def get_all_suggestions(
        cls,
        red_flags: List[RedFlag],
        ml_risk_level: Optional[str] = None,
    ) -> List[ActionablePlan]:
        """
        Aggregate suggestions for all detected issues.
        
        Args:
            red_flags: List of rule-based red flags
            ml_risk_level: ML-predicted risk level (optional)
            
        Returns:
            List of ActionablePlan, deduplicated and prioritized
        """
        plans: List[ActionablePlan] = []

        for flag in red_flags:
            plans.append(cls.get_suggestions_for_red_flag(flag))

        if ml_risk_level and ml_risk_level in ("At-Risk", "Critical"):
            ml_plan = cls.get_suggestions_for_risk_level(ml_risk_level)
            if ml_plan.mistake_id not in [p.mistake_id for p in plans]:
                plans.append(ml_plan)

        # Sort by priority: critical > high > medium > low
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        plans.sort(key=lambda p: priority_order.get(p.priority, 4))

        return plans
