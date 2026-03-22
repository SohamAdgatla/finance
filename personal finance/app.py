"""
Personal Finance Mistake Detector - Streamlit Dashboard

Interactive UI for inputting financial data and viewing:
- Red flags detected
- Financial health score
- ML risk classification
- Corrective action plans
- Income vs Expenses vs Savings chart
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rule_engine import RuleEngine, RedFlag
from src.ml_engine import MLEngine
from src.suggestions import SuggestionEngine
from src.health_score import calculate_financial_health_score
from src.synthetic_data import generate_synthetic_data, get_sample_user, DEMO_USERS


# -----------------------------------------------------------------------------
# Page Config & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Personal Finance Mistake Detector",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better visuals
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .red-flag-box {
        background: #fff5f5;
        border-left: 4px solid #e53e3e;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .suggestion-box {
        background: #f0fff4;
        border-left: 4px solid #38a169;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .score-excellent { color: #38a169; font-weight: bold; }
    .score-good { color: #68d391; font-weight: bold; }
    .score-fair { color: #ecc94b; font-weight: bold; }
    .score-poor { color: #ed8936; font-weight: bold; }
    .score-critical { color: #e53e3e; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


def get_score_class(score: float) -> str:
    """Return CSS class for score color."""
    if score >= 80:
        return "score-excellent"
    if score >= 65:
        return "score-good"
    if score >= 50:
        return "score-fair"
    if score >= 35:
        return "score-poor"
    return "score-critical"


def render_income_expense_chart(
    monthly_income: float,
    monthly_expenses: float,
    current_savings: float,
) -> go.Figure:
    """Create interactive bar chart: Income vs Expenses vs Savings."""
    categories = ["Income", "Expenses", "Savings"]
    values = [monthly_income, monthly_expenses, current_savings]
    colors = ["#2d5a87", "#e53e3e", "#38a169"]

    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f"${v:,.0f}" for v in values],
            textposition="outside",
            textfont=dict(size=14),
        )
    ])
    fig.update_layout(
        title="Income vs Expenses vs Savings",
        xaxis_title="Category",
        yaxis_title="Amount ($)",
        template="plotly_white",
        height=400,
        margin=dict(t=60, b=60, l=60, r=40),
        showlegend=False,
        font=dict(size=12),
    )
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return fig


def main() -> None:
    st.title("💰 Personal Finance Mistake Detector")
    st.markdown("*Hybrid rule-based + ML system for financial health analysis*")
    st.divider()

    # -------------------------------------------------------------------------
    # Sidebar: Data Input
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("📥 Input Your Data")
        st.caption("Enter your financial details or use synthetic data for testing.")

        # Quick load synthetic data
        use_synthetic = st.checkbox("Load synthetic test data", value=False)

        if use_synthetic:
            synthetic_df = generate_synthetic_data(n_users=50)
            profile = st.selectbox(
                "Select profile type",
                ["healthy", "at_risk", "critical"],
                format_func=lambda x: x.replace("_", " ").title(),
            )
            sample = get_sample_user(synthetic_df, profile)
            monthly_income = sample["monthly_income"]
            monthly_expenses = sample["monthly_expenses"]
            current_savings = sample["current_savings"]
            monthly_debt_payments = sample["monthly_debt_payments"]
            credit_card_spending = sample["credit_card_spending"]
            total_credit_limit = sample["total_credit_limit"]
            consecutive_deficit = sample["consecutive_deficit_months"]
            spending_history = sample.get("spending_history", [])
            st.info(f"Loaded: **{sample['user_id']}** ({profile})")
        else:
            # Demo presets
            preset = st.selectbox(
                "Quick preset (optional)",
                ["None", "Struggling", "Balanced", "Thriving"],
            )
            if preset != "None":
                idx = {"Struggling": 0, "Balanced": 1, "Thriving": 2}[preset]
                demo = DEMO_USERS[idx]
                monthly_income = demo["monthly_income"]
                monthly_expenses = demo["monthly_expenses"]
                current_savings = demo["current_savings"]
                monthly_debt_payments = demo["monthly_debt_payments"]
                credit_card_spending = demo["credit_card_spending"]
                total_credit_limit = demo["total_credit_limit"]
            else:
                monthly_income = 0.0
                monthly_expenses = 0.0
                current_savings = 0.0
                monthly_debt_payments = 0.0
                credit_card_spending = 0.0
                total_credit_limit = 0.0

            col1, col2 = st.columns(2)
            with col1:
                monthly_income = st.number_input(
                    "Monthly Income ($)",
                    min_value=0.0,
                    value=float(monthly_income),
                    step=100.0,
                )
                monthly_expenses = st.number_input(
                    "Monthly Expenses ($)",
                    min_value=0.0,
                    value=float(monthly_expenses),
                    step=100.0,
                )
                current_savings = st.number_input(
                    "Current Savings ($)",
                    min_value=0.0,
                    value=float(current_savings),
                    step=100.0,
                )
            with col2:
                monthly_debt_payments = st.number_input(
                    "Monthly Debt Payments ($)",
                    min_value=0.0,
                    value=float(monthly_debt_payments),
                    step=50.0,
                )
                credit_card_spending = st.number_input(
                    "Credit Card Balance ($)",
                    min_value=0.0,
                    value=float(credit_card_spending),
                    step=50.0,
                )
                total_credit_limit = st.number_input(
                    "Total Credit Limit ($)",
                    min_value=0.0,
                    value=float(total_credit_limit) if total_credit_limit else 5000.0,
                    step=500.0,
                )

            if preset == "None":
                consecutive_deficit = 0
                spending_history = []
            else:
                consecutive_deficit = 0
                spending_history = []

        # Avoid zero credit limit for calculations
        if total_credit_limit == 0:
            total_credit_limit = 1.0

    # -------------------------------------------------------------------------
    # Run Analysis
    # -------------------------------------------------------------------------
    red_flags = RuleEngine.run_all_checks(
        monthly_income,
        monthly_expenses,
        monthly_debt_payments,
        current_savings,
        credit_card_spending,
        total_credit_limit,
    )

    ml_engine = MLEngine()
    risk_pred = ml_engine.predict(
        monthly_income,
        monthly_expenses,
        current_savings,
        monthly_debt_payments,
        consecutive_deficit_months=consecutive_deficit,
        monthly_spending_history=spending_history if use_synthetic and spending_history else None,
    )

    health_score = calculate_financial_health_score(
        monthly_income,
        monthly_expenses,
        monthly_debt_payments,
        current_savings,
        credit_card_spending,
        total_credit_limit,
        red_flags,
    )

    suggestions = SuggestionEngine.get_all_suggestions(red_flags, risk_pred.risk_level)

    # -------------------------------------------------------------------------
    # Main Layout
    # -------------------------------------------------------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Financial Health Score", f"{health_score}/100")
        score_class = get_score_class(health_score)
        st.markdown(f'<p class="{score_class}">{"Excellent" if health_score >= 80 else "Good" if health_score >= 65 else "Fair" if health_score >= 50 else "Poor" if health_score >= 35 else "Critical"}</p>', unsafe_allow_html=True)

    with col2:
        st.metric("ML Risk Level", risk_pred.risk_level)
        st.caption(f"Confidence: {risk_pred.confidence:.0%}")

    with col3:
        st.metric("Red Flags", len(red_flags))
        if red_flags:
            st.caption(", ".join(r.rule_id for r in red_flags))

    st.divider()

    # Red Flags
    st.subheader("🚩 Red Flags Detected")
    if red_flags:
        for flag in red_flags:
            st.markdown(
                f'<div class="red-flag-box">'
                f'<strong>{flag.title}</strong><br/>'
                f'{flag.description}'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.success("No red flags detected. Your rule-based metrics look healthy.")

    # Corrective Suggestions
    st.subheader("📋 Corrective Action Plans")
    for plan in suggestions:
        with st.expander(f"**{plan.mistake_title}** (Priority: {plan.priority})", expanded=(plan.priority in ("high", "critical"))):
            for step in plan.steps:
                st.markdown(f"- {step}")

    st.divider()

    # Interactive Chart
    st.subheader("📊 Income vs Expenses vs Savings")
    fig = render_income_expense_chart(
        monthly_income, monthly_expenses, current_savings
    )
    st.plotly_chart(fig, use_container_width=True)

    # ML probabilities (optional expander)
    with st.expander("ML Risk Probabilities"):
        probs = risk_pred.probabilities
        prob_df = pd.DataFrame({
            "Risk Level": list(probs.keys()),
            "Probability": [f"{v:.1%}" for v in probs.values()],
        })
        st.dataframe(prob_df, hide_index=True, use_container_width=True)

    st.divider()
    st.caption("Built with Python, Pandas, Streamlit, Scikit-learn, Plotly | No data is stored or transmitted.")


if __name__ == "__main__":
    main()
