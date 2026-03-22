# Personal Finance Mistake Detector

A comprehensive Python-based system that combines **rule-based checks** and **machine learning** to detect financial health issues and provide actionable corrective suggestions.

## Features

- **Rule-Based Engine**: Emergency fund, debt-to-income (DTI), and credit utilization checks
- **ML Risk Classification**: Random Forest classifier for Safe / At-Risk / Critical levels
- **Corrective Suggestions**: 3-step actionable plans via dictionary mapping (LLM-ready)
- **Financial Health Score**: 0–100 composite score
- **Interactive Dashboard**: Streamlit UI with Plotly charts
- **Synthetic Test Data**: Run immediately without a real database

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) Pre-train the ML model
python train_model.py

# Launch the dashboard
streamlit run app.py
```

## Project Structure

```
personal finance/
├── app.py                 # Streamlit dashboard entry point
├── train_model.py         # Script to train ML model
├── requirements.txt
├── README.md
└── src/
    ├── __init__.py
    ├── rule_engine.py     # Rule-based detection (Emergency Fund, DTI, Credit Utilization)
    ├── ml_engine.py       # Random Forest risk classifier
    ├── suggestions.py     # Corrective action plans (dictionary mapping)
    ├── health_score.py    # 0–100 financial health score
    ├── synthetic_data.py  # Synthetic test data generator
    └── models/            # Saved ML model (auto-created on first run)
        └── risk_classifier.pkl
```

## Rule-Based Checks

| Rule | Threshold | Alert Condition |
|------|-----------|-----------------|
| Emergency Fund | 3× monthly expenses | Savings < 3× expenses |
| Debt-to-Income | 36% | Monthly debt > 36% of gross income |
| Credit Utilization | 30% | Credit card spending > 30% of limit |

## ML Risk Levels

- **Safe**: Healthy spending, positive cash flow
- **At-Risk**: Concerning patterns (e.g., occasional overspend)
- **Critical**: Severe issues (e.g., spending > income for 3+ months)

## Synthetic Test Data

Use the "Load synthetic test data" option in the sidebar to test with pre-generated profiles (healthy, at-risk, critical) without entering real data.

## Tech Stack

- Python, Pandas, NumPy
- Streamlit
- Scikit-learn (Random Forest)
- Plotly
