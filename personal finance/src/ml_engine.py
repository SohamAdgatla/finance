"""
ML-Based Risk Classification Engine

Uses Random Forest to classify user financial risk levels:
- Safe: Healthy spending patterns, positive cash flow
- At-Risk: Concerning patterns (e.g., spending > income occasionally)
- Critical: Severe issues (e.g., spending > income for 3+ consecutive months)

Features used: income, expenses, savings, debt, spending trends, consecutive deficit months.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

# Lazy import for sklearn to allow running without it during initial load
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# Risk level labels
RISK_LEVELS = ["Safe", "At-Risk", "Critical"]


@dataclass
class RiskPrediction:
    """Result of ML risk classification."""
    risk_level: str
    confidence: float
    probabilities: dict


class MLEngine:
    """
    Machine learning engine for financial risk classification.
    Uses Random Forest trained on synthetic spending patterns.
    """

    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize the ML engine.
        
        Args:
            model_path: Optional path to pre-trained model. If None, uses/trains default.
        """
        self.model = None
        self.scaler = None
        self.feature_names = [
            "monthly_income",
            "monthly_expenses",
            "current_savings",
            "monthly_debt_payments",
            "debt_to_income_ratio",
            "expense_to_income_ratio",
            "savings_to_expense_ratio",
            "consecutive_deficit_months",
            "avg_monthly_deficit",
        ]
        self.model_path = model_path or Path(__file__).parent / "models" / "risk_classifier.pkl"
        self._load_or_train_model()

    def _load_or_train_model(self) -> None:
        """Load pre-trained model or train a new one if not found."""
        if not SKLEARN_AVAILABLE:
            return

        import pickle
        import joblib

        self.model_path = Path(self.model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        if self.model_path.exists():
            try:
                loaded = joblib.load(self.model_path)
                self.model = loaded.get("model")
                self.scaler = loaded.get("scaler")
                if self.model and self.scaler:
                    return
            except Exception:
                pass

        # Train fresh model
        self._train_model()

    def _create_training_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Generate synthetic training data with realistic risk patterns.
        
        Risk assignment logic:
        - Critical: consecutive_deficit_months >= 3 OR expense_to_income > 1.2
        - At-Risk: consecutive_deficit_months 1-2 OR expense_to_income 0.95-1.05
        - Safe: positive cash flow, healthy ratios
        """
        np.random.seed(42)
        n_samples = 2000

        data = []
        for _ in range(n_samples):
            income = np.random.uniform(3000, 15000)
            expenses = income * np.random.uniform(0.5, 1.5)
            savings = np.random.uniform(0, income * 12)
            debt_payments = np.random.uniform(0, income * 0.5)
            consecutive_deficit = np.random.randint(0, 6)
            avg_deficit = max(0, (expenses - income) * np.random.uniform(0.5, 2)) if expenses > income else 0

            dti = (debt_payments / income * 100) if income > 0 else 0
            exp_to_inc = expenses / income if income > 0 else 0
            sav_to_exp = savings / (expenses * 3) if expenses > 0 else 0  # Months of expense coverage

            # Assign risk level based on rules
            if consecutive_deficit >= 3 or exp_to_inc > 1.2:
                risk = 2  # Critical
            elif consecutive_deficit >= 1 or (0.95 <= exp_to_inc <= 1.05) or sav_to_exp < 0.5:
                risk = 1  # At-Risk
            else:
                risk = 0  # Safe

            data.append({
                "monthly_income": income,
                "monthly_expenses": expenses,
                "current_savings": savings,
                "monthly_debt_payments": debt_payments,
                "debt_to_income_ratio": dti,
                "expense_to_income_ratio": exp_to_inc,
                "savings_to_expense_ratio": sav_to_exp,
                "consecutive_deficit_months": consecutive_deficit,
                "avg_monthly_deficit": avg_deficit,
                "risk_level": risk,
            })

        df = pd.DataFrame(data)
        X = df[self.feature_names]
        y = df["risk_level"]
        return X, y

    def _train_model(self) -> None:
        """Train the Random Forest classifier and save to disk."""
        if not SKLEARN_AVAILABLE:
            return

        import joblib

        X, y = self._create_training_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight="balanced"
        )
        self.model.fit(X_train_scaled, y_train)

        joblib.dump(
            {"model": self.model, "scaler": self.scaler},
            self.model_path
        )

    def _extract_features(
        self,
        monthly_income: float,
        monthly_expenses: float,
        current_savings: float,
        monthly_debt_payments: float,
        consecutive_deficit_months: int = 0,
        avg_monthly_deficit: float = 0,
    ) -> np.ndarray:
        """Extract feature vector for prediction."""
        dti = (monthly_debt_payments / monthly_income * 100) if monthly_income > 0 else 0
        exp_to_inc = monthly_expenses / monthly_income if monthly_income > 0 else 0
        sav_to_exp = current_savings / (monthly_expenses * 3) if monthly_expenses > 0 else 0

        return np.array([[
            monthly_income,
            monthly_expenses,
            current_savings,
            monthly_debt_payments,
            dti,
            exp_to_inc,
            sav_to_exp,
            consecutive_deficit_months,
            avg_monthly_deficit,
        ]])

    def predict(
        self,
        monthly_income: float,
        monthly_expenses: float,
        current_savings: float,
        monthly_debt_payments: float,
        consecutive_deficit_months: int = 0,
        monthly_spending_history: Optional[List[float]] = None,
    ) -> RiskPrediction:
        """
        Predict financial risk level.
        
        Args:
            monthly_income: Gross monthly income
            monthly_expenses: Current monthly expenses
            current_savings: Liquid savings
            monthly_debt_payments: Monthly debt obligations
            consecutive_deficit_months: Manual input for deficit months (optional)
            monthly_spending_history: List of past N months spending to compute deficit (optional)
            
        Returns:
            RiskPrediction with level, confidence, and class probabilities
        """
        # Compute consecutive deficit from history if provided
        if monthly_spending_history is not None and monthly_income > 0:
            consecutive_deficit_months = 0
            for spending in reversed(monthly_spending_history):
                if spending > monthly_income:
                    consecutive_deficit_months += 1
                else:
                    break

        avg_deficit = max(0, monthly_expenses - monthly_income) if monthly_expenses > monthly_income else 0

        if not SKLEARN_AVAILABLE or self.model is None:
            # Fallback rule-based classification
            exp_to_inc = monthly_expenses / monthly_income if monthly_income > 0 else 0
            if consecutive_deficit_months >= 3 or exp_to_inc > 1.2:
                risk_level = "Critical"
                confidence = 0.9
            elif consecutive_deficit_months >= 1 or 0.95 <= exp_to_inc <= 1.05:
                risk_level = "At-Risk"
                confidence = 0.85
            else:
                risk_level = "Safe"
                confidence = 0.9
            return RiskPrediction(
                risk_level=risk_level,
                confidence=confidence,
                probabilities={"Safe": 0.33, "At-Risk": 0.33, "Critical": 0.34}
            )

        X = self._extract_features(
            monthly_income,
            monthly_expenses,
            current_savings,
            monthly_debt_payments,
            consecutive_deficit_months,
            avg_deficit,
        )
        X_scaled = self.scaler.transform(X)

        pred = self.model.predict(X_scaled)[0]
        probs = self.model.predict_proba(X_scaled)[0]
        confidence = float(probs[pred])

        return RiskPrediction(
            risk_level=RISK_LEVELS[pred],
            confidence=confidence,
            probabilities={
                RISK_LEVELS[i]: float(probs[i])
                for i in range(len(RISK_LEVELS))
            }
        )
