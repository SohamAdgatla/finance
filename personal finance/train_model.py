"""
Train or Retrain the ML Risk Classifier

Run this script to (re)generate the Random Forest model used for
risk classification. The model is saved to src/models/risk_classifier.pkl
and loaded automatically by MLEngine.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.ml_engine import MLEngine


def main() -> None:
    print("Training ML Risk Classifier...")
    engine = MLEngine()
    engine._train_model()
    print("Model trained and saved to src/models/risk_classifier.pkl")


if __name__ == "__main__":
    main()
