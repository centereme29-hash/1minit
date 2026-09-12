"""Test script to verify 4 models are loaded."""
import sys
sys.path.insert(0, '.')

from src.ensemble import load_models

models = load_models()
print(f"Models loaded: {list(models.keys())}")
print(f"Total models: {len(models)}")

# Test per_model_breakdown logic
import numpy as np
from src.baselines import predict_proba, CLASS_NAMES

# Create dummy input
X = np.random.randn(100, 53).astype(np.float32)

for name, model in models.items():
    try:
        probs = predict_proba(model, X[-1:])
        cls = int(probs.argmax())
        conf = float(probs.max())
        print(f"  {name}: {CLASS_NAMES[cls]} ({conf:.1%})")
    except Exception as e:
        print(f"  {name}: ERROR - {e}")
