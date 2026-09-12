"""Comprehensive test of 4-model dashboard."""
import sys
import os
sys.path.insert(0, os.getcwd())

print("=" * 60)
print("Testing 4-Model Dashboard")
print("=" * 60)

# Test 1: Load models
print("\n1. Loading models...")
from src.ensemble import load_models
models = load_models()
print(f"   Models loaded: {list(models.keys())}")
assert len(models) == 4, f"Expected 4 models, got {len(models)}"
print("   ✓ 4 models loaded")

# Test 2: Test per_model_breakdown
print("\n2. Testing per_model_breakdown...")
from src.live_dashboard import per_model_breakdown
import numpy as np
import json

# Get transformer's feature names
with open('models/transformer_config.json') as f:
    config = json.load(f)
transformer_features = config['feature_cols']

# Create full feature set (81 features)
all_features = transformer_features + [f'extra_{i}' for i in range(81 - len(transformer_features))]

# Create dummy data
X = np.random.randn(320, len(all_features)).astype(np.float32)

# Test breakdown
breakdown = per_model_breakdown(models, X[-1:], feature_names=all_features)
print(f"   Models in breakdown: {[r['model'] for r in breakdown]}")
assert len(breakdown) == 4, f"Expected 4 models in breakdown, got {len(breakdown)}"
print("   ✓ 4 models in breakdown")

# Test 3: Test ensemble_predict with feature_names
print("\n3. Testing ensemble_predict with feature_names...")
from src.ensemble import ensemble_predict
pred, conf = ensemble_predict(models, X, feature_names=all_features)
print(f"   Prediction shape: {pred.shape}")
print(f"   Last prediction: class={pred[-1]}, conf={conf[-1]:.2f}")
assert pred.shape == (320,), f"Expected shape (320,), got {pred.shape}"
print("   ✓ ensemble_predict works")

# Test 4: Verify HTML generation would work
print("\n4. Testing HTML generation logic...")
# Simulate what run_once does
cols = all_features  # This would be the actual feature columns
X_row = X[-1:]  # Last row
breakdown = per_model_breakdown(models, X_row, feature_names=cols)
models_line = " · ".join(
    f"<b>{r['model']}</b> <span style='color:#8a8f98'>{r['class']}</span> {r['confidence']:.0%}"
    for r in breakdown
)
print(f"   Models line: {models_line}")
assert "transformer" in models_line.lower(), "Transformer not in models_line"
print("   ✓ HTML would include transformer")

print("\n" + "=" * 60)
print("ALL TESTS PASSED - 4 models will show in dashboard!")
print("=" * 60)
