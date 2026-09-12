"""Test dashboard per_model_breakdown."""
import sys
import os
sys.path.insert(0, os.getcwd())

from src.ensemble import load_models
from src.live_dashboard import per_model_breakdown
import numpy as np

# Load models
models = load_models()
print(f"Models: {list(models.keys())}")

# Simulate feature columns (same as what build_features would produce)
# We'll use the transformer's feature names plus extras
import json
with open('models/transformer_config.json') as f:
    config = json.load(f)
transformer_features = config['feature_cols']

# Create a feature set with all transformer features + extras to make 81 total
all_features = transformer_features + [f'extra_{i}' for i in range(81 - len(transformer_features))]

# Create dummy data
X = np.random.randn(320, len(all_features)).astype(np.float32)

# Test per_model_breakdown
breakdown = per_model_breakdown(models, X[-1:], feature_names=all_features)
print(f"\nBreakdown ({len(breakdown)} models):")
for r in breakdown:
    print(f"  {r['model']}: {r['class']} {r['confidence']:.0%}")

# Verify we have 4 models
assert len(breakdown) == 4, f"Expected 4 models, got {len(breakdown)}"
print("\n✓ All 4 models are showing!")
