"""Debug ensemble shapes."""
import sys
import os
sys.path.insert(0, os.getcwd())

from src.ensemble import load_models
from src.baselines import predict_proba
import numpy as np

models = load_models()
print(f"Models: {list(models.keys())}")

# Simulate what happens in run_once
# We need the actual feature columns that would be produced by build_features
# For now, let's load the transformer config to get its feature names
import json
with open('models/transformer_config.json') as f:
    config = json.load(f)
transformer_features = config['feature_cols']
print(f"Transformer needs {len(transformer_features)} features: {transformer_features[:5]}...")

# The baseline models expect 81 features
# Let's create dummy data with proper column names
# We'll include all transformer features plus extra ones
all_features = transformer_features + [f'extra_{i}' for i in range(81 - len(transformer_features))]
print(f"Total features: {len(all_features)}")

X = np.random.randn(320, len(all_features)).astype(np.float32)

# Test each model
for name, model in models.items():
    try:
        probs = predict_proba(model, X, feature_names=all_features)
        print(f"{name}: {probs.shape}")
    except Exception as e:
        print(f"{name}: ERROR - {e}")

# Now test ensemble_proba
from src.ensemble import ensemble_proba
try:
    probs = ensemble_proba(models, X, feature_names=all_features)
    print(f"\nEnsemble probs shape: {probs.shape}")
    print(f"Last row prediction: {probs[-1].argmax()} with conf {probs[-1].max():.2f}")
except Exception as e:
    print(f"\nEnsemble ERROR: {e}")
    import traceback
    traceback.print_exc()
