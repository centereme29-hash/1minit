"""Test per_model_breakdown with all 4 models."""
import sys
import os
sys.path.insert(0, os.getcwd())

import numpy as np
from src.ensemble import load_models
from src.live_dashboard import per_model_breakdown

models = load_models()
print(f"Models: {list(models.keys())}")

# Create dummy features - simulating what build_features would produce
# We need all 81 features that the baseline models expect
# The transformer will select its 53 features from these

# First, get the transformer's required features
from src.train import TransformerPredictor
transformer = TransformerPredictor.from_disk()
transformer_features = transformer.feature_cols
print(f"Transformer needs {len(transformer_features)} features")

# Create a full feature set that includes the transformer's features
# plus additional features for the baseline models
# For testing, we'll create 81 features where the first 53 are the transformer's
all_features = transformer_features + [f"extra_feat_{i}" for i in range(81 - len(transformer_features))]
print(f"Total features: {len(all_features)}")

X = np.random.randn(100, len(all_features)).astype(np.float32)

breakdown = per_model_breakdown(models, X[-1:], feature_names=all_features)
print(f"\nBreakdown ({len(breakdown)} models):")
for r in breakdown:
    print(f"  {r}")


