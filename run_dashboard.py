"""Run live dashboard and capture output with debug."""
import sys
import os
sys.path.insert(0, os.getcwd())

from src.live_dashboard import main
import argparse
import traceback

# Simulate command line args
sys.argv = ['live_dashboard.py', '--no-open', '--output', 'data/live_prediction.html']

try:
    main()
    print("Dashboard ran successfully")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
    
    # Additional debug
    print("\n--- Debug Info ---")
    try:
        from src.ensemble import load_models
        from src.train import TransformerPredictor
        import numpy as np
        
        models = load_models()
        print(f"Models: {list(models.keys())}")
        
        # Test transformer
        transformer = TransformerPredictor.from_disk()
        print(f"Transformer context_len: {transformer.cfg.context_len}")
        print(f"Transformer n_features: {transformer.cfg.n_features}")
        print(f"Transformer n_classes: {transformer.cfg.n_classes}")
        
        # Test with batch
        X = np.random.randn(320, 53).astype(np.float32)
        wrapper = models['transformer']
        probs = wrapper.predict_proba(X)
        print(f"Transformer output shape: {probs.shape}")
        
        # Test with xgboost
        from src.baselines import predict_proba
        xgb_probs = predict_proba(models['xgboost'], X)
        print(f"XGBoost output shape: {xgb_probs.shape}")
        
    except Exception as e2:
        print(f"Debug error: {e2}")
        traceback.print_exc()

