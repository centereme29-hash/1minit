"""Verify 4 models are loaded."""
import sys
import os
sys.path.insert(0, os.getcwd())
os.chdir(os.getcwd())

from src.ensemble import load_models

try:
    models = load_models()
    with open('models_check.txt', 'w') as f:
        f.write(f"Models loaded: {list(models.keys())}\n")
        f.write(f"Total models: {len(models)}\n")
        for name, model in models.items():
            f.write(f"  {name}: {type(model).__name__}\n")
    print("Success - see models_check.txt")
except Exception as e:
    with open('models_check.txt', 'w') as f:
        f.write(f"ERROR: {e}\n")
    print(f"Error: {e}")
