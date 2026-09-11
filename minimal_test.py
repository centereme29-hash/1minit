#!/usr/bin/env python3
"""Minimal test."""
import sys
sys.path.insert(0, 'c:\\Users\\user\\Desktop\\1minit')
print("Starting test...", flush=True)
import torch
print(f"PyTorch: {torch.__version__}", flush=True)
print(f"CUDA: {torch.cuda.is_available()}", flush=True)
try:
    from src.dataset import build_matrix
    print("Import successful", flush=True)
    print("Calling build_matrix...", flush=True)
    m, feature_cols = build_matrix()
    print(f"Loaded {len(m)} samples", flush=True)
except Exception as e:
    print(f"Error: {e}", flush=True)
    import traceback
    traceback.print_exc()
print("Test complete", flush=True)