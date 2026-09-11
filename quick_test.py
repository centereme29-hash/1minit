#!/usr/bin/env python3
"""Quick test of 1minit model - tests 1-3 only."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from src.params import FEATURES_DIR, NUM_CLASSES, TARGET_RETURNS
from src.dataset import build_matrix
from src.model import ModelConfig, MultiTimeframeTransformer, count_params, multihead_loss
from src.baselines import LABEL_MAP

print("="*70)
print("  1minit Quick Test Suite (Tests 1-3)")
print(f"  Python: {sys.version.split()[0]}")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA: {torch.cuda.is_available()}")
print("="*70)

# Test 1: Data Loading
print("\n=== TEST 1: Data Loading ===")
try:
    m, feature_cols = build_matrix()
    print(f"PASS: Loaded {len(m)} samples, {len(feature_cols)} features")
    print(f"  Shape: {m.shape}")
    print(f"  Time range: {m['time'].min()} -> {m['time'].max()}")
    
    label_dist = m["label"].value_counts().sort_index()
    print("\nLabel distribution:")
    for label, count in label_dist.items():
        name = {0: "DOWN", 1: "NO_MOVE", 2: "UP"}.get(label, str(label))
        print(f"  {name}: {count} ({100*count/len(m):.2f}%)")
    
    nan_count = m.isna().sum().sum()
    if nan_count > 0:
        print(f"FAIL: Found {nan_count} NaN values")
    else:
        print("PASS: No NaN values")
except Exception as e:
    print(f"FAIL: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Model Architecture
print("\n=== TEST 2: Model Architecture ===")
try:
    cfg = ModelConfig(
        n_features=83, d_model=256, layers=4, heads=4, dropout=0.1,
        seq_len=30, branches={"1m": 30, "5m": 36, "15m": 32},
        n_returns=len(TARGET_RETURNS), n_classes=NUM_CLASSES,
    )
    
    print(f"Config: d_model={cfg.d_model}, layers={cfg.layers}, heads={cfg.heads}")
    print(f"context_len={cfg.context_len}")
    
    model = MultiTimeframeTransformer(cfg)
    n_params = count_params(model)
    print(f"\nPASS: Model created: {n_params:,} parameters ({n_params/1e6:.2f}M)")
    
    batch_size = 4
    x = torch.randn(batch_size, cfg.context_len, cfg.n_features)
    print(f"\nInput shape: {x.shape}")
    
    with torch.no_grad():
        outputs = model(x)
    
    print("\nOutput shapes:")
    for name, tensor in outputs.items():
        print(f"  {name}: {tuple(tensor.shape)}")
    
    if torch.cuda.is_available():
        model_cuda = model.cuda()
        x_cuda = x.cuda()
        with torch.no_grad():
            outputs_cuda = model_cuda(x_cuda)
        print("\nPASS: CUDA test")
    else:
        print("\nINFO: CUDA not available")
except Exception as e:
    print(f"FAIL: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Loss Computation
print("\n=== TEST 3: Loss Computation ===")
try:
    cfg = ModelConfig(
        n_features=83, d_model=256, layers=4, heads=4, dropout=0.1,
        seq_len=30, branches={"1m": 30, "5m": 36, "15m": 32},
        n_returns=len(TARGET_RETURNS), n_classes=NUM_CLASSES,
    )
    model = MultiTimeframeTransformer(cfg)
    
    batch_size = 8
    x = torch.randn(batch_size, cfg.context_len, cfg.n_features)
    targets = {
        "direction": torch.randint(0, NUM_CLASSES, (batch_size,)),
        "returns": torch.randn(batch_size, len(TARGET_RETURNS)),
        "vol": torch.randn(batch_size),
        "max_up": torch.randn(batch_size),
        "max_down": torch.randn(batch_size),
    }
    
    outputs = model(x)
    total_loss, loss_parts = multihead_loss(outputs, targets, cfg)
    
    print(f"Total loss: {total_loss.item():.4f}")
    print("\nLoss breakdown:")
    for name, val in loss_parts.items():
        print(f"  {name}: {val:.4f}")
    
    total_loss.backward()
    
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.norm() > 0:
            print(f"\nPASS: Gradient found: {name} (norm={param.grad.norm():.6f})")
            has_grad = True
            break
    
    if not has_grad:
        print("FAIL: No gradients found")
    
    if torch.isfinite(total_loss):
        print("PASS: Loss is finite")
    else:
        print("FAIL: Loss is not finite")
except Exception as e:
    print(f"FAIL: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("  Quick test completed")
print("="*70)