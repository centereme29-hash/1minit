#!/usr/bin/env python3
"""Test suite for 1minit model."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from src.params import FEATURES_DIR, MODELS_DIR, NUM_CLASSES, RANDOM_STATE, TARGET_RETURNS
from src.dataset import build_matrix
from src.model import ModelConfig, MultiTimeframeTransformer, count_params, multihead_loss
from src.baselines import LABEL_MAP

DATA_DIR = FEATURES_DIR  # Use the same directory where train.csv, val.csv, test.csv are stored


def load_split(split_name: str):
    """Load a pre-split CSV file."""
    path = DATA_DIR / f"{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run src/dataset.py first.")
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    feature_cols = [c for c in df.columns if c not in ["time", "label", "future_ret_1", "future_ret_2", 
                                                         "future_ret_3", "future_ret_5", "future_ret_10",
                                                         "normalized_move", "future_max_up", "future_max_down", "future_vol"]]
    return df, feature_cols


def test_data():
    """Test 1: Data loading."""
    print("\n=== TEST 1: Data Loading ===")
    try:
        # Load train, val, test splits
        train_df, feature_cols = load_split("train")
        val_df, _ = load_split("val")
        test_df, _ = load_split("test")
        
        print(f"Train: {len(train_df)} samples")
        print(f"Val: {len(val_df)} samples")
        print(f"Test: {len(test_df)} samples")
        print(f"Features: {len(feature_cols)}")
        
        # Check train data
        m = train_df
        print(f"Shape: {m.shape}")
        print(f"Time range: {m['time'].min()} -> {m['time'].max()}")
        
        label_dist = m["label"].value_counts().sort_index()
        print("\nLabel distribution (train):")
        for label, count in label_dist.items():
            name = {0: "DOWN", 1: "NO_MOVE", 2: "UP"}.get(label, str(label))
            print(f"  {name}: {count} ({100*count/len(m):.2f}%)")
        
        nan_count = m[feature_cols].isna().sum().sum()
        if nan_count > 0:
            print(f"FAIL: Found {nan_count} NaN values in features")
            return False
        print("PASS: No NaN values in features")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model():
    """Test 2: Model architecture."""
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
        print(f"\nModel created: {n_params:,} parameters ({n_params/1e6:.2f}M)")
        
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
            print("\nCUDA test passed")
        else:
            print("\nCUDA not available")
        
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss():
    """Test 3: Loss computation."""
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
                print(f"\nGradient found: {name} (norm={param.grad.norm():.6f})")
                has_grad = True
                break
        
        if not has_grad:
            print("FAIL: No gradients found")
            return False
        
        if not torch.isfinite(total_loss):
            print("FAIL: Loss is not finite")
            return False
        
        print("PASS: Loss is finite")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training():
    """Test 4: Training loop."""
    print("\n=== TEST 4: Training Loop ===")
    try:
        # Load train data
        train_df, feature_cols = load_split("train")
        m = train_df
        
        X = m[feature_cols].to_numpy(dtype=np.float32)
        
        targets = {
            "direction": m["label"].map(LABEL_MAP).to_numpy(dtype=np.int64),
            "returns": m[[f"future_ret_{r}" for r in TARGET_RETURNS]].to_numpy(dtype=np.float32),
            "vol": m["future_vol"].to_numpy(dtype=np.float32),
            "max_up": m["future_max_up"].to_numpy(dtype=np.float32),
            "max_down": m["future_max_down"].to_numpy(dtype=np.float32),
        }
        
        print(f"Data: {X.shape[0]} samples, {len(feature_cols)} features")
        
        n_samples = min(2000, len(X))
        indices = np.random.RandomState(RANDOM_STATE).choice(len(X), size=n_samples, replace=False)
        split = int(0.8 * n_samples)
        train_idx = np.sort(indices[:split])
        val_idx = np.sort(indices[split:])
        
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
        
        cfg = ModelConfig(
            n_features=len(feature_cols), d_model=128, layers=2, heads=4,
            dropout=0.1, seq_len=30,
            branches={"1m": 30, "5m": 36, "15m": 32},
            n_returns=len(TARGET_RETURNS), n_classes=NUM_CLASSES,
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MultiTimeframeTransformer(cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        print(f"Model: {count_params(model):,} params on {device}")
        
        context_len = cfg.context_len
        batch_size = 32
        n_epochs = 2
        
        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            
            np.random.shuffle(train_idx)
            for i in range(0, len(train_idx), batch_size):
                batch_idx = train_idx[i:i+batch_size]
                if len(batch_idx) < 2:
                    continue
                
                x_batch, y_batch = [], {k: [] for k in targets}
                valid = True
                for idx in batch_idx:
                    start = idx - context_len + 1
                    if start < 0:
                        valid = False
                        break
                    x_batch.append(X[start:idx+1])
                    for k, v in targets.items():
                        y_batch[k].append(v[idx])
                
                if not valid or len(x_batch) < 2:
                    continue
                
                x_tensor = torch.from_numpy(np.array(x_batch)).to(device)
                y_tensor = {
                    k: torch.from_numpy(np.array(v, dtype=np.int64 if k == "direction" else np.float32)).to(device)
                    for k, v in y_batch.items()
                }
                
                optimizer.zero_grad()
                outputs = model(x_tensor)
                loss, _ = multihead_loss(outputs, y_tensor, cfg)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / max(n_batches, 1)
            
            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for i in range(0, len(val_idx), batch_size):
                    batch_idx = val_idx[i:i+batch_size]
                    if len(batch_idx) < 2:
                        continue
                    
                    x_batch, y_batch = [], {k: [] for k in targets}
                    valid = True
                    for idx in batch_idx:
                        start = idx - context_len + 1
                        if start < 0:
                            valid = False
                            break
                        x_batch.append(X[start:idx+1])
                        for k, v in targets.items():
                            y_batch[k].append(v[idx])
                    
                    if not valid or len(x_batch) < 2:
                        continue
                    
                    x_tensor = torch.from_numpy(np.array(x_batch)).to(device)
                    y_tensor = {
                        k: torch.from_numpy(np.array(v, dtype=np.int64 if k == "direction" else np.float32)).to(device)
                        for k, v in y_batch.items()
                    }
                    
                    outputs = model(x_tensor)
                    loss, _ = multihead_loss(outputs, y_tensor, cfg)
                    val_loss += loss.item()
                    n_val += 1
            
            avg_val = val_loss / max(n_val, 1)
            print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}, val_loss={avg_val:.4f}")
        
        print("PASS: Training completed")
        
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "config": cfg}, MODELS_DIR / "test_model.pt")
        print(f"Model saved to {MODELS_DIR / 'test_model.pt'}")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference():
    """Test 5: Inference."""
    print("\n=== TEST 5: Inference ===")
    try:
        # Load train data (could also use test data)
        train_df, feature_cols = load_split("train")
        m = train_df
        X = m[feature_cols].to_numpy(dtype=np.float32)
        
        cfg = ModelConfig(
            n_features=len(feature_cols), d_model=128, layers=2, heads=4,
            dropout=0.1, seq_len=30,
            branches={"1m": 30, "5m": 36, "15m": 32},
            n_returns=len(TARGET_RETURNS), n_classes=NUM_CLASSES,
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MultiTimeframeTransformer(cfg).to(device)
        model.eval()
        
        context_len = cfg.context_len
        end_idx = len(X) - 1
        start_idx = end_idx - context_len + 1
        
        if start_idx < 0:
            print("FAIL: Not enough data")
            return False
        
        x = torch.from_numpy(X[start_idx:end_idx+1].unsqueeze(0)).to(device)
        
        with torch.no_grad():
            outputs = model(x)
        
        probs = torch.softmax(outputs["direction"], dim=-1).cpu().numpy()
        pred_class = np.argmax(probs, axis=-1)[0]
        confidence = probs[0, pred_class]
        
        label_map_rev = {v: k for k, v in LABEL_MAP.items()}
        pred_label = label_map_rev.get(pred_class, str(pred_class))
        
        print(f"Prediction: {pred_label} (confidence: {confidence:.2%})")
        print(f"  Probabilities: DOWN={probs[0,0]:.2%}, NO_MOVE={probs[0,1]:.2%}, UP={probs[0,2]:.2%}")
        print(f"  Returns: {outputs['returns'].cpu().numpy()[0]}")
        print(f"  Vol: {outputs['vol'].cpu().numpy()[0]:.6f}")
        
        print("PASS: Inference successful")
        return True
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("  1minit Model Test Suite")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Device: {torch.cuda.get_device_name(0)}")
    print("="*70)
    
    results = {
        "Data Loading": test_data(),
        "Model Architecture": test_model(),
        "Loss Computation": test_loss(),
        "Training Loop": test_training(),
        "Inference": test_inference(),
    }
    
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    
    passed = 0
    failed = 0
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{name:25s}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    print("="*70 + "\n")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())