# Quick Reference Card
## Rocket League Goal Prediction System

---

## 🚀 Essential Commands

### Training
```bash
# Basic training
python train_model.py --config config.yaml

# Resume from checkpoint
python train_model.py --config config.yaml --resume outputs/.../checkpoints/checkpoint_epoch_25.pt

# Monitor training
tensorboard --logdir outputs/rl_goal_prediction_TIMESTAMP/tensorboard
```

### Evaluation
```bash
# Evaluate best model
python evaluate_model.py --checkpoint outputs/.../checkpoints/best_model.pt --config config.yaml

# Evaluate on specific replays
python evaluate_model.py --checkpoint best_model.pt --config config.yaml --replay-dir "path/to/replays"
```

---

## ⚙️ Key Config Settings

### GPU Memory Issues?
```yaml
training:
  batch_size: 2          # Reduce from 4
  mixed_precision: true  # Keep enabled
data:
  target_fps: 5          # Reduce from 10
  num_workers: 2         # Reduce from 4
```

### Training Too Slow?
```yaml
data:
  target_fps: 5                    # Reduce from 10
  prediction_interval_frames: 16    # Increase from 8
hardware:
  num_workers: 6                    # Increase from 4
```

### Model Not Learning?
```yaml
training:
  learning_rate: 0.00005   # Reduce from 0.0001
  loss:
    type: "focal"          # Ensure using focal
    focal_gamma: 3.0       # Increase from 2.0
```

### Want Better Accuracy?
```yaml
model:
  spatial_encoder:
    backbone: "efficientnet_b3"  # Upgrade from b0
  temporal_encoder:
    hidden_size: 512             # Increase from 256
    num_layers: 3                # Increase from 2
training:
  epochs: 100                    # Increase from 50
```

---

## 📊 Metric Interpretations

### During Training (Validation)
| Metric | Good | Okay | Bad | Fix |
|--------|------|------|-----|-----|
| AP | >0.70 | 0.50-0.70 | <0.50 | More data, focal loss |
| MAE | <0.15 | 0.15-0.25 | >0.25 | Check targets, reduce LR |
| Early Det. | >70% | 50-70% | <50% | Increase window, check targets |
| False Alarms | <2/min | 2-4/min | >4/min | Increase threshold, focal loss |

### After Evaluation
| Metric | Good | Okay | Bad | Fix |
|--------|------|------|-----|-----|
| MAE | <0.10 | 0.10-0.20 | >0.20 | More training, better model |
| Correlation | >0.80 | 0.70-0.80 | <0.70 | Check distribution quality |
| Peak Error | <15 frames | 15-30 | >30 | Temporal resolution, more context |

---

## 🐛 Quick Fixes

### CUDA Out of Memory
```python
# In config.yaml
batch_size: 2
target_fps: 5
```

### Loss = NaN
```python
# In config.yaml
learning_rate: 0.00001
grad_clip: 0.5
```

### Model Predicts All Zeros
```python
# Check your data
python -c "import numpy as np; d = np.load('Replay Data/Replay_001/distribution.npy'); print(f'Mean: {d.mean():.4f}, Max: {d.max():.4f}, Non-zero: {(d>0).sum()}')"

# Use focal loss
loss:
  type: "focal"
  focal_alpha: 0.25
  focal_gamma: 2.0
```

### Training Crashes
```python
# Reduce complexity
batch_size: 1
num_workers: 0
mixed_precision: false
```

---

## 📁 File Locations

### Inputs
```
Replay Data/Replay_XXX/
├── *.mp4              ← Video
├── goals.csv          ← Goal times
└── distribution.npy   ← Target (from create_threat_distribution.py)
```

### Outputs
```
outputs/rl_goal_prediction_TIMESTAMP/
├── checkpoints/best_model.pt      ← Use for evaluation
├── config.yaml                    ← Config used
└── training_history.json          ← Training curves

evaluation_results/
├── Replay_XXX_predictions.npy     ← Model predictions
├── Replay_XXX_comparison.png      ← Visualization
└── evaluation_results.json        ← All metrics
```

---

## 🎯 Typical Workflow

```bash
# 1. Setup (once)
pip install -r requirements.txt

# 2. Prepare data (per replay)
python create_threat_distribution.py --n-seconds 10

# 3. Train model (2-3 hours)
python train_model.py --config config.yaml

# 4. Evaluate (5-10 minutes)
python evaluate_model.py \
  --checkpoint outputs/.../checkpoints/best_model.pt \
  --config config.yaml

# 5. Review results
cat evaluation_results/evaluation_results.json
open evaluation_results/Replay_001_comparison.png
```

---

## 🔥 Pro Tips

1. **Start small**: Train with batch_size=2, epochs=10 to verify everything works
2. **Monitor GPU**: `watch -n 1 nvidia-smi` to check utilization
3. **Save checkpoints**: They're your friend if training crashes
4. **Compare metrics**: Train vs Test to detect overfitting
5. **Visualize**: Look at the comparison plots, not just numbers
6. **Iterate fast**: Small experiments > long training runs

---

## 📞 Getting Help

### Check logs
```bash
# Training logs
tail -f outputs/.../tensorboard/events.out.tfevents.*

# System logs
dmesg | tail  # For OOM errors
```

### Debug mode
```python
# Add to top of train_model.py
import torch
torch.autograd.set_detect_anomaly(True)
```

### Test components
```python
# Test dataset
python dataset.py

# Test model
python model.py

# Test losses
python losses.py

# Test metrics
python metrics.py
```

---

## 📈 Performance Targets

### Minimum (with 8 replays)
- Training completes without errors
- Val AP > 0.50
- Model learns something (not all zeros/ones)

### Good (with 8 replays)
- Val AP > 0.70
- MAE < 0.15
- Visual comparison looks reasonable

### Excellent (with 20+ replays)
- Val AP > 0.85
- MAE < 0.10
- Correlation > 0.90
- Ready for production

---

**Print this and keep it handy! 📄**
