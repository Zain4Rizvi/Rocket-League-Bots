"""
Rocket League Goal Prediction - Training Script
===============================================
Complete training pipeline with evaluation and visualization.

Usage:
    python train_model.py --config config.yaml
"""

import os
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import json
from datetime import datetime

# Import our modules
from model import get_model
from dataset import get_dataloaders, RocketLeagueDataset
from losses import get_loss_function
from metrics import MetricsCalculator, DistributionComparison


class Trainer:
    """
    Handles model training, validation, and checkpointing.
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['hardware']['device'])
        
        # Set random seed
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        
        # Create output directories
        self.setup_directories()
        
        # Initialize model
        print("=" * 80)
        print("INITIALIZING MODEL")
        print("=" * 80)
        self.model = get_model(config, self.device)
        
        # Setup optimization
        self.setup_optimization()
        
        # Setup data loaders
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)
        self.train_loader, self.test_loader = get_dataloaders(config)
        
        # Setup metrics
        self.metrics_calc = MetricsCalculator(config)
        self.dist_comp = DistributionComparison()
        
        # Setup logging
        self.setup_logging()
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_ap = 0.0
        self.epochs_without_improvement = 0
        
        # Mixed precision training
        self.use_amp = config['training']['mixed_precision']
        if self.use_amp:
            self.scaler = GradScaler()
            print("\n[INFO] Mixed precision training enabled")
    
    def setup_directories(self):
        """Create output directories."""
        self.output_dir = Path(self.config['logging']['output_dir'])
        self.experiment_name = self.config['logging']['experiment_name']
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"{self.experiment_name}_{timestamp}"
        
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.plots_dir = self.run_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Save config
        with open(self.run_dir / "config.yaml", 'w') as f:
            yaml.dump(self.config, f)
        
        print(f"\n[INFO] Output directory: {self.run_dir}")
    
    def setup_optimization(self):
        """Setup optimizer, loss, and scheduler."""
        # Optimizer
        lr = self.config['training']['learning_rate']
        wd = self.config['training']['weight_decay']
        
        if self.config['training']['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=wd
            )
        elif self.config['training']['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=wd
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config['training']['optimizer']}")
        
        # Loss function
        self.criterion = get_loss_function(self.config)
        
        # Scheduler
        scheduler_type = self.config['training']['scheduler']
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs']
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.5
            )
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
        else:
            self.scheduler = None
    
    def setup_logging(self):
        """Setup TensorBoard logging."""
        if self.config['logging']['use_tensorboard']:
            log_dir = self.run_dir / "tensorboard"
            self.writer = SummaryWriter(log_dir=str(log_dir))
            print(f"[INFO] TensorBoard logging to: {log_dir}")
        else:
            self.writer = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_metrics': [],
            'val_loss': [],
            'val_metrics': []
        }
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_loss_components = {'total': 0.0, 'main': 0.0, 'smooth': 0.0}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}")
        
        for batch_idx, (frames, targets) in enumerate(pbar):
            frames = frames.to(self.device)  # (B, T, C, H, W)
            targets = targets.to(self.device)  # (B, T)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    predictions = self.model(frames)  # (B, T)
                    loss, loss_dict = self.criterion(predictions, targets)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config['training']['grad_clip'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['grad_clip']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(frames)
                loss, loss_dict = self.criterion(predictions, targets)
                loss.backward()
                
                if self.config['training']['grad_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['grad_clip']
                    )
                
                self.optimizer.step()
            
            # Accumulate loss
            epoch_loss += loss.item()
            for key in loss_dict:
                epoch_loss_components[key] += loss_dict[key]
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Log to TensorBoard
            if self.writer and batch_idx % self.config['logging']['log_every'] == 0:
                global_step = self.epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('train/batch_loss', loss.item(), global_step)
        
        # Average losses
        n_batches = len(self.train_loader)
        epoch_loss /= n_batches
        for key in epoch_loss_components:
            epoch_loss_components[key] /= n_batches
        
        return epoch_loss, epoch_loss_components
    
    @torch.no_grad()
    def validate(self):
        """Validate on test set."""
        self.model.eval()
        
        val_loss = 0.0
        all_predictions = []
        all_targets = []
        
        for frames, targets in tqdm(self.test_loader, desc="Validation"):
            frames = frames.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    predictions = self.model(frames)
                    loss, _ = self.criterion(predictions, targets)
            else:
                predictions = self.model(frames)
                loss, _ = self.criterion(predictions, targets)
            
            val_loss += loss.item()
            
            # Collect for metrics
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
        
        val_loss /= len(self.test_loader)
        
        # Calculate metrics
        predictions_np = np.concatenate(all_predictions, axis=0)
        targets_np = np.concatenate(all_targets, axis=0)
        
        metrics = self.metrics_calc.calculate_all_metrics(
            predictions_np,
            targets_np,
            fps=self.config['data']['target_fps']
        )
        
        return val_loss, metrics
    
    def save_checkpoint(self, is_best=False, suffix=""):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_ap': self.best_val_ap,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest
        path = self.checkpoint_dir / f"checkpoint_epoch_{self.epoch}{suffix}.pt"
        torch.save(checkpoint, path)
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"[SAVE] Best model saved: {best_path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_ap = checkpoint.get('best_val_ap', 0.0)
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"[LOAD] Checkpoint loaded from: {path}")
    
    def train(self):
        """Main training loop."""
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        print(f"Epochs: {self.config['training']['epochs']}")
        print(f"Batch size: {self.config['training']['batch_size']}")
        print(f"Learning rate: {self.config['training']['learning_rate']}")
        print(f"Device: {self.device}")
        print("=" * 80 + "\n")
        
        for epoch in range(self.config['training']['epochs']):
            self.epoch = epoch
            
            # Unfreeze backbone after N epochs
            if epoch == self.config['model']['spatial_encoder'].get('freeze_epochs', 0):
                if epoch > 0:
                    self.model.unfreeze_backbone()
                    print(f"\n[INFO] Unfreezing backbone at epoch {epoch}")
            
            # Train
            train_loss, train_loss_components = self.train_epoch()
            
            # Validate
            val_loss, val_metrics = self.validate()
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Log
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.config['training']['epochs']}")
            print(f"  Train Loss: {train_loss:.4f} (main: {train_loss_components['main']:.4f}, smooth: {train_loss_components['smooth']:.4f})")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val AP: {val_metrics['average_precision']:.4f}")
            print(f"  Val MAE: {val_metrics['mae']:.4f}")
            print(f"  Early Detection: {val_metrics['early_detection_rate']:.2%}")
            print(f"  False Alarms/min: {val_metrics['false_alarm_rate']:.2f}")
            
            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar('train/loss', train_loss, epoch)
                self.writer.add_scalar('val/loss', val_loss, epoch)
                for key, value in val_metrics.items():
                    self.writer.add_scalar(f'val/{key}', value, epoch)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Check for improvement
            improved = False
            if val_metrics['average_precision'] > self.best_val_ap:
                self.best_val_ap = val_metrics['average_precision']
                self.best_val_loss = val_loss
                improved = True
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoints
            if (epoch + 1) % self.config['training']['save_every'] == 0:
                self.save_checkpoint(suffix=f"_regular")
            
            if improved:
                self.save_checkpoint(is_best=True)
            
            # Early stopping
            if self.config['training']['early_stopping']['enabled']:
                patience = self.config['training']['early_stopping']['patience']
                if self.epochs_without_improvement >= patience:
                    print(f"\n[EARLY STOP] No improvement for {patience} epochs")
                    break
        
        # Save final model
        self.save_checkpoint(suffix="_final")
        
        # Save training history
        history_path = self.run_dir / "training_history.json"
        with open(history_path, 'w') as f:
            # Convert numpy types to Python types for JSON
            history_json = {
                'train_loss': [float(x) for x in self.history['train_loss']],
                'val_loss': [float(x) for x in self.history['val_loss']],
                'val_metrics': [
                    {k: float(v) for k, v in m.items()}
                    for m in self.history['val_metrics']
                ]
            }
            json.dump(history_json, f, indent=2)
        
        print(f"\n[SAVE] Training history saved: {history_path}")
        
        if self.writer:
            self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train Rocket League Goal Prediction Model")
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer
    trainer = Trainer(config)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    try:
        trainer.train()
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE!")
        print("=" * 80)
        print(f"Best validation AP: {trainer.best_val_ap:.4f}")
        print(f"Output directory: {trainer.run_dir}")
        
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Training interrupted by user")
        trainer.save_checkpoint(suffix="_interrupted")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
