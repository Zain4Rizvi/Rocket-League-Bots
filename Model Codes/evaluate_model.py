"""
Post-Training Evaluation Script
================================
Generates full predictions for all videos and compares with target distributions.

Usage:
    python evaluate_model.py --checkpoint path/to/best_model.pt --config config.yaml
"""

import argparse
import yaml
import numpy as np
import torch
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from model import get_model
from metrics import DistributionComparison


class FullVideoDataset(Dataset):
    """
    Dataset for generating predictions on entire videos.
    Unlike training dataset, this processes videos sequentially.
    """
    
    def __init__(self, video_path, config):
        self.video_path = video_path
        self.config = config
        
        # Video properties
        cap = cv2.VideoCapture(video_path)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Config
        data_cfg = config['data']
        self.lookback_frames = data_cfg['lookback_frames']
        self.frame_downsample = data_cfg['video_fps'] // data_cfg['target_fps']
        self.prediction_interval = data_cfg['prediction_interval_frames']
        
        self.frame_height = data_cfg['frame_height']
        self.frame_width = data_cfg['frame_width']
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.frame_height, self.frame_width)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Build sample indices
        self.frame_indices = list(range(
            self.lookback_frames,
            self.total_frames,
            self.prediction_interval
        ))
    
    def __len__(self):
        return len(self.frame_indices)
    
    def __getitem__(self, idx):
        center_frame = self.frame_indices[idx]
        
        # Load frames
        frames = self._load_frames(center_frame)
        
        # Transform
        frames_tensor = torch.stack([self.transform(f) for f in frames])
        
        return frames_tensor, center_frame
    
    def _load_frames(self, center_frame):
        """Load frames for prediction."""
        cap = cv2.VideoCapture(self.video_path)
        
        frames = []
        start_frame = center_frame - self.lookback_frames
        frame_indices = range(start_frame, center_frame, self.frame_downsample)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        return frames


class ModelEvaluator:
    """
    Evaluates trained model on full videos.
    """
    
    def __init__(self, checkpoint_path, config_path):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(self.config['hardware']['device'])
        
        # Load model
        print("Loading model...")
        self.model = get_model(self.config, self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from: {checkpoint_path}")
        print(f"Trained for {checkpoint['epoch']} epochs")
        print(f"Best val AP: {checkpoint.get('best_val_ap', 'N/A')}")
        
        # Setup output directory
        self.output_dir = Path("evaluation_results")
        self.output_dir.mkdir(exist_ok=True)
        
        self.dist_comp = DistributionComparison()
    
    @torch.no_grad()
    def predict_full_video(self, video_path):
        """
        Generate predictions for entire video.
        
        Returns:
            predicted_distribution: (total_frames,) array
        """
        # Create dataset
        dataset = FullVideoDataset(video_path, self.config)
        loader = DataLoader(
            dataset,
            batch_size=1,  # Process one window at a time
            shuffle=False,
            num_workers=0
        )
        
        # Initialize prediction array
        total_frames = dataset.total_frames
        predicted_dist = np.zeros(total_frames)
        prediction_counts = np.zeros(total_frames)  # For averaging overlaps
        
        # Generate predictions
        for frames, center_frame in tqdm(loader, desc="Predicting"):
            frames = frames.to(self.device)  # (1, T, C, H, W)
            center_frame = center_frame.item()
            
            # Forward pass
            predictions = self.model(frames)  # (1, T)
            predictions = predictions.cpu().numpy()[0]  # (T,)
            
            # Map predictions back to video frames
            lookback_frames = self.config['data']['lookback_frames']
            frame_downsample = self.config['data']['video_fps'] // self.config['data']['target_fps']
            
            start_frame = center_frame - lookback_frames
            video_frame_indices = range(start_frame, center_frame, frame_downsample)
            
            for video_idx, pred in zip(video_frame_indices, predictions):
                if 0 <= video_idx < total_frames:
                    predicted_dist[video_idx] += pred
                    prediction_counts[video_idx] += 1
        
        # Average overlapping predictions
        mask = prediction_counts > 0
        predicted_dist[mask] /= prediction_counts[mask]
        
        return predicted_dist
    
    def evaluate_replay(self, replay_folder):
        """
        Evaluate model on a single replay.
        
        Returns:
            results: Dict with predictions, targets, and metrics
        """
        replay_name = Path(replay_folder).name
        print(f"\nEvaluating: {replay_name}")
        
        # Find video file
        video_files = list(Path(replay_folder).glob("*.mp4"))
        if not video_files:
            print(f"  [ERROR] No video file found")
            return None
        
        video_path = str(video_files[0])
        
        # Load target distribution
        dist_path = Path(replay_folder) / "distribution.npy"
        if not dist_path.exists():
            print(f"  [ERROR] No distribution.npy found")
            return None
        
        target_dist = np.load(dist_path)
        
        # Generate predictions
        predicted_dist = self.predict_full_video(video_path)
        
        # Ensure same length
        min_len = min(len(predicted_dist), len(target_dist))
        predicted_dist = predicted_dist[:min_len]
        target_dist = target_dist[:min_len]
        
        # Compare distributions
        metrics = self.dist_comp.compare_distributions(predicted_dist, target_dist)
        
        # Print metrics
        print(f"\n  Results:")
        print(f"    MAE: {metrics['mae']:.4f}")
        print(f"    RMSE: {metrics['rmse']:.4f}")
        print(f"    Correlation: {metrics['correlation']:.4f}")
        print(f"    Peak Alignment Error: {metrics['peak_alignment_error']:.2f} frames")
        print(f"    Shape Similarity: {metrics['shape_similarity']:.4f}")
        
        # Create visualization
        self.plot_comparison(
            predicted_dist,
            target_dist,
            replay_name,
            metrics
        )
        
        # Save predictions
        save_path = self.output_dir / f"{replay_name}_predictions.npy"
        np.save(save_path, predicted_dist)
        print(f"    Saved predictions: {save_path}")
        
        return {
            'replay_name': replay_name,
            'predicted_dist': predicted_dist,
            'target_dist': target_dist,
            'metrics': metrics
        }
    
    def plot_comparison(self, predicted, target, replay_name, metrics):
        """
        Create comparison plot of predicted vs target distribution.
        """
        fig, axes = plt.subplots(3, 1, figsize=(16, 10))
        
        frames = np.arange(len(predicted))
        
        # Plot 1: Overlay
        ax = axes[0]
        ax.plot(frames, target, 'b-', label='Target', alpha=0.7, linewidth=1.5)
        ax.plot(frames, predicted, 'r-', label='Predicted', alpha=0.7, linewidth=1.5)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Threat Level')
        ax.set_title(f'{replay_name} - Prediction vs Target', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        
        # Add metrics text
        metrics_text = f"MAE: {metrics['mae']:.4f} | RMSE: {metrics['rmse']:.4f} | Corr: {metrics['correlation']:.4f}"
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot 2: Error (difference)
        ax = axes[1]
        error = predicted - target
        ax.plot(frames, error, 'purple', linewidth=1, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.fill_between(frames, 0, error, alpha=0.3, color='purple')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Error (Pred - Target)')
        ax.set_title('Prediction Error', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add error statistics
        mae = np.abs(error).mean()
        std = error.std()
        error_text = f"Mean: {error.mean():.4f} | MAE: {mae:.4f} | Std: {std:.4f}"
        ax.text(0.02, 0.98, error_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Plot 3: Scatter plot
        ax = axes[2]
        ax.scatter(target, predicted, alpha=0.5, s=1)
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Prediction')
        ax.set_xlabel('Target Threat')
        ax.set_ylabel('Predicted Threat')
        ax.set_title('Target vs Predicted (Scatter)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"{replay_name}_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved plot: {plot_path}")
    
    def evaluate_all_replays(self, replay_data_dir):
        """
        Evaluate model on all replays (train + test).
        """
        replay_folders = sorted([
            str(p) for p in Path(replay_data_dir).iterdir()
            if p.is_dir() and p.name.startswith('Replay')
        ])
        
        print(f"\nFound {len(replay_folders)} replays to evaluate")
        
        all_results = []
        
        for replay_folder in replay_folders:
            result = self.evaluate_replay(replay_folder)
            if result:
                all_results.append(result)
        
        # Aggregate statistics
        self.print_summary(all_results)
        
        # Save detailed results
        self.save_results(all_results)
        
        return all_results
    
    def print_summary(self, all_results):
        """Print summary statistics across all replays."""
        if not all_results:
            print("\nNo results to summarize")
            return
        
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        
        # Separate train and test (assuming naming convention)
        n_train = self.config['data']['train_replays']
        train_results = all_results[:n_train]
        test_results = all_results[n_train:]
        
        for split_name, results in [("TRAIN", train_results), ("TEST", test_results)]:
            if not results:
                continue
            
            print(f"\n{split_name} SET ({len(results)} replays):")
            
            # Aggregate metrics
            mae_list = [r['metrics']['mae'] for r in results]
            rmse_list = [r['metrics']['rmse'] for r in results]
            corr_list = [r['metrics']['correlation'] for r in results]
            peak_err_list = [r['metrics']['peak_alignment_error'] for r in results]
            
            print(f"  MAE:        {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}")
            print(f"  RMSE:       {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}")
            print(f"  Correlation: {np.mean(corr_list):.4f} ± {np.std(corr_list):.4f}")
            print(f"  Peak Error:  {np.mean(peak_err_list):.2f} ± {np.std(peak_err_list):.2f} frames")
            
            # Per-replay breakdown
            print(f"\n  Per-Replay MAE:")
            for r in results:
                print(f"    {r['replay_name']}: {r['metrics']['mae']:.4f}")
        
        print("\n" + "=" * 80)
    
    def save_results(self, all_results):
        """Save detailed results to JSON."""
        results_json = {
            'replays': []
        }
        
        for r in all_results:
            replay_data = {
                'name': r['replay_name'],
                'metrics': {k: float(v) if not isinstance(v, str) else v 
                           for k, v in r['metrics'].items()}
            }
            results_json['replays'].append(replay_data)
        
        # Aggregate statistics
        mae_list = [r['metrics']['mae'] for r in all_results]
        results_json['aggregate'] = {
            'mean_mae': float(np.mean(mae_list)),
            'std_mae': float(np.std(mae_list)),
            'mean_rmse': float(np.mean([r['metrics']['rmse'] for r in all_results])),
            'mean_correlation': float(np.mean([r['metrics']['correlation'] for r in all_results]))
        }
        
        save_path = self.output_dir / "evaluation_results.json"
        with open(save_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"\nResults saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Goal Prediction Model")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--replay-dir', type=str, default=None,
                       help='Path to Replay Data directory (default: from config)')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(args.checkpoint, args.config)
    
    # Get replay directory
    if args.replay_dir:
        replay_dir = args.replay_dir
    else:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        replay_dir = config['data']['replay_data_dir']
    
    # Evaluate all replays
    evaluator.evaluate_all_replays(replay_dir)
    
    print("\n✓ Evaluation complete!")
    print(f"Results saved to: {evaluator.output_dir}")


if __name__ == "__main__":
    main()
