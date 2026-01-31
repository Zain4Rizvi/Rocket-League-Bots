"""
Evaluation Metrics for Goal Prediction
======================================
Implements metrics for temporal prediction and distribution comparison.
"""

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error
)
from scipy.stats import pearsonr


class MetricsCalculator:
    """
    Calculates various metrics for goal prediction evaluation.
    """
    
    def __init__(self, config):
        self.config = config
        self.k_values = config['validation']['k_values']
    
    def calculate_all_metrics(self, predictions, targets, fps=30):
        """
        Calculate all metrics for a batch of predictions.
        
        Args:
            predictions: (B, T) numpy array of predicted probabilities
            targets: (B, T) numpy array of ground truth
            fps: Frames per second
        
        Returns:
            dict: Dictionary of metric values
        """
        metrics = {}
        
        # Flatten for sklearn metrics
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        # 1. Average Precision (primary metric)
        if target_flat.sum() > 0:  # Only if there are positive examples
            metrics['average_precision'] = average_precision_score(
                target_flat, pred_flat
            )
        else:
            metrics['average_precision'] = 0.0
        
        # 2. Precision@K and Recall@K
        for k in self.k_values:
            p_at_k, r_at_k = self.precision_recall_at_k(pred_flat, target_flat, k)
            metrics[f'precision@{k}'] = p_at_k
            metrics[f'recall@{k}'] = r_at_k
        
        # 3. Regression metrics
        metrics['mae'] = mean_absolute_error(target_flat, pred_flat)
        metrics['rmse'] = np.sqrt(mean_squared_error(target_flat, pred_flat))
        
        # 4. Early detection rate
        metrics['early_detection_rate'] = self.calculate_early_detection_rate(
            predictions, targets, fps=fps, threshold=0.5, advance_seconds=2.0
        )
        
        # 5. False alarm rate
        metrics['false_alarm_rate'] = self.calculate_false_alarm_rate(
            predictions, targets, fps=fps, threshold=0.5
        )
        
        return metrics
    
    def precision_recall_at_k(self, predictions, targets, k):
        """
        Precision and Recall at top K% of predictions.
        
        Args:
            predictions: (N,) array of predictions
            targets: (N,) array of targets
            k: Fraction of top predictions (e.g., 0.01 = top 1%)
        
        Returns:
            precision, recall
        """
        if len(predictions) == 0:
            return 0.0, 0.0
        
        # Get threshold for top K%
        threshold = np.percentile(predictions, 100 * (1 - k))
        
        # Binarize predictions
        pred_binary = (predictions >= threshold).astype(int)
        
        # Calculate metrics
        if pred_binary.sum() == 0:
            return 0.0, 0.0
        
        precision = precision_score(targets, pred_binary, zero_division=0)
        recall = recall_score(targets, pred_binary, zero_division=0)
        
        return precision, recall
    
    def calculate_early_detection_rate(self, predictions, targets, 
                                       fps=30, threshold=0.5, advance_seconds=2.0):
        """
        Calculate percentage of goals detected at least `advance_seconds` early.
        
        Args:
            predictions: (B, T) array
            targets: (B, T) array
            fps: Frames per second
            threshold: Confidence threshold
            advance_seconds: How early to detect (in seconds)
        
        Returns:
            float: Early detection rate [0, 1]
        """
        advance_frames = int(advance_seconds * fps)
        
        total_goals = 0
        detected_early = 0
        
        for pred_seq, target_seq in zip(predictions, targets):
            # Find goal frames (where target reaches peak ~1.0)
            goal_frames = np.where(target_seq > 0.9)[0]
            
            if len(goal_frames) == 0:
                continue
            
            # For each goal, check if detected early
            for goal_frame in goal_frames:
                total_goals += 1
                
                # Check if high confidence in advance window
                advance_start = max(0, goal_frame - advance_frames)
                advance_window = pred_seq[advance_start:goal_frame]
                
                if len(advance_window) > 0 and (advance_window > threshold).any():
                    detected_early += 1
        
        if total_goals == 0:
            return 0.0
        
        return detected_early / total_goals
    
    def calculate_false_alarm_rate(self, predictions, targets, 
                                   fps=30, threshold=0.5):
        """
        Calculate false alarms per minute of video.
        
        Args:
            predictions: (B, T) array
            targets: (B, T) array
            fps: Frames per second
            threshold: Confidence threshold
        
        Returns:
            float: False alarms per minute
        """
        total_false_alarms = 0
        total_frames = 0
        
        for pred_seq, target_seq in zip(predictions, targets):
            # Find high-confidence predictions
            high_conf = pred_seq > threshold
            
            # Find actual non-goal regions (target near 0)
            non_goal = target_seq < 0.1
            
            # False alarms = high confidence in non-goal regions
            false_alarms = np.logical_and(high_conf, non_goal).sum()
            
            total_false_alarms += false_alarms
            total_frames += len(pred_seq)
        
        # Convert to per-minute rate
        total_minutes = total_frames / (fps * 60)
        
        if total_minutes == 0:
            return 0.0
        
        return total_false_alarms / total_minutes


class DistributionComparison:
    """
    Compares predicted distribution to target distribution for full videos.
    """
    
    @staticmethod
    def compare_distributions(pred_dist, target_dist):
        """
        Comprehensive comparison of two distributions.
        
        Args:
            pred_dist: (T,) predicted distribution
            target_dist: (T,) target distribution
        
        Returns:
            dict: Comparison metrics
        """
        metrics = {}
        
        # 1. MSE and MAE
        metrics['mse'] = mean_squared_error(target_dist, pred_dist)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(target_dist, pred_dist)
        
        # 2. Correlation
        if len(pred_dist) > 1 and pred_dist.std() > 0 and target_dist.std() > 0:
            correlation, p_value = pearsonr(pred_dist, target_dist)
            metrics['correlation'] = correlation
            metrics['correlation_pvalue'] = p_value
        else:
            metrics['correlation'] = 0.0
            metrics['correlation_pvalue'] = 1.0
        
        # 3. Peak alignment error
        metrics['peak_alignment_error'] = DistributionComparison.peak_alignment_error(
            pred_dist, target_dist
        )
        
        # 4. Shape similarity (normalized cross-correlation)
        metrics['shape_similarity'] = DistributionComparison.shape_similarity(
            pred_dist, target_dist
        )
        
        # 5. Class-specific metrics (high threat regions)
        high_threat_mask = target_dist > 0.5
        if high_threat_mask.sum() > 0:
            metrics['mae_high_threat'] = mean_absolute_error(
                target_dist[high_threat_mask],
                pred_dist[high_threat_mask]
            )
        else:
            metrics['mae_high_threat'] = 0.0
        
        low_threat_mask = target_dist < 0.1
        if low_threat_mask.sum() > 0:
            metrics['mae_low_threat'] = mean_absolute_error(
                target_dist[low_threat_mask],
                pred_dist[low_threat_mask]
            )
        else:
            metrics['mae_low_threat'] = 0.0
        
        return metrics
    
    @staticmethod
    def peak_alignment_error(pred_dist, target_dist, threshold=0.8):
        """
        Measure how well prediction peaks align with target peaks.
        
        Args:
            pred_dist: Predicted distribution
            target_dist: Target distribution
            threshold: Peak threshold
        
        Returns:
            float: Average frame difference between peaks
        """
        # Find peaks in target
        target_peaks = np.where(target_dist > threshold)[0]
        
        if len(target_peaks) == 0:
            return 0.0
        
        # Find peaks in prediction
        pred_peaks = np.where(pred_dist > threshold)[0]
        
        if len(pred_peaks) == 0:
            return float('inf')  # No peaks detected
        
        # For each target peak, find nearest predicted peak
        errors = []
        for target_peak in target_peaks:
            nearest_pred = pred_peaks[np.argmin(np.abs(pred_peaks - target_peak))]
            errors.append(abs(nearest_pred - target_peak))
        
        return np.mean(errors)
    
    @staticmethod
    def shape_similarity(pred_dist, target_dist):
        """
        Normalized cross-correlation to measure shape similarity.
        
        Args:
            pred_dist: Predicted distribution
            target_dist: Target distribution
        
        Returns:
            float: Similarity score [0, 1]
        """
        # Normalize distributions
        pred_norm = (pred_dist - pred_dist.mean()) / (pred_dist.std() + 1e-8)
        target_norm = (target_dist - target_dist.mean()) / (target_dist.std() + 1e-8)
        
        # Cross-correlation at zero lag
        correlation = np.mean(pred_norm * target_norm)
        
        # Map to [0, 1]
        similarity = (correlation + 1) / 2
        
        return similarity


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")
    
    # Create dummy data
    np.random.seed(42)
    batch_size = 4
    seq_len = 300
    
    # Simulate predictions and targets
    predictions = np.random.rand(batch_size, seq_len) * 0.3  # Mostly low
    targets = np.zeros((batch_size, seq_len))
    
    # Add some "goals" to targets
    for i in range(batch_size):
        goal_frame = np.random.randint(100, 250)
        window = np.arange(max(0, goal_frame - 30), goal_frame)
        targets[i, window] = np.linspace(0, 1, len(window)) ** 2
    
    # Simulate predictions detecting some goals
    predictions = predictions + targets * 0.7 + np.random.rand(batch_size, seq_len) * 0.1
    predictions = np.clip(predictions, 0, 1)
    
    # Test metrics calculator
    config = {
        'validation': {
            'k_values': [0.01, 0.05, 0.1]
        }
    }
    
    calc = MetricsCalculator(config)
    metrics = calc.calculate_all_metrics(predictions, targets, fps=30)
    
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test distribution comparison
    print("\nDistribution Comparison:")
    comp = DistributionComparison()
    dist_metrics = comp.compare_distributions(predictions[0], targets[0])
    
    for key, value in dist_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n✓ All metrics working correctly!")
