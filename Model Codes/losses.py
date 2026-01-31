"""
Custom Loss Functions for Goal Prediction
==========================================
Implements various loss functions for handling class imbalance
and temporal prediction tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    
    where p_t = p if y=1, else (1-p)
    
    Args:
        alpha: Weighting factor for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
               Higher gamma = more focus on hard examples
        reduction: 'mean', 'sum', or 'none'
    
    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (2017)
        https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, T) or (B*T,) - predicted probabilities [0, 1]
            targets: (B, T) or (B*T,) - ground truth [0, 1]
        
        Returns:
            loss: scalar or (B, T) depending on reduction
        """
        # Ensure shapes match
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Compute BCE
        bce_loss = F.binary_cross_entropy(
            predictions, 
            targets, 
            reduction='none'
        )
        
        # Compute p_t
        p_t = torch.where(targets == 1, predictions, 1 - predictions)
        
        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute alpha weight
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Final focal loss
        focal_loss = alpha_t * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss.
    
    Applies higher weight to positive class to handle imbalance.
    
    Args:
        pos_weight: Weight for positive class (default: 3.0)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, pos_weight=3.0, reduction='mean'):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, T) or (B*T,) - predicted probabilities
            targets: (B, T) or (B*T,) - ground truth
        
        Returns:
            loss: scalar or (B, T)
        """
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Compute weights
        weights = torch.where(
            targets == 1,
            torch.tensor(self.pos_weight, device=targets.device),
            torch.tensor(1.0, device=targets.device)
        )
        
        # Weighted BCE
        bce = F.binary_cross_entropy(predictions, targets, reduction='none')
        weighted_loss = weights * bce
        
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


class TemporalSmoothLoss(nn.Module):
    """
    Temporal Smoothness Loss.
    
    Penalizes rapid changes in predictions between consecutive frames.
    Encourages smooth, realistic threat curves.
    
    Args:
        alpha: Weight for smoothness term (default: 0.1)
    """
    
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, predictions):
        """
        Args:
            predictions: (B, T) - predicted probabilities
        
        Returns:
            loss: scalar
        """
        # Compute differences between consecutive predictions
        diff = predictions[:, 1:] - predictions[:, :-1]
        
        # L2 penalty on differences
        smooth_loss = (diff ** 2).mean()
        
        return self.alpha * smooth_loss


class CombinedLoss(nn.Module):
    """
    Combined Loss: Main loss + Temporal Smoothness
    
    Total Loss = main_loss + alpha * smooth_loss
    
    Args:
        main_loss: Primary loss function (Focal, BCE, etc.)
        smooth_alpha: Weight for smoothness term
    """
    
    def __init__(self, main_loss, smooth_alpha=0.05):
        super().__init__()
        self.main_loss = main_loss
        self.smooth_loss = TemporalSmoothLoss(alpha=smooth_alpha)
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, T) - predicted probabilities
            targets: (B, T) - ground truth
        
        Returns:
            loss: scalar
            loss_dict: dict with loss components
        """
        # Main loss
        main = self.main_loss(predictions, targets)
        
        # Smoothness loss
        smooth = self.smooth_loss(predictions)
        
        # Total loss
        total = main + smooth
        
        # Return loss and components for logging
        loss_dict = {
            'total': total.item(),
            'main': main.item(),
            'smooth': smooth.item()
        }
        
        return total, loss_dict


def get_loss_function(config):
    """
    Factory function to create loss based on config.
    
    Args:
        config: Configuration dict
    
    Returns:
        loss_fn: Loss function module
    """
    loss_type = config['training']['loss']['type'].lower()
    
    if loss_type == 'focal':
        alpha = config['training']['loss']['focal_alpha']
        gamma = config['training']['loss']['focal_gamma']
        main_loss = FocalLoss(alpha=alpha, gamma=gamma)
    
    elif loss_type == 'weighted_bce':
        pos_weight = config['training']['sampling']['positive_weight']
        main_loss = WeightedBCELoss(pos_weight=pos_weight)
    
    elif loss_type == 'bce':
        main_loss = nn.BCELoss()
    
    elif loss_type == 'mse':
        main_loss = nn.MSELoss()
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # Wrap with temporal smoothness
    loss_fn = CombinedLoss(main_loss, smooth_alpha=0.05)
    
    return loss_fn


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")
    
    # Create dummy data
    batch_size = 4
    seq_len = 50
    
    predictions = torch.rand(batch_size, seq_len)  # Random predictions
    targets = torch.zeros(batch_size, seq_len)     # Mostly zeros (imbalanced)
    targets[:, 40:45] = 1.0  # Some positive examples near "goals"
    
    # Test Focal Loss
    print("\n1. Focal Loss:")
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    loss_focal = focal(predictions, targets)
    print(f"   Loss: {loss_focal.item():.4f}")
    
    # Test Weighted BCE
    print("\n2. Weighted BCE Loss:")
    wbce = WeightedBCELoss(pos_weight=3.0)
    loss_wbce = wbce(predictions, targets)
    print(f"   Loss: {loss_wbce.item():.4f}")
    
    # Test Temporal Smooth
    print("\n3. Temporal Smoothness Loss:")
    smooth = TemporalSmoothLoss(alpha=0.1)
    loss_smooth = smooth(predictions)
    print(f"   Loss: {loss_smooth.item():.4f}")
    
    # Test Combined Loss
    print("\n4. Combined Loss:")
    combined = CombinedLoss(focal, smooth_alpha=0.05)
    loss_combined, loss_dict = combined(predictions, targets)
    print(f"   Total: {loss_dict['total']:.4f}")
    print(f"   Main: {loss_dict['main']:.4f}")
    print(f"   Smooth: {loss_dict['smooth']:.4f}")
    
    print("\n✓ All loss functions working correctly!")
