"""
Rocket League Goal Prediction Model
===================================
Temporal CNN-LSTM architecture for goal prediction.

Architecture:
  Input: (B, T, C, H, W) - batch of video sequences
  │
  ├─> Spatial Encoder (EfficientNet) - extract features per frame
  │   Output: (B, T, F) where F = feature dim
  │
  ├─> Temporal Encoder (LSTM) - model temporal dependencies
  │   Output: (B, T, H) where H = hidden dim
  │
  └─> Prediction Head (Dense layers) - predict threat
      Output: (B, T, 1) - threat probability per frame
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights, ResNet18_Weights, ResNet34_Weights


class SpatialEncoder(nn.Module):
    """
    Extracts spatial features from individual frames.
    Uses pretrained CNN backbone (EfficientNet, ResNet, etc.)
    """
    
    def __init__(self, backbone='efficientnet_b0', pretrained=True, freeze=False):
        super().__init__()
        
        self.backbone_name = backbone
        
        if backbone == 'efficientnet_b0':
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.efficientnet_b0(weights=weights)
            self.features = model.features
            self.pool = model.avgpool
            self.feature_dim = 1280
            
        elif backbone == 'resnet18':
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.resnet18(weights=weights)
            # Remove final FC layer
            self.features = nn.Sequential(*list(model.children())[:-2])
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.feature_dim = 512
            
        elif backbone == 'resnet34':
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.resnet34(weights=weights)
            self.features = nn.Sequential(*list(model.children())[:-2])
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.feature_dim = 512
        
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Optionally freeze backbone
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze backbone for fine-tuning."""
        for param in self.features.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) - batch of frames
        
        Returns:
            features: (B, F) - spatial features
        """
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x


class TemporalEncoder(nn.Module):
    """
    Models temporal dependencies across frames.
    """
    
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, 
                 dropout=0.3, encoder_type='lstm', bidirectional=False):
        super().__init__()
        
        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        if encoder_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        elif encoder_type == 'gru':
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        # Output dimension (doubled if bidirectional)
        self.output_dim = hidden_dim * (2 if bidirectional else 1)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, F) - sequence of features
        
        Returns:
            output: (B, T, H) - temporal features
        """
        output, _ = self.rnn(x)
        return output


class PredictionHead(nn.Module):
    """
    Predicts threat probability from temporal features.
    """
    
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.4):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Output [0, 1] probability
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, H) - temporal features
        
        Returns:
            output: (B, T, 1) - threat probabilities
        """
        return self.network(x)


class RocketLeagueGoalPredictor(nn.Module):
    """
    Complete model for goal prediction.
    
    Processes video sequences and outputs threat probability at each frame.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Spatial encoder
        spatial_config = config['model']['spatial_encoder']
        self.spatial_encoder = SpatialEncoder(
            backbone=spatial_config['backbone'],
            pretrained=spatial_config['pretrained'],
            freeze=spatial_config.get('freeze_backbone', False)
        )
        
        # Temporal encoder
        temporal_config = config['model']['temporal_encoder']
        self.temporal_encoder = TemporalEncoder(
            input_dim=self.spatial_encoder.feature_dim,
            hidden_dim=temporal_config['hidden_size'],
            num_layers=temporal_config['num_layers'],
            dropout=temporal_config['dropout'],
            encoder_type=temporal_config['type'],
            bidirectional=temporal_config.get('bidirectional', False)
        )
        
        # Prediction head
        head_config = config['model']['prediction_head']
        self.prediction_head = PredictionHead(
            input_dim=self.temporal_encoder.output_dim,
            hidden_dims=head_config['hidden_dims'],
            dropout=head_config['dropout']
        )
    
    def unfreeze_backbone(self):
        """Unfreeze spatial encoder backbone for fine-tuning."""
        self.spatial_encoder.unfreeze()
    
    def forward(self, video_frames):
        """
        Forward pass through the model.
        
        Args:
            video_frames: (B, T, C, H, W) - batch of video sequences
                B = batch size
                T = temporal length (number of frames)
                C = channels (3 for RGB)
                H, W = height, width
        
        Returns:
            predictions: (B, T) - threat probability at each frame
        """
        batch_size, seq_len, channels, height, width = video_frames.shape
        
        # Reshape to process all frames through spatial encoder
        # (B, T, C, H, W) -> (B*T, C, H, W)
        frames_flat = video_frames.view(batch_size * seq_len, channels, height, width)
        
        # Extract spatial features
        spatial_features = self.spatial_encoder(frames_flat)  # (B*T, F)
        
        # Reshape back to sequence
        # (B*T, F) -> (B, T, F)
        feature_dim = spatial_features.shape[1]
        spatial_features = spatial_features.view(batch_size, seq_len, feature_dim)
        
        # Temporal encoding
        temporal_features = self.temporal_encoder(spatial_features)  # (B, T, H)
        
        # Prediction
        predictions = self.prediction_head(temporal_features)  # (B, T, 1)
        
        # Squeeze last dimension
        predictions = predictions.squeeze(-1)  # (B, T)
        
        return predictions
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model(config, device='cuda'):
    """
    Factory function to create and initialize model.
    
    Args:
        config: Configuration dict
        device: Device to place model on
    
    Returns:
        model: Initialized model
    """
    model = RocketLeagueGoalPredictor(config)
    model = model.to(device)
    
    # Print model summary
    print(f"\nModel: {config['model']['name']}")
    print(f"Backbone: {config['model']['spatial_encoder']['backbone']}")
    print(f"Temporal Encoder: {config['model']['temporal_encoder']['type']}")
    print(f"Total Parameters: {model.count_parameters():,}")
    print(f"Device: {device}\n")
    
    return model


if __name__ == "__main__":
    # Test model
    import yaml
    
    print("Testing model architecture...")
    
    # Load config
    with open('/mnt/user-data/outputs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(config, device=device)
    
    # Test forward pass
    batch_size = 2
    seq_len = 50  # 5 seconds at 10 FPS
    channels = 3
    height = 224
    width = 224
    
    # Dummy input
    dummy_input = torch.randn(batch_size, seq_len, channels, height, width).to(device)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    print("\n✓ Model test passed!")
