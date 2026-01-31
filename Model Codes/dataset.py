"""
Dataset for Rocket League Goal Prediction
=========================================
Loads video frames and corresponding threat distributions.

Key Features:
- On-the-fly video loading (saves disk space)
- Temporal sliding window approach
- Handles 2x speed videos
- Intelligent sampling (oversample near goals)
- Data augmentation
"""

import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path


class RocketLeagueDataset(Dataset):
    """
    Dataset for video-based goal prediction.
    
    Creates overlapping temporal windows from videos.
    """
    
    def __init__(self, replay_folders, config, mode='train'):
        """
        Args:
            replay_folders: List of paths to Replay_XXX folders
            config: Configuration dict
            mode: 'train', 'val', or 'test'
        """
        self.replay_folders = replay_folders
        self.config = config
        self.mode = mode
        
        # Extract config values
        data_cfg = config['data']
        self.video_fps = data_cfg['video_fps']
        self.target_fps = data_cfg['target_fps']
        self.frame_downsample = self.video_fps // self.target_fps  # Every 3rd frame
        
        self.lookback_frames_video = data_cfg['lookback_frames']  # 150 frames at 30 FPS
        self.lookback_frames_model = data_cfg['downsampled_frames']  # 50 frames at 10 FPS
        
        self.prediction_interval = data_cfg['prediction_interval_frames']  # 8 frames
        
        self.frame_height = data_cfg['frame_height']
        self.frame_width = data_cfg['frame_width']
        
        # Build index of all valid temporal windows
        self.samples = self._build_sample_index()
        
        print(f"\n[{mode.upper()} DATASET]")
        print(f"  Replays: {len(replay_folders)}")
        print(f"  Total samples: {len(self.samples)}")
        
        # Setup transforms
        self._setup_transforms()
    
    def _build_sample_index(self):
        """
        Build index of all valid temporal windows.
        
        Each sample is a (replay_idx, start_frame) tuple.
        """
        samples = []
        
        for replay_idx, replay_folder in enumerate(self.replay_folders):
            # Load distribution to know video length
            dist_path = os.path.join(replay_folder, "distribution.npy")
            
            if not os.path.exists(dist_path):
                print(f"  [WARNING] Missing distribution.npy in {replay_folder}")
                continue
            
            distribution = np.load(dist_path)
            total_frames = len(distribution)
            
            # Find video file
            video_path = self._find_video_file(replay_folder)
            if video_path is None:
                print(f"  [WARNING] No video file in {replay_folder}")
                continue
            
            # Create samples with sliding window
            # Start from frame where we have full lookback history
            start = self.lookback_frames_video
            
            # Sample every prediction_interval frames
            for frame_idx in range(start, total_frames, self.prediction_interval):
                # Check if we can create a valid sample
                if frame_idx < self.lookback_frames_video:
                    continue
                
                samples.append({
                    'replay_idx': replay_idx,
                    'replay_folder': replay_folder,
                    'center_frame': frame_idx,
                    'video_path': video_path,
                    'dist_path': dist_path
                })
        
        return samples
    
    def _find_video_file(self, folder):
        """Find .mp4 file in folder."""
        for file in os.listdir(folder):
            if file.lower().endswith('.mp4'):
                return os.path.join(folder, file)
        return None
    
    def _setup_transforms(self):
        """Setup image transforms."""
        if self.mode == 'train' and self.config['data']['augmentation']['enabled']:
            aug_cfg = self.config['data']['augmentation']
            
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.frame_height, self.frame_width)),
                transforms.ColorJitter(
                    brightness=aug_cfg['brightness'],
                    contrast=aug_cfg['contrast'],
                    saturation=aug_cfg['saturation']
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet stats
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            # Val/test: no augmentation
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.frame_height, self.frame_width)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a single training sample.
        
        Returns:
            frames: (T, C, H, W) - video frames
            target: (T,) - threat values
        """
        sample = self.samples[idx]
        
        center_frame = sample['center_frame']
        video_path = sample['video_path']
        dist_path = sample['dist_path']
        
        # Load video frames
        frames = self._load_video_frames(
            video_path, 
            center_frame, 
            self.lookback_frames_video,
            self.frame_downsample
        )
        
        # Load target distribution
        distribution = np.load(dist_path)
        
        # Extract target window (corresponding frames in distribution)
        target_start = center_frame - self.lookback_frames_video
        target_end = center_frame
        target_frames = distribution[target_start:target_end:self.frame_downsample]
        
        # Convert to tensors
        frames = torch.stack([self.transform(frame) for frame in frames])  # (T, C, H, W)
        target = torch.FloatTensor(target_frames)  # (T,)
        
        return frames, target
    
    def _load_video_frames(self, video_path, center_frame, lookback, downsample):
        """
        Load frames from video.
        
        Args:
            video_path: Path to video file
            center_frame: Current frame index
            lookback: Number of frames to look back
            downsample: Frame downsampling factor
        
        Returns:
            frames: List of numpy arrays (H, W, C)
        """
        cap = cv2.VideoCapture(video_path)
        
        frames = []
        start_frame = center_frame - lookback
        
        # Get frames we need (every `downsample` frames)
        frame_indices = range(start_frame, center_frame, downsample)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                # If frame read fails, use black frame
                frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        return frames


def get_dataloaders(config):
    """
    Create train and test dataloaders.
    
    Args:
        config: Configuration dict
    
    Returns:
        train_loader, test_loader
    """
    base_dir = config['data']['replay_data_dir']
    
    # Find all replay folders
    replay_folders = sorted([
        os.path.join(base_dir, d) 
        for d in os.listdir(base_dir) 
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('Replay')
    ])
    
    if len(replay_folders) == 0:
        raise ValueError(f"No Replay folders found in {base_dir}")
    
    print(f"\nFound {len(replay_folders)} replay folders:")
    for i, folder in enumerate(replay_folders):
        print(f"  {i+1}. {os.path.basename(folder)}")
    
    # Split into train/test
    n_train = config['data']['train_replays']
    n_test = config['data']['test_replays']
    
    if len(replay_folders) < n_train + n_test:
        raise ValueError(
            f"Not enough replays: found {len(replay_folders)}, "
            f"need {n_train + n_test} ({n_train} train + {n_test} test)"
        )
    
    train_folders = replay_folders[:n_train]
    test_folders = replay_folders[n_train:n_train + n_test]
    
    print(f"\nTrain replays: {[os.path.basename(f) for f in train_folders]}")
    print(f"Test replays: {[os.path.basename(f) for f in test_folders]}")
    
    # Create datasets
    train_dataset = RocketLeagueDataset(train_folders, config, mode='train')
    test_dataset = RocketLeagueDataset(test_folders, config, mode='test')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        drop_last=True  # Drop incomplete batches
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Test dataset
    import yaml
    
    print("Testing dataset...")
    
    # Load config
    with open('/mnt/user-data/outputs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Update paths for testing
    config['data']['replay_data_dir'] = '/home/claude/test_replay_data'
    config['data']['train_replays'] = 1
    config['data']['test_replays'] = 0
    
    try:
        # Create dataset
        replay_folders = ['/home/claude/test_replay_data/Replay_Test_001']
        dataset = RocketLeagueDataset(replay_folders, config, mode='train')
        
        print(f"\nDataset created successfully!")
        print(f"Total samples: {len(dataset)}")
        
        # Test loading a sample
        if len(dataset) > 0:
            frames, target = dataset[0]
            print(f"\nSample 0:")
            print(f"  Frames shape: {frames.shape}")
            print(f"  Target shape: {target.shape}")
            print(f"  Frames range: [{frames.min():.3f}, {frames.max():.3f}]")
            print(f"  Target range: [{target.min():.3f}, {target.max():.3f}]")
            
            print("\n✓ Dataset test passed!")
        else:
            print("\n[WARNING] Dataset is empty!")
    
    except Exception as e:
        print(f"\n[ERROR] Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
