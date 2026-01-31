"""
Rocket League Threat Distribution Generator
============================================
Creates target labels for goal prediction: P(goal in next n seconds)

Key Design Principles:
1. Threat = probability of goal occurring within time window
2. Smooth gradients for better neural network training
3. Independent goals (threat resets after each goal)
4. No baseline - only positive examples within window
5. Dynamic frame count from actual video files

Author: Claude (based on rigorous ML design discussion)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Codes/ -> Src/
REPLAY_DATA_DIR = os.path.join(BASE_DIR, "Replay Data")

# Threat window parameters
N_SECONDS = 10.0            # Time window: P(goal in next N seconds)
                            # Recommended: 3.0 (test 2-4 range empirically)

SMOOTHING = "quadratic"     # Options: "constant", "linear", "quadratic", "exponential"
                            # Recommended: "quadratic" for smooth gradients

# Visualization
PLOT_EACH = True           # Show plot for each replay
SAVE_PLOTS = False          # Save plots to file
PLOT_SMOOTHING_COMPARISON = True  # Create comparison of all smoothing methods

# Video settings
EXPECTED_FPS = 30           # Expected FPS (will verify against actual video)


# =============================================================================
# VIDEO UTILITIES
# =============================================================================

def get_video_properties(video_path):
    """
    Extract FPS and total frame count from video file.
    
    Args:
        video_path: Path to .mp4 video file
        
    Returns:
        tuple: (fps, total_frames) or (None, None) if error
    """
    if not os.path.exists(video_path):
        print(f"  [WARNING] Video file not found: {video_path}")
        return None, None
    
    try:
        video = cv2.VideoCapture(video_path)
        
        if not video.isOpened():
            print(f"  [ERROR] Could not open video: {video_path}")
            return None, None
        
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        video.release()
        
        if fps == 0 or frame_count == 0:
            print(f"  [ERROR] Invalid video properties - FPS: {fps}, Frames: {frame_count}")
            return None, None
        
        duration = frame_count / fps
        
        # Validate FPS is reasonable
        if abs(fps - EXPECTED_FPS) > 5:  # Allow 5 FPS tolerance
            print(f"  [WARNING] Video FPS ({fps:.1f}) differs from expected ({EXPECTED_FPS})")
        
        return fps, frame_count
        
    except Exception as e:
        print(f"  [ERROR] Failed to read video properties: {e}")
        return None, None


def find_video_file(folder_path):
    """
    Find the .mp4 file in the replay folder.
    
    Args:
        folder_path: Path to replay folder
        
    Returns:
        str: Path to video file or None if not found
    """
    # Look for .mp4 files in the folder
    for file in os.listdir(folder_path):
        if file.lower().endswith('.mp4'):
            return os.path.join(folder_path, file)
    
    return None


# =============================================================================
# THREAT DISTRIBUTION FUNCTIONS
# =============================================================================

def create_threat_distribution(total_frames, goal_frames, n_seconds=3.0, fps=60, 
                               smoothing='quadratic'):
    """
    Creates threat distribution modeling P(goal in next n seconds).
    
    Key principles:
    - Threat is ONLY non-zero in the n-second window before goals
    - After a goal, threat IMMEDIATELY resets to 0 (independent events)
    - Goals CANNOT affect frames after previous goals (strict independence)
    - Smoothing provides gradient signal for neural network training
    
    Args:
        total_frames: Total number of frames in replay
        goal_frames: List of frame indices where goals occur
        n_seconds: Time window for prediction (e.g., 3.0 = "goal in next 3 seconds")
        fps: Frames per second
        smoothing: How threat builds as goal approaches
            - 'constant': Step function (1.0 within window, 0.0 outside)
            - 'linear': Linear ramp (t/n)
            - 'quadratic': Smooth quadratic (t/n)^2 - RECOMMENDED
            - 'exponential': Sharp rise near goal
    
    Returns:
        numpy array of shape (total_frames,) with threat values in [0, 1]
    
    Mathematical formulation:
        For each goal at frame g with previous goal at g_prev:
        - Valid window: [max(g_prev + 1, g - n*fps), g)
        - threat[f] = 0 for f >= g (hard reset at goal)
        - threat[f] = 0 for f <= g_prev (previous goal resets)
        - threat[f] = smoothing_function for f in valid window
    """
    threat = np.zeros(total_frames, dtype=np.float32)
    window_frames = int(n_seconds * fps)
    
    # Sort goals to process chronologically
    sorted_goals = sorted(goal_frames)
    
    for i, goal_frame in enumerate(sorted_goals):
        # Determine the valid start of the threat window
        # Cannot start at or before the previous goal frame
        if i == 0:
            # First goal: can look back full window
            window_start = max(0, goal_frame - window_frames)
        else:
            # Subsequent goals: must start AFTER previous goal
            prev_goal = sorted_goals[i - 1]
            # Start at prev_goal + 1 (first frame after reset)
            # or goal_frame - window_frames, whichever is later
            window_start = max(prev_goal + 1, goal_frame - window_frames)
        
        window_end = goal_frame  # Exclusive - threat is 0 at goal frame itself
        
        if window_end <= window_start:
            # No valid window (goal at or before previous goal + 1 frame)
            continue
        
        # Calculate actual window length
        actual_window_length = window_end - window_start
        
        # Generate threat values for this window
        # We want smooth ramp from small positive to 1 as we approach the goal
        # Start at 1/(2*length) to avoid exactly 0 at first frame
        # This ensures all frames in window have positive threat
        start_val = 1.0 / (2 * actual_window_length) if actual_window_length > 0 else 0
        positions = np.linspace(start_val, 1, actual_window_length, endpoint=False)
        
        # Apply smoothing function
        if smoothing == 'constant':
            # Step function: constant 1.0 within window
            ramp_values = np.ones_like(positions)
            
        elif smoothing == 'linear':
            # Linear ramp: threat increases linearly
            ramp_values = positions
            
        elif smoothing == 'quadratic':
            # Quadratic ramp: smooth acceleration
            ramp_values = positions ** 2
            
        elif smoothing == 'exponential':
            # Exponential: sharp rise near goal
            k = 4  # Sharpness parameter
            ramp_values = 1 - np.exp(-k * positions)
            
        else:
            raise ValueError(f"Unknown smoothing type: {smoothing}")
        
        # Assign threat values to the valid window
        threat[window_start:window_end] = ramp_values
    
    return threat


def plot_threat_distribution(threat, goal_frames, title, save_path=None):
    """
    Visualize threat distribution with goal markers.
    
    Args:
        threat: Threat array
        goal_frames: List of goal frame indices
        title: Plot title
        save_path: Optional path to save plot
    """
    plt.figure(figsize=(14, 5))
    
    # Plot threat curve
    plt.plot(threat, color='#e74c3c', linewidth=1.5, alpha=0.9, label='Threat Level')
    
    # Mark goals
    if len(goal_frames) > 0:
        plt.scatter(goal_frames, [1.0] * len(goal_frames), 
                   color='#2c3e50', marker='x', s=150, 
                   linewidths=3, zorder=10, label='Goal Event')
    
    plt.xlabel('Frame', fontsize=11)
    plt.ylabel('Threat Level (Probability)', fontsize=11)
    plt.title(title, fontsize=13, fontweight='bold')
    plt.ylim(-0.05, 1.1)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [SAVED PLOT] {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_smoothing_comparison(goal_frames, total_frames, n_seconds, fps):
    """
    Create side-by-side comparison of all smoothing methods.
    """
    smoothing_methods = ['constant', 'linear', 'quadratic', 'exponential']
    colors = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, (method, color) in enumerate(zip(smoothing_methods, colors)):
        threat = create_threat_distribution(total_frames, goal_frames, 
                                           n_seconds, fps, smoothing=method)
        
        ax = axes[idx]
        ax.plot(threat, color=color, linewidth=1.5, alpha=0.8)
        ax.scatter(goal_frames, [1.0] * len(goal_frames), 
                  color='black', marker='x', s=100, linewidths=2, zorder=10)
        
        ax.set_xlabel('Frame', fontsize=11)
        ax.set_ylabel('Threat Level', fontsize=11)
        ax.set_title(f'Smoothing: {method.capitalize()}', fontsize=12, fontweight='bold')
        ax.set_ylim(-0.05, 1.1)
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        non_zero = threat[threat > 0]
        if len(non_zero) > 0:
            stats_text = f'Non-zero frames: {len(non_zero)}\nMean threat: {non_zero.mean():.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Threat Distribution Comparison (n={n_seconds}s window)', 
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    return fig


def analyze_distribution_stats(threat, goal_frames, n_seconds, fps):
    """
    Print statistical analysis of the threat distribution.
    """
    window_frames = int(n_seconds * fps)
    
    # Basic stats
    non_zero_frames = np.sum(threat > 0)
    total_threat_frames = len(goal_frames) * window_frames
    percent_positive = (non_zero_frames / len(threat)) * 100
    
    print(f"\n  Distribution Statistics:")
    print(f"  ├─ Total frames: {len(threat)}")
    print(f"  ├─ Goals: {len(goal_frames)}")
    print(f"  ├─ Frames with threat > 0: {non_zero_frames} ({percent_positive:.1f}%)")
    print(f"  ├─ Expected positive frames: {total_threat_frames}")
    print(f"  ├─ Mean threat (all): {threat.mean():.4f}")
    print(f"  ├─ Mean threat (non-zero): {threat[threat > 0].mean():.4f}" if non_zero_frames > 0 else "  ├─ Mean threat (non-zero): N/A")
    print(f"  ├─ Max threat: {threat.max():.4f}")
    print(f"  └─ Class imbalance ratio: 1:{(len(threat) - non_zero_frames) / max(non_zero_frames, 1):.1f}")


# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================

def process_all_replays():
    """
    Process all replay folders and create threat distributions.
    Uses actual video properties (FPS and frame count) from each .mp4 file.
    """
    print("=" * 80)
    print("ROCKET LEAGUE THREAT DISTRIBUTION GENERATOR")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  ├─ Time window (n): {N_SECONDS} seconds")
    print(f"  ├─ Smoothing method: {SMOOTHING}")
    print(f"  ├─ Expected FPS: {EXPECTED_FPS}")
    print(f"  └─ Frame count: DYNAMIC (from video files)")
    print("=" * 80)
    
    processed_count = 0
    skipped_count = 0
    
    for folder in sorted(os.listdir(REPLAY_DATA_DIR)):
        folder_path = os.path.join(REPLAY_DATA_DIR, folder)
        
        if not os.path.isdir(folder_path):
            continue
        
        csv_path = os.path.join(folder_path, "goals.csv")
        
        if not os.path.exists(csv_path):
            print(f"\n[SKIP] {folder} - No goals.csv found")
            skipped_count += 1
            continue
        
        # Find video file
        video_path = find_video_file(folder_path)
        if video_path is None:
            print(f"\n[SKIP] {folder} - No .mp4 video file found")
            skipped_count += 1
            continue
        
        print(f"\n[PROCESSING] {folder}")
        print(f"  ├─ Video: {os.path.basename(video_path)}")
        
        # Get video properties
        fps, total_frames = get_video_properties(video_path)
        
        if fps is None or total_frames is None:
            print(f"  └─ [SKIP] Could not read video properties")
            skipped_count += 1
            continue
        
        duration = total_frames / fps
        print(f"  ├─ FPS: {fps:.2f}")
        print(f"  ├─ Total frames: {total_frames}")
        print(f"  ├─ Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
        # Load goals
        df = pd.read_csv(csv_path)
        
        if df.empty:
            print(f"  ├─ Goals: 0 (empty CSV)")
            goal_frames = []
        else:
            goal_frames = sorted(df["Frame"].tolist())
            print(f"  ├─ Goals: {len(goal_frames)}")
            
            # Validate goal frames are within video length
            invalid_goals = [g for g in goal_frames if g >= total_frames]
            if invalid_goals:
                print(f"  ├─ [WARNING] {len(invalid_goals)} goal(s) beyond video length:")
                for g in invalid_goals[:3]:  # Show first 3
                    print(f"  │   - Frame {g} (video ends at {total_frames})")
                # Filter out invalid goals
                goal_frames = [g for g in goal_frames if g < total_frames]
                print(f"  ├─ Valid goals after filtering: {len(goal_frames)}")
        
        # Create threat distribution
        threat = create_threat_distribution(
            total_frames=total_frames,
            goal_frames=goal_frames,
            n_seconds=N_SECONDS,
            fps=fps,
            smoothing=SMOOTHING
        )
        
        # Analyze statistics
        if len(goal_frames) > 0:
            analyze_distribution_stats(threat, goal_frames, N_SECONDS, fps)
        
        # Save distribution
        npy_path = os.path.join(folder_path, "distribution.npy")
        np.save(npy_path, threat)
        print(f"  ├─ [SAVED] distribution.npy")
        
        # Save metadata
        metadata = {
            'fps': fps,
            'total_frames': total_frames,
            'duration_seconds': duration,
            'n_seconds': N_SECONDS,
            'smoothing': SMOOTHING,
            'num_goals': len(goal_frames),
            'video_file': os.path.basename(video_path)
        }
        metadata_path = os.path.join(folder_path, "distribution_metadata.txt")
        with open(metadata_path, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        print(f"  ├─ [SAVED] distribution_metadata.txt")
        
        # Plot individual replay
        if PLOT_EACH:
            plot_path = os.path.join(folder_path, "distribution_plot.png") if SAVE_PLOTS else None
            plot_threat_distribution(
                threat, 
                goal_frames, 
                title=f"{folder} - Threat Distribution ({SMOOTHING}, n={N_SECONDS}s)\n"
                      f"FPS: {fps:.1f}, Duration: {duration/60:.1f}min",
                save_path=plot_path
            )
        
        print(f"  └─ [DONE]")
        processed_count += 1
        
        # Create smoothing comparison for first replay with goals (as example)
        if processed_count == 1 and PLOT_SMOOTHING_COMPARISON and len(goal_frames) > 0:
            print(f"\n  [CREATING] Smoothing comparison plot...")
            fig = plot_smoothing_comparison(goal_frames, total_frames, N_SECONDS, fps)
            comparison_path = os.path.join(REPLAY_DATA_DIR, "smoothing_comparison.png")
            fig.savefig(comparison_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  [SAVED] {comparison_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"  ├─ Replays processed: {processed_count}")
    print(f"  ├─ Replays skipped: {skipped_count}")
    print(f"  └─ Method: {SMOOTHING} with {N_SECONDS}s window")
    print("=" * 80)
    
    print("\n💡 RECOMMENDATIONS FOR MODEL TRAINING:")
    print("  1. Use Focal Loss to handle class imbalance")
    print("  2. Sample frames intelligently (oversample near goals)")
    print("  3. Evaluate with Average Precision, not just MSE/BCE")
    print("  4. Consider temporal sampling: every 0.5s instead of every frame")
    print(f"  5. Window size varies by FPS - at 30fps, {N_SECONDS}s = {int(N_SECONDS * 30)} frames")
    print("\n🔧 TUNING SUGGESTIONS:")
    print(f"  - If model predicts too early: decrease n_seconds (current: {N_SECONDS}s)")
    print(f"  - If model predicts too late: increase n_seconds")
    print(f"  - If training unstable: try 'quadratic' smoothing (current: {SMOOTHING})")
    print(f"  - If need sharper predictions: try 'exponential' smoothing")
    print("\n📊 VIDEO PROPERTIES:")
    print(f"  - FPS is now read dynamically from each video")
    print(f"  - Frame counts vary by game length (overtime, etc.)")
    print(f"  - Distribution arrays are video-specific sizes")


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate threat distributions for Rocket League goal prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default settings (quadratic, 10s window, 30fps expected)
  python create_threat_distribution.py
  
  # Try different smoothing
  python create_threat_distribution.py --smoothing exponential
  
  # Adjust time window
  python create_threat_distribution.py --n-seconds 5.0
  
  # Disable plotting
  python create_threat_distribution.py --no-plot
  
  # Expect different FPS (for validation warning)
  python create_threat_distribution.py --expected-fps 60

Note: FPS and total frames are now read dynamically from each video file.
The --expected-fps flag is only used to warn if video FPS differs significantly.
        """
    )
    
    parser.add_argument('--smoothing', type=str, default=SMOOTHING,
                       choices=['constant', 'linear', 'quadratic', 'exponential'],
                       help='Smoothing function type (default: quadratic)')
    
    parser.add_argument('--n-seconds', type=float, default=N_SECONDS,
                       help='Time window in seconds (default: 10.0)')
    
    parser.add_argument('--expected-fps', type=int, default=EXPECTED_FPS,
                       help='Expected FPS for validation (default: 30, actual FPS read from video)')
    
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable individual plotting')
    
    parser.add_argument('--save-plots', action='store_true',
                       help='Save plots to files instead of displaying')
    
    parser.add_argument('--no-comparison', action='store_true',
                       help='Disable smoothing comparison plot')
    
    args = parser.parse_args()
    
    # Override configuration with command-line arguments
    SMOOTHING = args.smoothing
    N_SECONDS = args.n_seconds
    EXPECTED_FPS = args.expected_fps
    PLOT_EACH = not args.no_plot
    SAVE_PLOTS = args.save_plots
    PLOT_SMOOTHING_COMPARISON = not args.no_comparison
    
    # Run processing
    process_all_replays()