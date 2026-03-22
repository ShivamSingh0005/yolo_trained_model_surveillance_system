"""
Monitor Wildfire-focused training progress
Tracks Wildfire class performance specifically
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json

def monitor_training():
    """Monitor training progress and display Wildfire metrics"""
    
    results_file = Path('runs/detect/wildfire_focused_v2/results.csv')
    
    print("=" * 80)
    print("WILDFIRE TRAINING MONITOR")
    print("=" * 80)
    print("\nWaiting for training to start...")
    
    # Wait for results file
    while not results_file.exists():
        time.sleep(5)
    
    print(f"✓ Training started! Monitoring: {results_file}")
    print("\nPress Ctrl+C to stop monitoring\n")
    
    last_epoch = 0
    
    try:
        while True:
            if results_file.exists():
                # Read results
                df = pd.read_csv(results_file)
                df.columns = df.columns.str.strip()
                
                if len(df) > last_epoch:
                    # Get latest epoch
                    latest = df.iloc[-1]
                    epoch = int(latest['epoch']) + 1
                    
                    # Overall metrics
                    map50 = latest['metrics/mAP50(B)']
                    map50_95 = latest['metrics/mAP50-95(B)']
                    
                    # Display progress
                    print(f"\n{'='*80}")
                    print(f"EPOCH {epoch}/200")
                    print(f"{'='*80}")
                    print(f"Overall mAP50: {map50:.4f} ({map50*100:.1f}%)")
                    print(f"Overall mAP50-95: {map50_95:.4f} ({map50_95*100:.1f}%)")
                    
                    # Loss values
                    if 'train/box_loss' in df.columns:
                        box_loss = latest['train/box_loss']
                        cls_loss = latest['train/cls_loss']
                        dfl_loss = latest['train/dfl_loss']
                        print(f"\nLosses: Box={box_loss:.3f}, Cls={cls_loss:.3f}, DFL={dfl_loss:.3f}")
                    
                    # Check if we have per-class metrics
                    # Note: YOLO doesn't save per-class metrics in CSV by default
                    # We'll estimate based on overall performance
                    
                    print(f"\n{'='*80}")
                    
                    # Estimate progress
                    if map50 > 0.7:
                        print("🎯 Good progress! Overall mAP50 > 70%")
                    elif map50 > 0.6:
                        print("✓ Training progressing well")
                    else:
                        print("⏳ Early training phase...")
                    
                    # Check for improvement
                    if len(df) > 10:
                        improvement = map50 - df.iloc[-10]['metrics/mAP50(B)']
                        if improvement > 0:
                            print(f"📈 Improved by {improvement*100:.1f}% in last 10 epochs")
                        elif improvement < -0.05:
                            print(f"⚠️  Performance decreased by {abs(improvement)*100:.1f}%")
                    
                    last_epoch = len(df)
                    
                    # Generate progress plot every 10 epochs
                    if epoch % 10 == 0:
                        generate_progress_plot(df)
            
            time.sleep(30)  # Check every 30 seconds
            
    except KeyboardInterrupt:
        print("\n\n✓ Monitoring stopped")
        if results_file.exists():
            df = pd.read_csv(results_file)
            generate_progress_plot(df)
            print(f"\n✓ Final progress plot saved")


def generate_progress_plot(df):
    """Generate training progress visualization"""
    
    df.columns = df.columns.str.strip()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = df['epoch'] + 1
    
    # mAP metrics
    axes[0, 0].plot(epochs, df['metrics/mAP50(B)'], 'b-', linewidth=2, label='mAP50')
    axes[0, 0].plot(epochs, df['metrics/mAP50-95(B)'], 'r-', linewidth=2, label='mAP50-95')
    axes[0, 0].axhline(y=0.8, color='g', linestyle='--', label='Target (80%)')
    axes[0, 0].set_xlabel('Epoch', fontweight='bold')
    axes[0, 0].set_ylabel('mAP', fontweight='bold')
    axes[0, 0].set_title('mAP Progress', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Losses
    if 'train/box_loss' in df.columns:
        axes[0, 1].plot(epochs, df['train/box_loss'], 'b-', linewidth=2, label='Box Loss')
        axes[0, 1].plot(epochs, df['train/cls_loss'], 'r-', linewidth=2, label='Cls Loss')
        axes[0, 1].plot(epochs, df['train/dfl_loss'], 'g-', linewidth=2, label='DFL Loss')
        axes[0, 1].set_xlabel('Epoch', fontweight='bold')
        axes[0, 1].set_ylabel('Loss', fontweight='bold')
        axes[0, 1].set_title('Training Losses', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Precision & Recall
    if 'metrics/precision(B)' in df.columns:
        axes[1, 0].plot(epochs, df['metrics/precision(B)'], 'b-', linewidth=2, label='Precision')
        axes[1, 0].plot(epochs, df['metrics/recall(B)'], 'r-', linewidth=2, label='Recall')
        axes[1, 0].set_xlabel('Epoch', fontweight='bold')
        axes[1, 0].set_ylabel('Score', fontweight='bold')
        axes[1, 0].set_title('Precision & Recall', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate
    if 'lr/pg0' in df.columns:
        axes[1, 1].plot(epochs, df['lr/pg0'], 'b-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch', fontweight='bold')
        axes[1, 1].set_ylabel('Learning Rate', fontweight='bold')
        axes[1, 1].set_title('Learning Rate Schedule', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Wildfire-Focused Training Progress', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig('wildfire_training_progress.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    monitor_training()
