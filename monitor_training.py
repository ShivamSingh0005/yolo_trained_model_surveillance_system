"""
Training Monitor - Check training progress in real-time
"""

import os
import time
from pathlib import Path
import pandas as pd

def monitor_training():
    """Monitor training progress"""
    results_csv = Path('runs/surveillance/train/results.csv')
    
    print("=" * 80)
    print("TRAINING MONITOR - Real-time Progress")
    print("=" * 80)
    print("\nWaiting for training to start...")
    
    # Wait for results file to be created
    while not results_csv.exists():
        print(".", end="", flush=True)
        time.sleep(5)
    
    print("\n\n✓ Training started! Monitoring progress...\n")
    
    last_epoch = 0
    
    while True:
        try:
            if results_csv.exists():
                df = pd.read_csv(results_csv)
                df.columns = df.columns.str.strip()
                
                current_epoch = len(df)
                
                if current_epoch > last_epoch:
                    last_epoch = current_epoch
                    
                    # Get latest metrics
                    latest = df.iloc[-1]
                    
                    print(f"\n{'='*80}")
                    print(f"EPOCH {current_epoch}/100")
                    print(f"{'='*80}")
                    
                    # Training losses
                    if 'train/box_loss' in df.columns:
                        print(f"Training Losses:")
                        print(f"  Box Loss:   {latest['train/box_loss']:.4f}")
                        print(f"  Class Loss: {latest['train/cls_loss']:.4f}")
                        print(f"  DFL Loss:   {latest['train/dfl_loss']:.4f}")
                    
                    # Validation losses
                    if 'val/box_loss' in df.columns:
                        print(f"\nValidation Losses:")
                        print(f"  Box Loss:   {latest['val/box_loss']:.4f}")
                        print(f"  Class Loss: {latest['val/cls_loss']:.4f}")
                        print(f"  DFL Loss:   {latest['val/dfl_loss']:.4f}")
                    
                    # Metrics
                    if 'metrics/precision(B)' in df.columns:
                        print(f"\nPerformance Metrics:")
                        print(f"  Precision:  {latest['metrics/precision(B)']:.4f}")
                        print(f"  Recall:     {latest['metrics/recall(B)']:.4f}")
                        print(f"  mAP@0.5:    {latest['metrics/mAP50(B)']:.4f}")
                        print(f"  mAP@0.5:0.95: {latest['metrics/mAP50-95(B)']:.4f}")
                    
                    # Progress
                    progress = (current_epoch / 100) * 100
                    bar_length = 50
                    filled = int(bar_length * current_epoch / 100)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    print(f"\nProgress: [{bar}] {progress:.1f}%")
                    
                    if current_epoch >= 100:
                        print("\n" + "="*80)
                        print("✓ TRAINING COMPLETED!")
                        print("="*80)
                        break
            
            time.sleep(10)  # Check every 10 seconds
            
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user.")
            break
        except Exception as e:
            print(f"\nError reading results: {e}")
            time.sleep(10)

if __name__ == "__main__":
    monitor_training()
