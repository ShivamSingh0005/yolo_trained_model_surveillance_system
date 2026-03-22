"""
Analyze dataset distribution and identify class imbalance
"""

import os
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def count_class_instances(labels_dir):
    """Count instances of each class in label files"""
    class_counts = Counter()
    file_counts = Counter()
    
    labels_path = Path(labels_dir)
    if not labels_path.exists():
        print(f"Directory not found: {labels_dir}")
        return class_counts, file_counts
    
    label_files = list(labels_path.glob('*.txt'))
    
    for label_file in label_files:
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    for line in lines:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            class_counts[class_id] += 1
                            file_counts[class_id] += 1
        except (FileNotFoundError, OSError) as e:
            # Skip files with path issues (Windows long path)
            continue
    
    return class_counts, len(label_files)

def analyze_dataset():
    """Analyze the complete dataset"""
    
    class_names = {
        0: 'Animal',
        1: 'Forest', 
        2: 'Militant',
        3: 'UAV-Drone',
        4: 'Wildfire'
    }
    
    print("=" * 80)
    print("DATASET ANALYSIS")
    print("=" * 80)
    
    # Analyze each split
    splits = ['train', 'test', 'valid']
    all_stats = {}
    
    for split in splits:
        labels_dir = f'{split}/labels'
        if os.path.exists(labels_dir):
            class_counts, total_files = count_class_instances(labels_dir)
            all_stats[split] = {
                'class_counts': class_counts,
                'total_files': total_files
            }
            
            print(f"\n{split.upper()} SET:")
            print(f"  Total label files: {total_files}")
            print(f"  Class distribution:")
            
            total_instances = sum(class_counts.values())
            for class_id in sorted(class_counts.keys()):
                count = class_counts[class_id]
                percentage = (count / total_instances * 100) if total_instances > 0 else 0
                print(f"    {class_names.get(class_id, f'Class {class_id}')}: {count} instances ({percentage:.1f}%)")
    
    # Visualize distribution
    if all_stats:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, split in enumerate(splits):
            if split in all_stats:
                class_counts = all_stats[split]['class_counts']
                
                # Prepare data
                classes = [class_names.get(i, f'Class {i}') for i in sorted(class_counts.keys())]
                counts = [class_counts[i] for i in sorted(class_counts.keys())]
                
                # Create bar plot
                colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#e67e22']
                bars = axes[idx].bar(classes, counts, color=colors[:len(classes)])
                
                # Highlight Wildfire
                if 4 in class_counts:
                    wildfire_idx = sorted(class_counts.keys()).index(4)
                    bars[wildfire_idx].set_color('#c0392b')
                    bars[wildfire_idx].set_edgecolor('red')
                    bars[wildfire_idx].set_linewidth(3)
                
                axes[idx].set_title(f'{split.upper()} Set Distribution', 
                                   fontsize=14, fontweight='bold')
                axes[idx].set_xlabel('Class', fontsize=12)
                axes[idx].set_ylabel('Number of Instances', fontsize=12)
                axes[idx].tick_params(axis='x', rotation=45)
                axes[idx].grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                                  f'{int(height)}',
                                  ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('dataset_distribution.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved distribution plot to: dataset_distribution.png")
        plt.close()
    
    # Calculate imbalance ratio
    print("\n" + "=" * 80)
    print("CLASS IMBALANCE ANALYSIS")
    print("=" * 80)
    
    if 'train' in all_stats:
        train_counts = all_stats['train']['class_counts']
        if train_counts:
            max_count = max(train_counts.values())
            min_count = min(train_counts.values())
            
            print(f"\nImbalance Ratio: {max_count / min_count:.2f}:1")
            print(f"Most common class: {max_count} instances")
            print(f"Least common class: {min_count} instances")
            
            print("\nRecommended augmentation multipliers:")
            for class_id in sorted(train_counts.keys()):
                count = train_counts[class_id]
                multiplier = max_count / count
                status = "⚠️ NEEDS AUGMENTATION" if multiplier > 2 else "✓ Balanced"
                print(f"  {class_names.get(class_id, f'Class {class_id}')}: {multiplier:.1f}x {status}")
    
    print("\n" + "=" * 80)
    
    return all_stats

if __name__ == "__main__":
    stats = analyze_dataset()
