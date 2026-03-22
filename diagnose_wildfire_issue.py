"""
Diagnose why Wildfire detection is failing
Analyze the wildfire samples to understand the issue
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
from collections import Counter

def analyze_wildfire_samples():
    """Analyze wildfire samples to identify issues"""
    
    print("=" * 80)
    print("WILDFIRE DETECTION ISSUE DIAGNOSIS")
    print("=" * 80)
    
    # Find wildfire samples in each split
    splits = ['train', 'test', 'valid']
    wildfire_class_id = 4
    
    analysis = {}
    
    for split in splits:
        labels_dir = Path(f'{split}/labels')
        images_dir = Path(f'{split}/images')
        
        if not labels_dir.exists():
            continue
        
        wildfire_files = []
        bbox_sizes = []
        bbox_counts = []
        
        print(f"\n{split.upper()} SET:")
        print("-" * 80)
        
        label_files = list(labels_dir.glob('*.txt'))
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    
                wildfire_boxes = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5 and int(parts[0]) == wildfire_class_id:
                        # Parse bbox
                        x_center, y_center, width, height = map(float, parts[1:5])
                        wildfire_boxes.append((width, height, width * height))
                
                if wildfire_boxes:
                    wildfire_files.append(label_file.stem)
                    bbox_counts.append(len(wildfire_boxes))
                    
                    for w, h, area in wildfire_boxes:
                        bbox_sizes.append({
                            'width': w,
                            'height': h,
                            'area': area
                        })
            except Exception as e:
                continue
        
        print(f"  Wildfire samples: {len(wildfire_files)}")
        print(f"  Total wildfire instances: {len(bbox_sizes)}")
        
        if bbox_sizes:
            areas = [b['area'] for b in bbox_sizes]
            widths = [b['width'] for b in bbox_sizes]
            heights = [b['height'] for b in bbox_sizes]
            
            print(f"\n  Bounding Box Statistics:")
            print(f"    Average area: {np.mean(areas):.4f} (normalized)")
            print(f"    Min area: {np.min(areas):.4f}")
            print(f"    Max area: {np.max(areas):.4f}")
            print(f"    Median area: {np.median(areas):.4f}")
            
            print(f"\n    Average width: {np.mean(widths):.4f}")
            print(f"    Average height: {np.mean(heights):.4f}")
            
            # Categorize by size
            tiny = sum(1 for a in areas if a < 0.01)
            small = sum(1 for a in areas if 0.01 <= a < 0.05)
            medium = sum(1 for a in areas if 0.05 <= a < 0.2)
            large = sum(1 for a in areas if a >= 0.2)
            
            print(f"\n  Size Distribution:")
            print(f"    Tiny (<1% of image): {tiny} ({tiny/len(areas)*100:.1f}%)")
            print(f"    Small (1-5%): {small} ({small/len(areas)*100:.1f}%)")
            print(f"    Medium (5-20%): {medium} ({medium/len(areas)*100:.1f}%)")
            print(f"    Large (>20%): {large} ({large/len(areas)*100:.1f}%)")
            
            if tiny > len(areas) * 0.5:
                print(f"\n  ⚠️  WARNING: {tiny/len(areas)*100:.1f}% of wildfire boxes are TINY!")
                print(f"     This makes detection very difficult.")
            
            # Instances per image
            print(f"\n  Instances per image:")
            print(f"    Average: {np.mean(bbox_counts):.2f}")
            print(f"    Max: {np.max(bbox_counts)}")
            
            analysis[split] = {
                'count': len(wildfire_files),
                'instances': len(bbox_sizes),
                'avg_area': float(np.mean(areas)),
                'median_area': float(np.median(areas)),
                'tiny_percent': tiny/len(areas)*100,
                'small_percent': small/len(areas)*100,
                'medium_percent': medium/len(areas)*100,
                'large_percent': large/len(areas)*100,
            }
    
    # Generate visualization
    if analysis:
        generate_diagnosis_plot(analysis)
    
    # Provide recommendations
    print("\n" + "=" * 80)
    print("DIAGNOSIS & RECOMMENDATIONS")
    print("=" * 80)
    
    if 'train' in analysis:
        train_tiny = analysis['train']['tiny_percent']
        train_avg_area = analysis['train']['avg_area']
        
        print("\n🔍 ROOT CAUSE ANALYSIS:")
        
        if train_tiny > 50:
            print(f"  ❌ CRITICAL: {train_tiny:.1f}% of wildfire boxes are tiny (<1% of image)")
            print(f"     Small objects are extremely hard for YOLO to detect")
            
        if train_avg_area < 0.05:
            print(f"  ❌ PROBLEM: Average wildfire size is only {train_avg_area*100:.1f}% of image")
            print(f"     YOLO struggles with small objects")
        
        print("\n💡 RECOMMENDED SOLUTIONS:")
        print("\n1. USE LARGER MODEL (CRITICAL):")
        print("   - Switch from YOLOv8n to YOLOv8m or YOLOv8l")
        print("   - Larger models have better small object detection")
        print("   - Command: model = YOLO('yolov8m.pt')")
        
        print("\n2. INCREASE IMAGE SIZE:")
        print("   - Train with imgsz=1280 instead of 640")
        print("   - Larger images preserve small object details")
        print("   - Command: model.train(imgsz=1280, ...)")
        
        print("\n3. ADJUST ANCHOR SIZES:")
        print("   - YOLO's default anchors may not match wildfire sizes")
        print("   - Use smaller anchor boxes for tiny objects")
        
        print("\n4. FOCUS ON SMALL OBJECT AUGMENTATION:")
        print("   - Use copy-paste augmentation")
        print("   - Zoom in on wildfire regions")
        print("   - Crop and resize to make wildfires larger")
        
        print("\n5. COLLECT BETTER DATA:")
        print("   - Get images where wildfire occupies >5% of frame")
        print("   - Close-up shots of fires")
        print("   - Avoid distant/tiny fire instances")
        
        print("\n6. TWO-STAGE DETECTION:")
        print("   - First detect general fire regions")
        print("   - Then zoom in and detect precisely")
        
        print("\n7. USE ATTENTION MECHANISMS:")
        print("   - YOLOv8 with attention layers")
        print("   - Helps focus on small objects")
    
    print("\n" + "=" * 80)
    
    # Save analysis
    with open('wildfire_diagnosis.json', 'w') as f:
        json.dump(analysis, f, indent=4)
    
    print(f"\n✓ Diagnosis saved to: wildfire_diagnosis.json")
    
    return analysis


def generate_diagnosis_plot(analysis):
    """Generate visualization of the diagnosis"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Size distribution
    splits = list(analysis.keys())
    categories = ['Tiny\n(<1%)', 'Small\n(1-5%)', 'Medium\n(5-20%)', 'Large\n(>20%)']
    
    x = np.arange(len(categories))
    width = 0.25
    
    for idx, split in enumerate(splits):
        values = [
            analysis[split]['tiny_percent'],
            analysis[split]['small_percent'],
            analysis[split]['medium_percent'],
            analysis[split]['large_percent']
        ]
        
        offset = (idx - len(splits)/2 + 0.5) * width
        axes[0].bar(x + offset, values, width, label=split.capitalize())
    
    axes[0].set_xlabel('Object Size Category', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Wildfire Object Size Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% threshold')
    
    # Plot 2: Average area comparison
    split_names = [s.capitalize() for s in splits]
    avg_areas = [analysis[s]['avg_area'] * 100 for s in splits]
    
    bars = axes[1].bar(split_names, avg_areas, color=['#3498db', '#e74c3c', '#2ecc71'][:len(splits)])
    axes[1].set_xlabel('Dataset Split', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Average Area (% of image)', fontsize=12, fontweight='bold')
    axes[1].set_title('Average Wildfire Object Size', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].axhline(y=5, color='g', linestyle='--', alpha=0.5, label='5% (good threshold)')
    axes[1].axhline(y=1, color='r', linestyle='--', alpha=0.5, label='1% (too small)')
    axes[1].legend()
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Wildfire Detection Issue Diagnosis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig('wildfire_diagnosis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Diagnosis plot saved to: wildfire_diagnosis.png")
    plt.close()


if __name__ == "__main__":
    analysis = analyze_wildfire_samples()
