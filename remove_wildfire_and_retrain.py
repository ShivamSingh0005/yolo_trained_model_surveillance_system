"""
Remove Wildfire class from dataset and retrain with 4 classes
Classes: Animal, Forest, Militant, UAV-Drone
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm
import yaml

def remove_wildfire_class():
    """Remove wildfire class (class 4) from all label files"""
    
    print("=" * 80)
    print("REMOVING WILDFIRE CLASS FROM DATASET")
    print("=" * 80)
    
    wildfire_class_id = 4
    splits = ['train', 'test', 'valid']
    
    stats = {}
    
    for split in splits:
        labels_dir = Path(f'{split}/labels')
        
        if not labels_dir.exists():
            continue
        
        print(f"\nProcessing {split.upper()} set...")
        
        label_files = list(labels_dir.glob('*.txt'))
        files_modified = 0
        instances_removed = 0
        
        for label_file in tqdm(label_files):
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                # Filter out wildfire instances
                new_lines = []
                removed_count = 0
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        if class_id != wildfire_class_id:
                            new_lines.append(line)
                        else:
                            removed_count += 1
                
                # Write back if changed
                if removed_count > 0:
                    with open(label_file, 'w') as f:
                        f.writelines(new_lines)
                    files_modified += 1
                    instances_removed += removed_count
                    
            except Exception as e:
                continue
        
        stats[split] = {
            'files_modified': files_modified,
            'instances_removed': instances_removed
        }
        
        print(f"  Files modified: {files_modified}")
        print(f"  Wildfire instances removed: {instances_removed}")
    
    print("\n" + "=" * 80)
    print("WILDFIRE CLASS REMOVAL COMPLETE")
    print("=" * 80)
    
    total_removed = sum(s['instances_removed'] for s in stats.values())
    print(f"\nTotal wildfire instances removed: {total_removed}")
    
    return stats


def create_4class_yaml():
    """Create new data.yaml for 4 classes"""
    
    yaml_content = """# 4-Class Surveillance Dataset (Wildfire removed)
path: .
train: train/images
val: valid/images
test: test/images

# Classes
names:
  0: Animal
  1: Forest
  2: Militant
  3: UAV-Drone

# Number of classes
nc: 4
"""
    
    with open('data_4class.yaml', 'w') as f:
        f.write(yaml_content)
    
    print("\n✓ Created data_4class.yaml")


def train_4class_model():
    """Train model with 4 classes"""
    
    from ultralytics import YOLO
    import torch
    
    print("\n" + "=" * 80)
    print("TRAINING 4-CLASS MODEL")
    print("=" * 80)
    print("\nClasses: Animal, Forest, Militant, UAV-Drone")
    print("-" * 80)
    
    # Load YOLOv8n
    model = YOLO('yolov8n.pt')
    
    # Train
    results = model.train(
        data='data_4class.yaml',
        epochs=100,
        batch=16,
        imgsz=640,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        
        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # Training settings
        patience=50,
        save=True,
        save_period=10,
        cache=False,
        device=0 if torch.cuda.is_available() else 'cpu',
        workers=8,
        project='runs/detect',
        name='4class_surveillance',
        exist_ok=True,
        pretrained=True,
        verbose=True,
        val=True,
        plots=True,
        cos_lr=True,
        close_mosaic=10,
    )
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    
    return results


def evaluate_4class_model():
    """Evaluate the 4-class model"""
    
    from ultralytics import YOLO
    import json
    
    print("\n" + "=" * 80)
    print("EVALUATING 4-CLASS MODEL")
    print("=" * 80)
    
    model_path = Path('runs/detect/4class_surveillance/weights/best.pt')
    
    if not model_path.exists():
        print(f"\n⚠️  Model not found: {model_path}")
        return None
    
    model = YOLO(str(model_path))
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    metrics = model.val(
        data='data_4class.yaml',
        split='test',
        save_json=True,
        plots=True,
        project='runs/detect',
        name='4class_eval'
    )
    
    # Extract metrics
    results_dict = {
        'model': 'YOLOv8n',
        'classes': 4,
        'class_names': ['Animal', 'Forest', 'Militant', 'UAV-Drone'],
        'training_epochs': 100,
        'mAP50': float(metrics.box.map50),
        'mAP50-95': float(metrics.box.map),
        'Precision': float(metrics.box.mp),
        'Recall': float(metrics.box.mr),
        'F1-Score': 2 * (float(metrics.box.mp) * float(metrics.box.mr)) / (float(metrics.box.mp) + float(metrics.box.mr) + 1e-6)
    }
    
    # Per-class metrics
    class_names = ['Animal', 'Forest', 'Militant', 'UAV-Drone']
    results_dict['per_class'] = {}
    
    for idx, class_name in enumerate(class_names):
        if idx < len(metrics.box.ap50):
            p = float(metrics.box.p[idx]) if hasattr(metrics.box, 'p') and idx < len(metrics.box.p) else 0.0
            r = float(metrics.box.r[idx]) if hasattr(metrics.box, 'r') and idx < len(metrics.box.r) else 0.0
            
            results_dict['per_class'][class_name] = {
                'AP50': float(metrics.box.ap50[idx]),
                'AP': float(metrics.box.ap[idx]) if hasattr(metrics.box, 'ap') else 0.0,
                'Precision': p,
                'Recall': r,
                'F1-Score': 2 * (p * r) / (p + r + 1e-6)
            }
    
    # Save results
    results_dir = Path('4class_results')
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / '4class_metrics.json', 'w') as f:
        json.dump(results_dict, f, indent=4)
    
    # Copy best model
    shutil.copy2(model_path, results_dir / 'best_4class_model.pt')
    
    # Display results
    print("\n" + "=" * 80)
    print("4-CLASS MODEL PERFORMANCE")
    print("=" * 80)
    
    print("\nOVERALL METRICS:")
    print("-" * 80)
    print(f"  mAP50: {results_dict['mAP50']*100:.2f}%")
    print(f"  mAP50-95: {results_dict['mAP50-95']*100:.2f}%")
    print(f"  Precision: {results_dict['Precision']*100:.2f}%")
    print(f"  Recall: {results_dict['Recall']*100:.2f}%")
    print(f"  F1-Score: {results_dict['F1-Score']*100:.2f}%")
    
    print("\nPER-CLASS PERFORMANCE:")
    print("-" * 80)
    print(f"{'Class':<15} {'AP50':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 80)
    
    for class_name in class_names:
        if class_name in results_dict['per_class']:
            cls = results_dict['per_class'][class_name]
            print(f"{class_name:<15} {cls['AP50']*100:>6.1f}%{'':<5} {cls['Precision']*100:>6.1f}%{'':<5} {cls['Recall']*100:>6.1f}%{'':<5} {cls['F1-Score']*100:>6.1f}%")
    
    print("\n" + "=" * 80)
    
    # Check if all classes >= 80%
    all_above_80 = all(results_dict['per_class'][c]['AP50'] >= 0.80 for c in class_names)
    
    if all_above_80:
        print("🎉 SUCCESS! All classes achieved >= 80% AP50")
    elif results_dict['mAP50'] >= 0.80:
        print("✓ Overall mAP50 >= 80%")
    
    print(f"\n✓ Results saved to: {results_dir}/")
    print(f"✓ Model saved to: {results_dir}/best_4class_model.pt")
    print("=" * 80)
    
    return results_dict


def main():
    """Main execution"""
    
    print("\n" + "=" * 80)
    print("4-CLASS SURVEILLANCE SYSTEM")
    print("Removing Wildfire class and retraining")
    print("=" * 80)
    
    # Step 1: Remove wildfire class
    stats = remove_wildfire_class()
    
    # Step 2: Create new yaml
    create_4class_yaml()
    
    # Step 3: Train
    print("\nStarting training...")
    train_results = train_4class_model()
    
    # Step 4: Evaluate
    eval_results = evaluate_4class_model()
    
    print("\n" + "=" * 80)
    print("PROCESS COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review results in: 4class_results/")
    print("  2. Generate visualizations: python generate_4class_visualizations.py")
    print("  3. Update documentation with new 4-class system")
    print("=" * 80)


if __name__ == "__main__":
    main()
