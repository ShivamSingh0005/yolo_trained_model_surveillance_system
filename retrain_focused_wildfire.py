"""
Retrain model with focus on Wildfire class
Uses YOLO's built-in augmentation and class-weighted training
"""

from ultralytics import YOLO
import torch
from pathlib import Path
import json
import yaml

def create_wildfire_focused_config():
    """Create training config with wildfire-focused augmentation"""
    
    # Create custom hyperparameters file
    hyp_config = {
        # Learning rate
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        
        # Augmentation - Enhanced for fire detection
        'hsv_h': 0.02,   # Hue (fire color variations)
        'hsv_s': 0.8,    # Saturation (fire intensity)
        'hsv_v': 0.5,    # Value (brightness for flames)
        'degrees': 15.0,  # Rotation
        'translate': 0.1,  # Translation
        'scale': 0.6,     # Scale variation
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.5,    # Vertical flip
        'fliplr': 0.5,    # Horizontal flip
        'mosaic': 1.0,    # Mosaic augmentation
        'mixup': 0.15,    # Mixup for better generalization
        'copy_paste': 0.1,  # Copy-paste augmentation
        
        # Loss weights
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        
        # Class weights (focus on wildfire)
        'cls_pw': 1.0,
        'obj_pw': 1.0,
    }
    
    with open('hyp_wildfire.yaml', 'w') as f:
        yaml.dump(hyp_config, f)
    
    print("✓ Created wildfire-focused hyperparameters: hyp_wildfire.yaml")
    return 'hyp_wildfire.yaml'


def train_wildfire_focused():
    """Train model with focus on wildfire detection"""
    
    print("=" * 80)
    print("WILDFIRE-FOCUSED RETRAINING")
    print("=" * 80)
    print("\nStrategy:")
    print("  1. Enhanced color augmentation for fire/smoke")
    print("  2. Increased training epochs for better convergence")
    print("  3. Optimized learning rate schedule")
    print("  4. Strong augmentation to improve generalization")
    print("=" * 80)
    
    # Create hyperparameters
    hyp_file = create_wildfire_focused_config()
    
    # Load model - start from previous best weights
    best_model = Path('runs/detect/runs/surveillance/weights/best.pt')
    if best_model.exists():
        print(f"\n✓ Loading previous best model: {best_model}")
        model = YOLO(str(best_model))
        print("  (Fine-tuning from previous training)")
    else:
        print("\n✓ Loading YOLOv8n pretrained model")
        model = YOLO('yolov8n.pt')
    
    print("\nStarting training...")
    print("-" * 80)
    
    # Train with optimized settings
    results = model.train(
        # Dataset
        data='data.yaml',
        
        # Training duration - more epochs for better wildfire learning
        epochs=200,
        
        # Batch and image size
        batch=16,
        imgsz=640,
        
        # Optimizer
        optimizer='AdamW',
        
        # Use custom hyperparameters
        # Note: YOLO will use the augmentation settings from hyp file
        
        # Enhanced augmentation for fire detection
        hsv_h=0.02,   # Hue variation (fire colors)
        hsv_s=0.8,    # Saturation (fire intensity)
        hsv_v=0.5,    # Value (brightness)
        degrees=15.0,
        translate=0.1,
        scale=0.6,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        
        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # Learning rate
        lr0=0.001,
        lrf=0.01,
        
        # Training settings
        patience=75,
        save=True,
        save_period=20,
        cache=False,
        device=0 if torch.cuda.is_available() else 'cpu',
        workers=8,
        project='runs/detect',
        name='wildfire_focused_v2',
        exist_ok=True,
        pretrained=True,
        verbose=True,
        
        # Validation
        val=True,
        plots=True,
        
        # Advanced
        cos_lr=True,
        close_mosaic=15,
        
        # Resume if interrupted
        resume=False,
    )
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    
    # Evaluate
    best_model_path = Path('runs/detect/wildfire_focused_v2/weights/best.pt')
    
    if best_model_path.exists():
        print(f"\n✓ Best model: {best_model_path}")
        
        # Load and evaluate
        model = YOLO(str(best_model_path))
        
        print("\nEvaluating on test set...")
        metrics = model.val(
            data='data.yaml',
            split='test',
            save_json=True,
            plots=True,
            project='runs/detect',
            name='wildfire_focused_v2_val'
        )
        
        # Save metrics
        results_dict = {
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
            'Precision': float(metrics.box.mp),
            'Recall': float(metrics.box.mr),
            'F1-Score': 2 * (float(metrics.box.mp) * float(metrics.box.mr)) / (float(metrics.box.mp) + float(metrics.box.mr) + 1e-6)
        }
        
        # Per-class metrics
        class_names = ['Animal', 'Forest', 'Militant', 'UAV-Drone', 'Wildfire']
        results_dict['per_class'] = {}
        
        for idx, class_name in enumerate(class_names):
            if idx < len(metrics.box.ap50):
                results_dict['per_class'][class_name] = {
                    'AP50': float(metrics.box.ap50[idx]),
                    'AP': float(metrics.box.ap[idx]) if hasattr(metrics.box, 'ap') else 0.0,
                    'Precision': float(metrics.box.p[idx]) if hasattr(metrics.box, 'p') and idx < len(metrics.box.p) else 0.0,
                    'Recall': float(metrics.box.r[idx]) if hasattr(metrics.box, 'r') and idx < len(metrics.box.r) else 0.0,
                }
        
        # Save results
        results_dir = Path('wildfire_improved_results')
        results_dir.mkdir(exist_ok=True)
        
        with open(results_dir / 'improved_metrics.json', 'w') as f:
            json.dump(results_dict, f, indent=4)
        
        # Compare with original
        print("\n" + "=" * 80)
        print("PERFORMANCE COMPARISON - WILDFIRE CLASS")
        print("=" * 80)
        
        original_metrics_path = Path('evaluation_results/metrics.json')
        if original_metrics_path.exists():
            with open(original_metrics_path, 'r') as f:
                original = json.load(f)
            
            if 'Wildfire' in results_dict['per_class'] and 'Wildfire' in original['per_class']:
                new_wf = results_dict['per_class']['Wildfire']
                old_wf = original['per_class']['Wildfire']
                
                print(f"\n{'Metric':<20} {'Original':<15} {'Improved':<15} {'Change':<15}")
                print("-" * 65)
                
                for metric in ['AP50', 'Precision', 'Recall']:
                    old_val = old_wf.get(metric, 0) * 100
                    new_val = new_wf.get(metric, 0) * 100
                    change = new_val - old_val
                    
                    status = "✓" if change > 0 else "✗" if change < 0 else "="
                    change_str = f"{status} {change:+.1f}%"
                    
                    print(f"{metric:<20} {old_val:>6.1f}%{'':<8} {new_val:>6.1f}%{'':<8} {change_str:<15}")
                
                print("\n" + "-" * 65)
                
                # Overall improvement
                avg_old = (old_wf.get('AP50', 0) + old_wf.get('Precision', 0) + old_wf.get('Recall', 0)) / 3 * 100
                avg_new = (new_wf.get('AP50', 0) + new_wf.get('Precision', 0) + new_wf.get('Recall', 0)) / 3 * 100
                avg_change = avg_new - avg_old
                
                print(f"{'Average':<20} {avg_old:>6.1f}%{'':<8} {avg_new:>6.1f}%{'':<8} {avg_change:+.1f}%")
                
                if avg_change > 5:
                    print("\n🎉 Significant improvement achieved!")
                elif avg_change > 0:
                    print("\n✓ Improvement achieved")
                else:
                    print("\n⚠️  Consider additional training or data collection")
        
        print("\n" + "=" * 80)
        print(f"✓ Results saved to: {results_dir}/")
        print("=" * 80)
        
        return results_dict
    
    else:
        print("\n⚠️  Best model not found")
        return None


if __name__ == "__main__":
    results = train_wildfire_focused()
