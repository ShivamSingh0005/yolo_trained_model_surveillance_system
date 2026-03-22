"""
Train YOLO model with augmented dataset and class weights
Focus on improving Wildfire detection performance
"""

from ultralytics import YOLO
import torch
from pathlib import Path
import json

def train_improved_model():
    """Train model with augmented data and optimized hyperparameters"""
    
    print("=" * 80)
    print("TRAINING WITH AUGMENTED DATA - WILDFIRE IMPROVEMENT")
    print("=" * 80)
    
    # Check if augmented dataset exists
    if not Path('train_augmented').exists():
        print("\n⚠️  Augmented dataset not found!")
        print("Please run: python improve_wildfire_detection.py first")
        return
    
    # Load model
    print("\nLoading YOLOv8n model...")
    model = YOLO('yolov8n.pt')
    
    # Training configuration optimized for wildfire detection
    print("\nStarting training with optimized hyperparameters...")
    print("Focus: Improved Wildfire detection")
    print("-" * 80)
    
    results = model.train(
        # Dataset
        data='data_augmented.yaml',
        
        # Training duration
        epochs=150,  # More epochs for better convergence
        
        # Batch and image size
        batch=16,
        imgsz=640,
        
        # Optimization
        optimizer='AdamW',  # Better for imbalanced datasets
        lr0=0.001,  # Initial learning rate
        lrf=0.01,  # Final learning rate
        momentum=0.937,
        weight_decay=0.0005,
        
        # Augmentation (additional to albumentations)
        hsv_h=0.015,  # Hue augmentation
        hsv_s=0.7,    # Saturation augmentation (important for fire)
        hsv_v=0.4,    # Value augmentation
        degrees=10.0,  # Rotation
        translate=0.1,  # Translation
        scale=0.5,     # Scaling
        shear=0.0,     # Shear
        perspective=0.0,  # Perspective
        flipud=0.5,    # Vertical flip
        fliplr=0.5,    # Horizontal flip
        mosaic=1.0,    # Mosaic augmentation
        mixup=0.1,     # Mixup augmentation
        
        # Loss weights - increase box loss for better localization
        box=7.5,       # Box loss weight
        cls=0.5,       # Class loss weight  
        dfl=1.5,       # Distribution focal loss weight
        
        # Training settings
        patience=50,   # Early stopping patience
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        cache=False,   # Don't cache (memory intensive)
        device=0 if torch.cuda.is_available() else 'cpu',
        workers=8,
        project='runs/detect',
        name='wildfire_improved',
        exist_ok=True,
        pretrained=True,
        verbose=True,
        
        # Validation
        val=True,
        plots=True,
        
        # Advanced
        cos_lr=True,  # Cosine learning rate scheduler
        close_mosaic=10,  # Disable mosaic in last 10 epochs
    )
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    
    # Get best model path
    best_model_path = Path('runs/detect/wildfire_improved/weights/best.pt')
    
    if best_model_path.exists():
        print(f"\n✓ Best model saved to: {best_model_path}")
        
        # Evaluate on test set
        print("\nEvaluating improved model...")
        model = YOLO(str(best_model_path))
        
        metrics = model.val(
            data='data_augmented.yaml',
            split='test',
            save_json=True,
            plots=True
        )
        
        # Extract metrics
        results_dict = {
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
            'Precision': float(metrics.box.mp),
            'Recall': float(metrics.box.mr),
        }
        
        # Per-class metrics
        if hasattr(metrics.box, 'maps'):
            class_names = ['Animal', 'Forest', 'Militant', 'UAV-Drone', 'Wildfire']
            results_dict['per_class'] = {}
            
            for idx, class_name in enumerate(class_names):
                if idx < len(metrics.box.maps):
                    results_dict['per_class'][class_name] = {
                        'AP50': float(metrics.box.ap50[idx]),
                        'AP': float(metrics.box.maps[idx]),
                        'Precision': float(metrics.box.p[idx]) if hasattr(metrics.box, 'p') else 0.0,
                        'Recall': float(metrics.box.r[idx]) if hasattr(metrics.box, 'r') else 0.0,
                    }
        
        # Save results
        results_path = Path('wildfire_improved_results')
        results_path.mkdir(exist_ok=True)
        
        with open(results_path / 'metrics.json', 'w') as f:
            json.dump(results_dict, f, indent=4)
        
        print("\n" + "=" * 80)
        print("PERFORMANCE COMPARISON")
        print("=" * 80)
        
        # Load original metrics
        original_metrics_path = Path('evaluation_results/metrics.json')
        if original_metrics_path.exists():
            with open(original_metrics_path, 'r') as f:
                original_metrics = json.load(f)
            
            print("\nWILDFIRE CLASS PERFORMANCE:")
            print("-" * 80)
            
            if 'Wildfire' in results_dict.get('per_class', {}):
                new_wildfire = results_dict['per_class']['Wildfire']
                old_wildfire = original_metrics['per_class']['Wildfire']
                
                print(f"{'Metric':<20} {'Original':<15} {'Improved':<15} {'Change':<15}")
                print("-" * 65)
                
                metrics_to_compare = ['AP50', 'Precision', 'Recall']
                for metric in metrics_to_compare:
                    old_val = old_wildfire.get(metric, 0) * 100
                    new_val = new_wildfire.get(metric, 0) * 100
                    change = new_val - old_val
                    change_str = f"+{change:.1f}%" if change >= 0 else f"{change:.1f}%"
                    
                    print(f"{metric:<20} {old_val:>6.1f}%{'':<8} {new_val:>6.1f}%{'':<8} {change_str:<15}")
                
                print("\n" + "=" * 80)
        
        print("\n✓ Improved model evaluation complete")
        print(f"✓ Results saved to: {results_path}/")
        
    else:
        print("\n⚠️  Training completed but best model not found")
    
    return results


if __name__ == "__main__":
    results = train_improved_model()
