"""
Fine-tune from the original best model with conservative settings
Focus on improving Wildfire without degrading other classes
"""

from ultralytics import YOLO
import torch
from pathlib import Path
import json

def finetune_for_wildfire():
    """Fine-tune the original best model"""
    
    print("=" * 80)
    print("FINE-TUNING FOR WILDFIRE IMPROVEMENT")
    print("=" * 80)
    print("\nStrategy:")
    print("  1. Start from original best model (mAP50: 80%)")
    print("  2. Use conservative learning rate for fine-tuning")
    print("  3. Moderate augmentation to avoid overfitting")
    print("  4. Fewer epochs with early stopping")
    print("=" * 80)
    
    # Load original best model
    original_model = Path('runs/detect/runs/surveillance/weights/best.pt')
    
    if not original_model.exists():
        print(f"\n⚠️  Original model not found: {original_model}")
        print("Using YOLOv8n pretrained instead")
        model = YOLO('yolov8n.pt')
    else:
        print(f"\n✓ Loading original best model: {original_model}")
        model = YOLO(str(original_model))
    
    print("\nStarting fine-tuning...")
    print("-" * 80)
    
    # Fine-tune with conservative settings
    results = model.train(
        # Dataset
        data='data.yaml',
        
        # Training duration - fewer epochs for fine-tuning
        epochs=100,
        
        # Batch and image size
        batch=16,
        imgsz=640,
        
        # Optimizer - SGD is more stable for fine-tuning
        optimizer='SGD',
        
        # Conservative learning rate for fine-tuning
        lr0=0.0001,  # Much lower than initial training
        lrf=0.001,
        momentum=0.937,
        weight_decay=0.0005,
        
        # Moderate augmentation - less aggressive
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.8,  # Reduced
        mixup=0.05,  # Reduced
        copy_paste=0.0,  # Disabled
        
        # Loss weights - balanced
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # Training settings
        patience=30,  # Early stopping
        save=True,
        save_period=10,
        cache=False,
        device=0 if torch.cuda.is_available() else 'cpu',
        workers=8,
        project='runs/detect',
        name='wildfire_finetuned',
        exist_ok=True,
        pretrained=False,  # We're loading from best.pt
        verbose=True,
        
        # Validation
        val=True,
        plots=True,
        
        # Advanced
        cos_lr=True,
        close_mosaic=10,
        
        # Freeze early layers to preserve learned features
        freeze=10,  # Freeze first 10 layers
    )
    
    print("\n" + "=" * 80)
    print("FINE-TUNING COMPLETED")
    print("=" * 80)
    
    # Evaluate
    best_model_path = Path('runs/detect/wildfire_finetuned/weights/best.pt')
    
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
            name='wildfire_finetuned_eval'
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
                p = float(metrics.box.p[idx]) if hasattr(metrics.box, 'p') and idx < len(metrics.box.p) else 0.0
                r = float(metrics.box.r[idx]) if hasattr(metrics.box, 'r') and idx < len(metrics.box.r) else 0.0
                
                results_dict['per_class'][class_name] = {
                    'AP50': float(metrics.box.ap50[idx]),
                    'AP': float(metrics.box.ap[idx]) if hasattr(metrics.box, 'ap') else 0.0,
                    'Precision': p,
                    'Recall': r,
                }
        
        # Save results
        results_dir = Path('wildfire_finetuned_results')
        results_dir.mkdir(exist_ok=True)
        
        with open(results_dir / 'finetuned_metrics.json', 'w') as f:
            json.dump(results_dict, f, indent=4)
        
        # Compare
        print("\n" + "=" * 80)
        print("PERFORMANCE COMPARISON")
        print("=" * 80)
        
        original_metrics_path = Path('evaluation_results/metrics.json')
        if original_metrics_path.exists():
            with open(original_metrics_path, 'r') as f:
                original = json.load(f)
            
            print("\nWILDFIRE CLASS:")
            print("-" * 80)
            
            if 'Wildfire' in results_dict['per_class']:
                wf_new = results_dict['per_class']['Wildfire']
                wf_old = original['per_class']['Wildfire']
                
                print(f"{'Metric':<20} {'Original':<15} {'Fine-tuned':<15} {'Change':<15}")
                print("-" * 80)
                
                for metric in ['AP50', 'Precision', 'Recall']:
                    old_val = wf_old.get(metric, 0) * 100
                    new_val = wf_new.get(metric, 0) * 100
                    change = new_val - old_val
                    
                    status = "✓" if change > 0 else "✗" if change < 0 else "="
                    change_str = f"{status} {change:+.1f}%"
                    
                    print(f"{metric:<20} {old_val:>6.1f}%{'':<8} {new_val:>6.1f}%{'':<8} {change_str:<15}")
                
                if wf_new['AP50'] >= 0.80:
                    print("\n🎉 TARGET ACHIEVED! Wildfire AP50 >= 80%")
                elif wf_new['AP50'] >= 0.60:
                    print("\n✓ Good improvement!")
                elif wf_new['AP50'] > wf_old['AP50']:
                    print("\n✓ Improved, but more work needed")
                else:
                    print("\n⚠️  No improvement. Try different approach.")
        
        print("\n" + "=" * 80)
        print(f"✓ Results saved to: {results_dir}/")
        print("=" * 80)
        
        return results_dict
    
    else:
        print("\n⚠️  Best model not found")
        return None


if __name__ == "__main__":
    results = finetune_for_wildfire()
