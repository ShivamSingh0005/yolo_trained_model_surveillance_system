"""
Quick Start Script - Minimal Example
For quick testing and inference
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

def quick_train(epochs=10):
    """Quick training for testing"""
    print("[INFO] Quick training (10 epochs for testing)...")
    model = YOLO('yolov8n.pt')
    model.train(data='data.yaml', epochs=epochs, imgsz=640, batch=8, device=0)
    print("[SUCCESS] Quick training done!")

def quick_predict(image_path='test/images', model_path='runs/surveillance/train/weights/best.pt'):
    """Quick prediction on test images"""
    print("[INFO] Running predictions...")
    model = YOLO(model_path)
    
    # Get first 3 test images
    test_dir = Path(image_path)
    images = list(test_dir.glob('*.jpg'))[:3]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, img_path in enumerate(images):
        results = model(img_path, verbose=False)
        annotated = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        axes[idx].imshow(annotated_rgb)
        axes[idx].axis('off')
        axes[idx].set_title(f'{img_path.name[:20]}...', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('quick_predictions.png', dpi=150, bbox_inches='tight')
    print("[SUCCESS] Predictions saved to quick_predictions.png")
    plt.close()

def quick_metrics(model_path='runs/surveillance/train/weights/best.pt'):
    """Quick metrics evaluation"""
    print("[INFO] Evaluating model...")
    model = YOLO(model_path)
    metrics = model.val(data='data.yaml', split='test')
    
    print("\n" + "=" * 50)
    print("QUICK METRICS SUMMARY")
    print("=" * 50)
    print(f"mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print(f"Precision    : {metrics.box.mp:.4f}")
    print(f"Recall       : {metrics.box.mr:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python quick_start.py train    - Quick training (10 epochs)")
        print("  python quick_start.py predict  - Quick prediction")
        print("  python quick_start.py metrics  - Quick metrics")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'train':
        quick_train()
    elif command == 'predict':
        quick_predict()
    elif command == 'metrics':
        quick_metrics()
    else:
        print(f"Unknown command: {command}")
