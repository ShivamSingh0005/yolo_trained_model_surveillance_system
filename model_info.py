"""
Model Information and Usage Script
Display model details and demonstrate how to use the trained model
"""

from ultralytics import YOLO
from pathlib import Path
import torch

def display_model_info():
    """Display comprehensive model information"""
    model_path = 'runs/detect/runs/surveillance/weights/best.pt'
    
    print("=" * 80)
    print("TRAINED MODEL INFORMATION")
    print("=" * 80)
    
    # Load model
    model = YOLO(model_path)
    
    print(f"\n📁 Model Path: {model_path}")
    print(f"📦 File Size: {Path(model_path).stat().st_size / (1024*1024):.2f} MB")
    
    # Model architecture
    print("\n" + "=" * 80)
    print("MODEL ARCHITECTURE")
    print("=" * 80)
    print(f"Base Architecture: YOLOv8n (nano)")
    print(f"Total Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    print(f"Trainable Parameters: {sum(p.numel() for p in model.model.parameters() if p.requires_grad):,}")
    
    # Model details
    print("\n" + "=" * 80)
    print("MODEL DETAILS")
    print("=" * 80)
    
    # Load checkpoint to get training info
    ckpt = torch.load(model_path, map_location='cpu')
    
    if 'epoch' in ckpt:
        print(f"Training Epochs: {ckpt['epoch']}")
    
    if 'model' in ckpt:
        print(f"Model Type: {type(ckpt['model']).__name__}")
    
    # Class names
    print("\n" + "=" * 80)
    print("DETECTION CLASSES (5 classes)")
    print("=" * 80)
    class_names = model.names
    for idx, name in class_names.items():
        print(f"  {idx}: {name}")
    
    # Performance metrics (if available)
    if 'best_fitness' in ckpt:
        print("\n" + "=" * 80)
        print("TRAINING PERFORMANCE")
        print("=" * 80)
        print(f"Best Fitness Score: {ckpt['best_fitness']:.4f}")
    
    # Model summary
    print("\n" + "=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    model.info()
    
    print("\n" + "=" * 80)
    print("USAGE EXAMPLES")
    print("=" * 80)
    print("""
# 1. Load the model
from ultralytics import YOLO
model = YOLO('runs/detect/runs/surveillance/weights/best.pt')

# 2. Run inference on an image
results = model('path/to/image.jpg')

# 3. Run inference on a folder
results = model('path/to/folder/')

# 4. Run inference on a video
results = model('path/to/video.mp4')

# 5. Get predictions
for result in results:
    boxes = result.boxes  # Bounding boxes
    probs = result.probs  # Class probabilities
    
# 6. Save results
results[0].save('output.jpg')

# 7. Display results
results[0].show()

# 8. Export model to different formats
model.export(format='onnx')  # Export to ONNX
model.export(format='torchscript')  # Export to TorchScript
model.export(format='tflite')  # Export to TensorFlow Lite
    """)

def test_inference():
    """Test inference on a sample image"""
    model_path = 'runs/detect/runs/surveillance/weights/best.pt'
    test_images_dir = Path('test/images')
    
    if not test_images_dir.exists():
        print("\n[WARNING] Test images directory not found")
        return
    
    print("\n" + "=" * 80)
    print("TESTING INFERENCE")
    print("=" * 80)
    
    # Load model
    model = YOLO(model_path)
    
    # Get first test image
    test_images = list(test_images_dir.glob('*.jpg'))
    if not test_images:
        print("[WARNING] No test images found")
        return
    
    test_image = test_images[0]
    print(f"\nRunning inference on: {test_image.name}")
    
    # Run inference
    results = model(str(test_image), verbose=False)
    
    # Display results
    result = results[0]
    print(f"\nDetections found: {len(result.boxes)}")
    
    if len(result.boxes) > 0:
        print("\nDetected objects:")
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]
            print(f"  - {class_name}: {confidence:.2%} confidence")
    
    # Save result
    output_path = 'test_inference_result.jpg'
    result.save(output_path)
    print(f"\n✓ Result saved to: {output_path}")

def export_model():
    """Export model to different formats"""
    model_path = 'runs/detect/runs/surveillance/weights/best.pt'
    
    print("\n" + "=" * 80)
    print("MODEL EXPORT OPTIONS")
    print("=" * 80)
    
    model = YOLO(model_path)
    
    print("""
Available export formats:
  1. ONNX        - Cross-platform inference
  2. TorchScript - PyTorch deployment
  3. TensorFlow  - TensorFlow deployment
  4. TFLite      - Mobile/edge devices
  5. CoreML      - iOS devices
  6. OpenVINO    - Intel hardware optimization

To export, run:
  model.export(format='onnx')
  model.export(format='torchscript')
  model.export(format='tflite')
    """)

def main():
    """Main execution"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'info':
            display_model_info()
        elif command == 'test':
            test_inference()
        elif command == 'export':
            export_model()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python model_info.py [info|test|export]")
    else:
        # Run all by default
        display_model_info()
        test_inference()

if __name__ == "__main__":
    main()
