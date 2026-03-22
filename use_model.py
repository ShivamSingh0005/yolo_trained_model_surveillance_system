"""
Simple Model Usage Script
Quick examples of how to use your trained model
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

def load_model():
    """Load the trained model"""
    model_path = 'runs/detect/runs/surveillance/weights/best.pt'
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    print("✓ Model loaded successfully!")
    return model

def predict_single_image(model, image_path):
    """Run prediction on a single image"""
    print(f"\nRunning prediction on: {image_path}")
    
    # Run inference
    results = model(image_path, verbose=False)
    result = results[0]
    
    # Display detections
    print(f"Found {len(result.boxes)} objects:")
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]
        coords = box.xyxy[0].tolist()
        print(f"  - {class_name}: {confidence:.2%} at [{coords[0]:.0f}, {coords[1]:.0f}, {coords[2]:.0f}, {coords[3]:.0f}]")
    
    return result

def predict_and_save(model, image_path, output_path='prediction_result.jpg'):
    """Predict and save result"""
    result = predict_single_image(model, image_path)
    
    # Save annotated image
    result.save(output_path)
    print(f"\n✓ Result saved to: {output_path}")
    
    return output_path

def predict_multiple_images(model, image_folder='test/images', num_images=3):
    """Run predictions on multiple images"""
    image_dir = Path(image_folder)
    
    if not image_dir.exists():
        print(f"[ERROR] Directory not found: {image_folder}")
        return
    
    images = list(image_dir.glob('*.jpg'))[:num_images]
    
    print(f"\nProcessing {len(images)} images...")
    
    results = []
    for img_path in images:
        print(f"\n{'='*60}")
        result = predict_single_image(model, str(img_path))
        results.append(result)
    
    return results

def create_comparison_grid(model, num_images=6):
    """Create a grid of predictions"""
    test_dir = Path('test/images')
    
    if not test_dir.exists():
        print("[ERROR] Test images directory not found")
        return
    
    images = list(test_dir.glob('*.jpg'))[:num_images]
    
    if len(images) == 0:
        print("[ERROR] No test images found")
        return
    
    print(f"\nCreating comparison grid with {len(images)} images...")
    
    # Run predictions
    results = model(images, verbose=False)
    
    # Create grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (result, img_path) in enumerate(zip(results, images)):
        if idx >= 6:
            break
        
        # Get annotated image
        annotated = result.plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        # Plot
        axes[idx].imshow(annotated_rgb)
        axes[idx].axis('off')
        axes[idx].set_title(f'{img_path.stem[:20]}... ({len(result.boxes)} detections)', 
                           fontsize=10)
    
    plt.tight_layout()
    output_path = 'prediction_grid.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Grid saved to: {output_path}")
    plt.close()

def model_summary(model):
    """Display model summary"""
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    
    print(f"\nClasses ({len(model.names)}):")
    for idx, name in model.names.items():
        print(f"  {idx}: {name}")
    
    print(f"\nModel Architecture: YOLOv8n")
    print(f"Parameters: 3,011,823")
    print(f"Model Size: 5.97 MB")
    print(f"Input Size: 640×640")
    
    print("\n" + "="*60)

def main():
    """Main execution"""
    print("="*60)
    print("YOLO SURVEILLANCE MODEL - USAGE DEMO")
    print("="*60)
    
    # Load model
    model = load_model()
    
    # Show model info
    model_summary(model)
    
    # Test on first image
    test_dir = Path('test/images')
    if test_dir.exists():
        images = list(test_dir.glob('*.jpg'))
        if images:
            print("\n" + "="*60)
            print("SINGLE IMAGE PREDICTION")
            print("="*60)
            predict_and_save(model, str(images[0]))
            
            print("\n" + "="*60)
            print("MULTIPLE IMAGES PREDICTION")
            print("="*60)
            predict_multiple_images(model, num_images=3)
            
            print("\n" + "="*60)
            print("CREATING PREDICTION GRID")
            print("="*60)
            create_comparison_grid(model, num_images=6)
    
    print("\n" + "="*60)
    print("USAGE EXAMPLES")
    print("="*60)
    print("""
# Load model
from ultralytics import YOLO
model = YOLO('runs/detect/runs/surveillance/weights/best.pt')

# Predict on image
results = model('image.jpg')

# Predict on folder
results = model('test/images/')

# Predict on video
results = model('video.mp4')

# Get boxes and save
for result in results:
    boxes = result.boxes
    result.save('output.jpg')
    """)
    
    print("\n✓ Demo completed!")

if __name__ == "__main__":
    main()
