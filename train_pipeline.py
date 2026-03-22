"""
Training Pipeline for Surveillance System
YOLOv8 Model Training with Comprehensive Logging
"""

import os
from pathlib import Path
from ultralytics import YOLO
import torch
import yaml

def setup_training():
    """Initialize model and check environment"""
    print("[INFO] Setting up training environment...")
    
    # Check CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")
    if device == 'cuda':
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] CUDA Version: {torch.version.cuda}")
    
    # Load pre-trained YOLOv8 model
    model = YOLO('yolov8n.pt')
    print("[INFO] Loaded YOLOv8n base model")
    
    return model

def train_model(model, epochs=100, batch=16, imgsz=640):
    """Train the model with specified parameters"""
    print(f"\n[INFO] Starting training...")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch Size: {batch}")
    print(f"  - Image Size: {imgsz}")
    
    # Training configuration
    results = model.train(
        data='data.yaml',
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name='surveillance',
        patience=20,
        save=True,
        device=0 if torch.cuda.is_available() else 'cpu',
        workers=8,
        project='runs',
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        seed=42,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=False,
        close_mosaic=10,
        resume=False,
        amp=True,
        fraction=1.0,
        profile=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        split='val',
        save_period=-1,
        cache=False,
        plots=True,
        # Data augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0
    )
    
    print("\n[SUCCESS] Training completed!")
    print(f"[INFO] Best model saved at: runs/surveillance/train/weights/best.pt")
    print(f"[INFO] Last model saved at: runs/surveillance/train/weights/last.pt")
    
    return results

def main():
    """Main training execution"""
    print("=" * 70)
    print("SURVEILLANCE SYSTEM - MODEL TRAINING")
    print("=" * 70)
    
    # Setup
    model = setup_training()
    
    # Train
    results = train_model(model, epochs=100, batch=16)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()
