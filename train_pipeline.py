"""
Surveillance System - YOLOv8 Training Pipeline
Dataset: 5 classes - Animal, Forest, Militant, UAV-Drone, Wildfire
"""

import os
from pathlib import Path
from ultralytics import YOLO
import yaml

def setup_training():
    """Initialize and configure YOLOv8 model"""
    # Load pretrained YOLOv8 model
    model = YOLO('yolov8n.pt')  # nano model for faster training
    
    return model

def train_model(model, data_yaml='data.yaml', epochs=100, imgsz=640, batch=16):
    """Train the model with specified parameters"""
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=20,
        save=True,
        device=0,  # Use GPU if available, else CPU
        project='runs/surveillance',
        name='train',
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
        freeze=None,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        pose=12.0,
        kobj=1.0,
        label_smoothing=0.0,
        nbs=64,
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
    
    return results

if __name__ == "__main__":
    print("=" * 60)
    print("Surveillance System - YOLOv8 Training Pipeline")
    print("=" * 60)
    
    # Setup
    model = setup_training()
    
    # Train
    print("\n[INFO] Starting training...")
    results = train_model(model, epochs=100, batch=16)
    
    print("\n[INFO] Training completed!")
    print(f"[INFO] Best model saved at: runs/surveillance/train/weights/best.pt")
