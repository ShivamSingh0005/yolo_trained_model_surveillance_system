# Complete Usage Guide

## Quick Start (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Quick Test (10 epochs)
```bash
python quick_start.py train
python quick_start.py metrics
python quick_start.py predict
```

## Full Pipeline (Recommended)

### Complete Training & Evaluation
```bash
python complete_pipeline.py --mode all --epochs 100 --batch 16
```

This will:
1. Train YOLOv8 model for 100 epochs
2. Evaluate on test set
3. Generate all metrics and visualizations

**Expected Time**: 2-4 hours (depending on GPU)

## Step-by-Step Execution

### Step 1: Training Only
```bash
python train_pipeline.py
```

**Output**:
- `runs/surveillance/train/weights/best.pt` (best model)
- `runs/surveillance/train/weights/last.pt` (last checkpoint)
- `runs/surveillance/train/results.csv` (training logs)

**Expected Time**: 1-3 hours

### Step 2: Evaluation Only
```bash
python evaluate_model.py
```

**Prerequisites**: Trained model must exist

**Output**:
- `evaluation_results/metrics.json`
- `evaluation_results/evaluation_report.txt`
- `evaluation_results/overall_metrics.png`
- `evaluation_results/per_class_metrics.png`

**Expected Time**: 2-5 minutes

### Step 3: Visualization Only
```bash
python visualize_results.py
```

**Prerequisites**: Training and evaluation completed

**Output**:
- `evaluation_results/training_curves.png`
- `evaluation_results/class_distribution.png`
- `evaluation_results/performance_heatmap.png`
- `evaluation_results/inference_samples.png`
- `evaluation_results/summary_dashboard.png`

**Expected Time**: 1-2 minutes

## Custom Training Parameters

### Adjust Epochs
```bash
python complete_pipeline.py --mode train --epochs 50
```

### Adjust Batch Size
```bash
python complete_pipeline.py --mode train --batch 8
```

### Both
```bash
python complete_pipeline.py --mode all --epochs 150 --batch 32
```

## Understanding the Outputs

### 1. Metrics JSON (`evaluation_results/metrics.json`)
```json
{
    "mAP50": 0.85,
    "mAP50-95": 0.65,
    "Precision": 0.82,
    "Recall": 0.78,
    "F1-Score": 0.80,
    "per_class": {
        "Animal": {
            "AP50": 0.88,
            "AP": 0.70,
            "Precision": 0.85,
            "Recall": 0.82
        },
        ...
    }
}
```

### 2. Evaluation Report (`evaluation_results/evaluation_report.txt`)
Text-based comprehensive report with all metrics formatted for easy reading.

### 3. Visualizations

#### Overall Metrics (`overall_metrics.png`)
- Bar chart of main metrics
- Radar chart showing balanced performance

#### Per-Class Metrics (`per_class_metrics.png`)
- 4 subplots showing AP@0.5, AP@0.5:0.95, Precision, Recall for each class

#### Training Curves (`training_curves.png`)
- 6 subplots showing loss curves and metric progression over epochs

#### Class Distribution (`class_distribution.png`)
- Bar charts showing instance counts in train and test sets

#### Performance Heatmap (`performance_heatmap.png`)
- Color-coded matrix of all metrics across all classes

#### Inference Samples (`inference_samples.png`)
- 6 sample predictions with bounding boxes and labels

#### Summary Dashboard (`summary_dashboard.png`)
- Comprehensive single-page overview of all key metrics

## Inference on New Images

### Single Image
```python
from ultralytics import YOLO

model = YOLO('runs/surveillance/train/weights/best.pt')
results = model('path/to/image.jpg')
results[0].show()  # Display
results[0].save('output.jpg')  # Save
```

### Batch Inference
```python
from ultralytics import YOLO
from pathlib import Path

model = YOLO('runs/surveillance/train/weights/best.pt')
images = list(Path('test/images').glob('*.jpg'))

for img in images:
    results = model(img)
    results[0].save(f'predictions/{img.name}')
```

### Video Inference
```python
from ultralytics import YOLO

model = YOLO('runs/surveillance/train/weights/best.pt')
results = model('video.mp4', stream=True)

for result in results:
    result.show()  # Display frame
```

## Performance Metrics Explained

### mAP@0.5
- Mean Average Precision at IoU threshold 0.5
- Higher is better (0-1 range)
- Industry standard for object detection

### mAP@0.5:0.95
- Mean Average Precision averaged over IoU thresholds 0.5 to 0.95
- More strict metric
- COCO dataset standard

### Precision
- What percentage of detections are correct?
- Precision = TP / (TP + FP)

### Recall
- What percentage of ground truth objects are detected?
- Recall = TP / (TP + FN)

### F1-Score
- Harmonic mean of Precision and Recall
- Balanced metric: F1 = 2 * (P * R) / (P + R)

## Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size
python complete_pipeline.py --mode train --batch 4
```

### Slow Training
```bash
# Use smaller image size (edit train_pipeline.py)
# Change imgsz=640 to imgsz=416
```

### Model Not Found
```bash
# Ensure training completed successfully
# Check: runs/surveillance/train/weights/best.pt exists
```

### No GPU Detected
```bash
# Training will use CPU (slower)
# Install CUDA and PyTorch with GPU support
```

## Advanced Usage

### Resume Training
```python
from ultralytics import YOLO

model = YOLO('runs/surveillance/train/weights/last.pt')
model.train(data='data.yaml', epochs=50, resume=True)
```

### Export Model
```python
from ultralytics import YOLO

model = YOLO('runs/surveillance/train/weights/best.pt')

# Export to ONNX
model.export(format='onnx')

# Export to TensorRT
model.export(format='engine')

# Export to TFLite
model.export(format='tflite')
```

### Hyperparameter Tuning
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.tune(data='data.yaml', epochs=30, iterations=300)
```

## Best Practices

1. **Start Small**: Use quick_start.py for initial testing
2. **Monitor Training**: Check training curves for overfitting
3. **Validate Often**: Run evaluation after training
4. **Save Everything**: Keep all outputs for comparison
5. **Document Changes**: Note any modifications to hyperparameters

## Expected Performance Ranges

Based on similar surveillance datasets:

- **mAP@0.5**: 0.75 - 0.90 (Good: >0.80)
- **mAP@0.5:0.95**: 0.50 - 0.70 (Good: >0.60)
- **Precision**: 0.70 - 0.90 (Good: >0.80)
- **Recall**: 0.65 - 0.85 (Good: >0.75)

## Support

For issues or questions:
1. Check training logs in `runs/surveillance/train/`
2. Review evaluation report in `evaluation_results/`
3. Verify dataset structure matches expected format
