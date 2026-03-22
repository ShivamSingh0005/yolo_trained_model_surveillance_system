# Surveillance System - YOLOv8 Pipeline

Complete training, evaluation, and visualization pipeline for a surveillance system detecting 5 classes:
- Animal
- Forest
- Militant
- UAV-Drone
- Wildfire

## Dataset Structure

```
.
├── train/
│   ├── images/     (646 images)
│   └── labels/     (646 labels)
├── test/
│   ├── images/     (114 images)
│   └── labels/     (114 labels)
└── data.yaml       (dataset configuration)
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Complete Pipeline (Recommended)

Run training, evaluation, and visualization in one command:

```bash
python complete_pipeline.py --mode all --epochs 100 --batch 16
```

### Option 2: Individual Steps

#### Step 1: Train Model
```bash
python train_pipeline.py
```

#### Step 2: Evaluate Model
```bash
python evaluate_model.py
```

#### Step 3: Generate Visualizations
```bash
python visualize_results.py
```

### Option 3: Selective Execution

```bash
# Only training
python complete_pipeline.py --mode train --epochs 50 --batch 8

# Only evaluation
python complete_pipeline.py --mode eval

# Only visualization
python complete_pipeline.py --mode viz
```

## Performance Metrics

The pipeline generates comprehensive metrics including:

### Overall Metrics
- mAP@0.5 (Mean Average Precision at IoU=0.5)
- mAP@0.5:0.95 (Mean Average Precision at IoU=0.5 to 0.95)
- Precision
- Recall
- F1-Score

### Per-Class Metrics
- AP@0.5 for each class
- AP@0.5:0.95 for each class
- Precision per class
- Recall per class

## Generated Outputs

### Model Files
- `runs/surveillance/train/weights/best.pt` - Best model checkpoint
- `runs/surveillance/train/weights/last.pt` - Last epoch checkpoint

### Evaluation Results
- `evaluation_results/metrics.json` - All metrics in JSON format
- `evaluation_results/evaluation_report.txt` - Detailed text report

### Visualizations
1. `overall_metrics.png` - Overall performance bar chart and radar plot
2. `per_class_metrics.png` - Per-class performance comparison
3. `training_curves.png` - Training/validation loss and metrics over epochs
4. `class_distribution.png` - Dataset class distribution
5. `performance_heatmap.png` - Heatmap of all metrics across classes
6. `inference_samples.png` - Sample predictions on test images
7. `summary_dashboard.png` - Comprehensive performance dashboard

### Training Artifacts
- `runs/surveillance/train/confusion_matrix.png` - Confusion matrix
- `runs/surveillance/train/PR_curve.png` - Precision-Recall curve
- `runs/surveillance/train/results.csv` - Training metrics per epoch

## Model Architecture

- Base Model: YOLOv8n (nano)
- Input Size: 640x640
- Optimizer: Auto (AdamW/SGD)
- Learning Rate: 0.01
- Batch Size: 16 (configurable)
- Epochs: 100 (configurable)

## Training Configuration

Key hyperparameters:
- Image size: 640x640
- Batch size: 16
- Learning rate: 0.01
- Momentum: 0.937
- Weight decay: 0.0005
- Data augmentation: Mosaic, flip, HSV adjustments
- Early stopping patience: 20 epochs

## Evaluation Metrics Explained

- **mAP@0.5**: Average precision across all classes at IoU threshold 0.5
- **mAP@0.5:0.95**: Average precision across IoU thresholds from 0.5 to 0.95
- **Precision**: Ratio of true positives to all positive predictions
- **Recall**: Ratio of true positives to all actual positives
- **F1-Score**: Harmonic mean of precision and recall

## Example Results Structure

```
evaluation_results/
├── metrics.json
├── evaluation_report.txt
├── overall_metrics.png
├── per_class_metrics.png
├── training_curves.png
├── class_distribution.png
├── performance_heatmap.png
├── inference_samples.png
└── summary_dashboard.png

runs/surveillance/train/
├── weights/
│   ├── best.pt
│   └── last.pt
├── confusion_matrix.png
├── PR_curve.png
├── F1_curve.png
├── P_curve.png
├── R_curve.png
└── results.csv
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 5GB+ disk space

## License

CC BY 4.0

## Dataset Source

Roboflow Universe - Surveillance System Dataset
