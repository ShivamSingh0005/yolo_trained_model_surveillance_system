# Example Metrics Output

This document shows what the actual output will look like after running the pipeline.

## Console Output Example

```
============================================================
OVERALL PERFORMANCE METRICS
============================================================
mAP@0.5      : 0.8542
mAP@0.5:0.95 : 0.6378
Precision    : 0.8234
Recall       : 0.7891
F1-Score     : 0.8059

============================================================
PER-CLASS METRICS
============================================================
      Class  AP@0.5  AP@0.5:0.95  Precision  Recall
     Animal  0.8923       0.6845     0.8567  0.8234
     Forest  0.7834       0.5678     0.7456  0.7123
   Militant  0.9012       0.7234     0.8890  0.8456
  UAV-Drone  0.8456       0.6512     0.8234  0.7890
   Wildfire  0.8485       0.6621     0.8023  0.7752
```

## JSON Metrics Example

```json
{
    "mAP50": 0.8542,
    "mAP50-95": 0.6378,
    "Precision": 0.8234,
    "Recall": 0.7891,
    "F1-Score": 0.8059,
    "per_class": {
        "Animal": {
            "AP50": 0.8923,
            "AP": 0.6845,
            "Precision": 0.8567,
            "Recall": 0.8234
        },
        "Forest": {
            "AP50": 0.7834,
            "AP": 0.5678,
            "Precision": 0.7456,
            "Recall": 0.7123
        },
        "Militant": {
            "AP50": 0.9012,
            "AP": 0.7234,
            "Precision": 0.8890,
            "Recall": 0.8456
        },
        "UAV-Drone": {
            "AP50": 0.8456,
            "AP": 0.6512,
            "Precision": 0.8234,
            "Recall": 0.7890
        },
        "Wildfire": {
            "AP50": 0.8485,
            "AP": 0.6621,
            "Precision": 0.8023,
            "Recall": 0.7752
        }
    }
}
```

## Evaluation Report Example

```
======================================================================
SURVEILLANCE SYSTEM - MODEL EVALUATION REPORT
======================================================================

Dataset Information:
----------------------------------------------------------------------
Classes: Animal, Forest, Militant, UAV-Drone, Wildfire
Total Classes: 5
Training Images: 646
Test Images: 114

Overall Performance Metrics:
----------------------------------------------------------------------
mAP@0.5           : 0.8542
mAP@0.5:0.95      : 0.6378
Precision         : 0.8234
Recall            : 0.7891
F1-Score          : 0.8059

Per-Class Performance:
----------------------------------------------------------------------
Class           AP@0.5     AP@0.5:0.95    Precision    Recall    
----------------------------------------------------------------------
Animal          0.8923     0.6845         0.8567       0.8234    
Forest          0.7834     0.5678         0.7456       0.7123    
Militant        0.9012     0.7234         0.8890       0.8456    
UAV-Drone       0.8456     0.6512         0.8234       0.7890    
Wildfire        0.8485     0.6621         0.8023       0.7752    

======================================================================
```

## Training Progress Example

```
Epoch    Box Loss    Cls Loss    DFL Loss    Precision    Recall    mAP@0.5
-----    --------    --------    --------    ---------    ------    -------
  1/100    1.2345      0.8765      1.4567      0.4523     0.3891    0.3245
  10/100   0.8234      0.5432      1.1234      0.6234     0.5678    0.5432
  25/100   0.5678      0.3456      0.8765      0.7456     0.6789    0.6890
  50/100   0.3456      0.2345      0.6543      0.8123     0.7456    0.7890
  75/100   0.2345      0.1678      0.5234      0.8345     0.7789    0.8234
 100/100   0.1890      0.1234      0.4567      0.8234     0.7891    0.8542

Training completed in 2h 34m 12s
Best epoch: 98
Best mAP@0.5: 0.8542
```

## Class-wise Performance Interpretation

### Excellent Performance (AP@0.5 > 0.85)
- **Militant**: 0.9012 - Model excels at detecting militants
- **Animal**: 0.8923 - Strong animal detection

### Good Performance (AP@0.5 > 0.80)
- **Wildfire**: 0.8485 - Reliable wildfire detection
- **UAV-Drone**: 0.8456 - Good drone detection

### Acceptable Performance (AP@0.5 > 0.75)
- **Forest**: 0.7834 - Decent forest detection, room for improvement

## Recommendations Based on Results

### If mAP@0.5 < 0.70
- Increase training epochs
- Add more data augmentation
- Try larger model (yolov8s or yolov8m)
- Check data quality and annotations

### If Precision is Low (< 0.75)
- Too many false positives
- Increase confidence threshold during inference
- Review and clean training data

### If Recall is Low (< 0.70)
- Missing detections
- Decrease confidence threshold
- Add more training data
- Increase model capacity

### If Class Imbalance Detected
- Use weighted loss
- Oversample minority classes
- Add more augmentation for underrepresented classes

## Visualization Descriptions

### 1. Overall Metrics Bar Chart
- Horizontal bars showing 5 main metrics
- Color-coded for easy identification
- Values displayed on bars

### 2. Performance Radar Chart
- Pentagon shape showing balanced performance
- Larger area = better overall performance
- Easy to spot weaknesses

### 3. Per-Class Metrics (4 subplots)
- AP@0.5 comparison across classes
- AP@0.5:0.95 comparison
- Precision comparison
- Recall comparison

### 4. Training Curves (6 subplots)
- Box loss (train & val)
- Classification loss (train & val)
- DFL loss (train & val)
- Precision over epochs
- Recall over epochs
- mAP metrics over epochs

### 5. Class Distribution
- Training set distribution
- Test set distribution
- Helps identify imbalance

### 6. Performance Heatmap
- Color-coded matrix
- All metrics × all classes
- Green = good, Red = needs improvement

### 7. Inference Samples
- 6 test images with predictions
- Bounding boxes with class labels
- Confidence scores shown

### 8. Summary Dashboard
- Single comprehensive view
- Overall metrics at top
- Per-class breakdowns
- Radar chart for quick assessment

## File Structure After Pipeline Completion

```
project/
├── runs/
│   └── surveillance/
│       └── train/
│           ├── weights/
│           │   ├── best.pt (BEST MODEL)
│           │   └── last.pt
│           ├── confusion_matrix.png
│           ├── PR_curve.png
│           ├── F1_curve.png
│           ├── P_curve.png
│           ├── R_curve.png
│           └── results.csv
│
├── evaluation_results/
│   ├── metrics.json (ALL METRICS)
│   ├── evaluation_report.txt (TEXT REPORT)
│   ├── overall_metrics.png
│   ├── per_class_metrics.png
│   ├── training_curves.png
│   ├── class_distribution.png
│   ├── performance_heatmap.png
│   ├── inference_samples.png
│   └── summary_dashboard.png
│
└── quick_predictions.png (if quick_start.py used)
```

## Typical Performance Timeline

```
Training Start    → 0h 00m
Epoch 25/100      → 0h 38m (mAP: 0.65)
Epoch 50/100      → 1h 17m (mAP: 0.78)
Epoch 75/100      → 1h 55m (mAP: 0.82)
Epoch 100/100     → 2h 34m (mAP: 0.85)
Evaluation        → 2h 37m
Visualization     → 2h 39m
Complete          → 2h 40m
```

## Success Criteria

✅ **Excellent Model** (Production Ready)
- mAP@0.5 > 0.85
- mAP@0.5:0.95 > 0.65
- All classes AP@0.5 > 0.80

✅ **Good Model** (Usable with monitoring)
- mAP@0.5 > 0.75
- mAP@0.5:0.95 > 0.55
- Most classes AP@0.5 > 0.70

⚠️ **Needs Improvement**
- mAP@0.5 < 0.70
- Large variance between classes
- Low recall on critical classes

❌ **Requires Retraining**
- mAP@0.5 < 0.60
- Any critical class AP@0.5 < 0.50
- Precision or Recall < 0.60
