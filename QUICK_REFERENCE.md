# Quick Reference Guide

## One-Command Training

```bash
# Complete pipeline (training + evaluation + visualization + IEEE analysis)
python run_complete_training.py
```

## Step-by-Step Commands

```bash
# 1. Check environment
python check_environment.py

# 2. Train model
python train_pipeline.py

# 3. Evaluate model
python evaluate_model.py

# 4. Generate visualizations
python visualize_results.py

# 5. Generate IEEE paper analysis
python ieee_paper_analysis.py
```

## Quick Testing (10 epochs)

```bash
python quick_start.py train
python quick_start.py metrics
python quick_start.py predict
```

## Custom Training Parameters

```bash
# Custom epochs and batch size
python run_complete_training.py --epochs 50 --batch 8

# Skip training (use existing model)
python run_complete_training.py --skip-training

# Run only specific steps
python run_complete_training.py --only-eval
python run_complete_training.py --only-viz
python run_complete_training.py --only-ieee
```

## Key Output Files

### Model Weights
- `runs/surveillance/train/weights/best.pt` - Best model
- `runs/surveillance/train/weights/last.pt` - Last epoch

### Evaluation Results
- `evaluation_results/metrics.json` - All metrics
- `evaluation_results/evaluation_report.txt` - Detailed report
- `evaluation_results/*.png` - Visualization plots

### IEEE Paper Results
- `ieee_paper_results/figure_*.png` - Publication figures (300 DPI)
- `ieee_paper_results/latex_tables.tex` - LaTeX tables
- `ieee_paper_results/statistical_summary.txt` - Statistics
- `ieee_paper_results/ieee_paper_report.txt` - Full report

## Performance Metrics

### Overall Metrics
- **mAP@0.5**: Detection accuracy at 50% overlap
- **mAP@0.5:0.95**: Averaged over multiple IoU thresholds
- **Precision**: Correct detections / All detections
- **Recall**: Correct detections / All ground truth
- **F1-Score**: Harmonic mean of precision and recall

### Per-Class Metrics
- Individual metrics for each class:
  - Animal
  - Forest
  - Militant
  - UAV-Drone
  - Wildfire

## Troubleshooting

### CUDA Out of Memory
```bash
python run_complete_training.py --batch 8
```

### Slow Training
```bash
# Reduce epochs for testing
python run_complete_training.py --epochs 10
```

### Missing Dependencies
```bash
pip install -r requirements.txt --upgrade
```

## File Structure

```
yolo_trained_model_surveillance_system-main/
├── data.yaml                          # Dataset configuration
├── requirements.txt                   # Python dependencies
├── check_environment.py               # Environment checker
├── run_complete_training.py           # Master script
├── train_pipeline.py                  # Training script
├── evaluate_model.py                  # Evaluation script
├── visualize_results.py               # Visualization script
├── ieee_paper_analysis.py             # IEEE analysis script
├── TRAINING_GUIDE.md                  # Detailed guide
├── QUICK_REFERENCE.md                 # This file
│
├── train/                             # Training data
│   ├── images/                        # 646 images
│   └── labels/                        # 646 labels
│
├── test/                              # Test data
│   ├── images/                        # 114 images
│   └── labels/                        # 114 labels
│
├── runs/surveillance/train/           # Training outputs
│   ├── weights/
│   │   ├── best.pt
│   │   └── last.pt
│   └── *.png, *.csv                   # Metrics and plots
│
├── evaluation_results/                # Evaluation outputs
│   ├── metrics.json
│   ├── evaluation_report.txt
│   └── *.png                          # Visualization plots
│
└── ieee_paper_results/                # IEEE paper outputs
    ├── figure_*.png                   # Publication figures
    ├── latex_tables.tex               # LaTeX tables
    ├── statistical_summary.txt        # Statistics
    └── ieee_paper_report.txt          # Full report
```

## Expected Training Time

- **With GPU (NVIDIA RTX 3080)**: 30-60 minutes
- **With GPU (NVIDIA GTX 1080)**: 60-90 minutes
- **With CPU**: 2-4 hours

## Model Specifications

- **Architecture**: YOLOv8n (nano)
- **Input Size**: 640×640
- **Parameters**: ~3.2M
- **Inference Speed**: ~10ms per image (GPU)
- **Model Size**: ~6MB

## Dataset Statistics

- **Total Images**: 852
- **Training**: 646 images
- **Validation**: 92 images
- **Test**: 114 images
- **Classes**: 5 (Animal, Forest, Militant, UAV-Drone, Wildfire)

## IEEE Paper Figures

All figures are generated at 300 DPI for publication:

1. **Figure 1**: Experimental setup and dataset configuration
2. **Figure 2**: Training convergence and metrics evolution
3. **Figure 3**: Per-class performance analysis
4. **Figure 4**: Confusion matrix and detection samples
5. **Figure 5**: Comprehensive performance summary

## LaTeX Integration

```latex
% Include figure
\begin{figure}[htbp]
\centering
\includegraphics[width=\columnwidth]{ieee_paper_results/figure_1_configuration.png}
\caption{Experimental Setup and Dataset Configuration}
\label{fig:configuration}
\end{figure}

% Include table (copy from latex_tables.tex)
\begin{table}[htbp]
\caption{Overall Model Performance Metrics}
\label{tab:overall_performance}
\centering
\begin{tabular}{lc}
\hline
\textbf{Metric} & \textbf{Score} \\
\hline
% ... table content from latex_tables.tex
\end{tabular}
\end{table}
```

## Common Commands Summary

| Task | Command |
|------|---------|
| Check environment | `python check_environment.py` |
| Complete pipeline | `python run_complete_training.py` |
| Train only | `python train_pipeline.py` |
| Evaluate only | `python evaluate_model.py` |
| Visualize only | `python visualize_results.py` |
| IEEE analysis only | `python ieee_paper_analysis.py` |
| Quick test | `python quick_start.py train` |
| Custom training | `python run_complete_training.py --epochs 50 --batch 8` |

## Getting Help

For detailed information, see:
- `TRAINING_GUIDE.md` - Comprehensive training guide
- `README.md` - Project overview
- Console output - Detailed progress and error messages

## Tips for Best Results

1. **Use GPU**: Significantly faster training
2. **More epochs**: Better accuracy (100-150 epochs recommended)
3. **Larger batch**: Faster training if GPU memory allows
4. **Check results**: Review metrics.json and plots after training
5. **Iterate**: Adjust hyperparameters based on results

## Citation

```bibtex
@article{surveillance_yolov8_2024,
  title={Surveillance System Object Detection using YOLOv8},
  author={Your Name},
  journal={IEEE Conference/Journal},
  year={2024}
}
```

---

**Quick Start**: `python run_complete_training.py`

**Documentation**: See `TRAINING_GUIDE.md` for detailed instructions

**Support**: Check console output for errors and solutions
