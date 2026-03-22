"""
Model Evaluation and Performance Metrics
Generates comprehensive metrics and visualizations
"""

import os
from pathlib import Path
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
import pandas as pd

class ModelEvaluator:
    def __init__(self, model_path='runs/detect/runs/surveillance/weights/best.pt'):
        """Initialize evaluator with trained model"""
        # Try alternate path if default doesn't exist
        if not Path(model_path).exists():
            alt_path = 'runs/surveillance/train/weights/best.pt'
            if Path(alt_path).exists():
                model_path = alt_path
        self.model = YOLO(model_path)
        self.class_names = ['Animal', 'Forest', 'Militant', 'UAV-Drone', 'Wildfire']
        self.results_dir = Path('evaluation_results')
        self.results_dir.mkdir(exist_ok=True)
        
    def validate_model(self, data_yaml='data.yaml'):
        """Run validation and get metrics"""
        print("[INFO] Running validation...")
        metrics = self.model.val(data=data_yaml, split='test', save_json=True, save_hybrid=True)
        
        return metrics
    
    def extract_metrics(self, metrics):
        """Extract key performance metrics"""
        results = {
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
            'Precision': float(metrics.box.mp),
            'Recall': float(metrics.box.mr),
            'F1-Score': 2 * (float(metrics.box.mp) * float(metrics.box.mr)) / 
                       (float(metrics.box.mp) + float(metrics.box.mr) + 1e-6)
        }
        
        # Per-class metrics
        per_class = {}
        for idx, class_name in enumerate(self.class_names):
            per_class[class_name] = {
                'AP50': float(metrics.box.ap50[idx]),
                'AP': float(metrics.box.ap[idx]),
                'Precision': float(metrics.box.p[idx]) if hasattr(metrics.box, 'p') else 0,
                'Recall': float(metrics.box.r[idx]) if hasattr(metrics.box, 'r') else 0
            }
        
        results['per_class'] = per_class
        
        return results
    
    def save_metrics(self, metrics_dict):
        """Save metrics to JSON file"""
        output_path = self.results_dir / 'metrics.json'
        with open(output_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        print(f"[INFO] Metrics saved to {output_path}")
        
    def print_metrics(self, metrics_dict):
        """Print formatted metrics"""
        print("\n" + "=" * 60)
        print("OVERALL PERFORMANCE METRICS")
        print("=" * 60)
        print(f"mAP@0.5      : {metrics_dict['mAP50']:.4f}")
        print(f"mAP@0.5:0.95 : {metrics_dict['mAP50-95']:.4f}")
        print(f"Precision    : {metrics_dict['Precision']:.4f}")
        print(f"Recall       : {metrics_dict['Recall']:.4f}")
        print(f"F1-Score     : {metrics_dict['F1-Score']:.4f}")
        
        print("\n" + "=" * 60)
        print("PER-CLASS METRICS")
        print("=" * 60)
        
        df_data = []
        for class_name, class_metrics in metrics_dict['per_class'].items():
            df_data.append({
                'Class': class_name,
                'AP@0.5': f"{class_metrics['AP50']:.4f}",
                'AP@0.5:0.95': f"{class_metrics['AP']:.4f}",
                'Precision': f"{class_metrics['Precision']:.4f}",
                'Recall': f"{class_metrics['Recall']:.4f}"
            })
        
        df = pd.DataFrame(df_data)
        print(df.to_string(index=False))
        
    def plot_metrics_comparison(self, metrics_dict):
        """Create bar plot comparing metrics across classes"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
        
        classes = list(metrics_dict['per_class'].keys())
        
        # AP@0.5
        ap50_values = [metrics_dict['per_class'][c]['AP50'] for c in classes]
        axes[0, 0].bar(classes, ap50_values, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Average Precision @ IoU=0.5', fontweight='bold')
        axes[0, 0].set_ylabel('AP@0.5')
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].tick_params(axis='x', rotation=45)
        for i, v in enumerate(ap50_values):
            axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        # AP@0.5:0.95
        ap_values = [metrics_dict['per_class'][c]['AP'] for c in classes]
        axes[0, 1].bar(classes, ap_values, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('Average Precision @ IoU=0.5:0.95', fontweight='bold')
        axes[0, 1].set_ylabel('AP@0.5:0.95')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].tick_params(axis='x', rotation=45)
        for i, v in enumerate(ap_values):
            axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        # Precision
        precision_values = [metrics_dict['per_class'][c]['Precision'] for c in classes]
        axes[1, 0].bar(classes, precision_values, color='lightgreen', edgecolor='black')
        axes[1, 0].set_title('Precision', fontweight='bold')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].tick_params(axis='x', rotation=45)
        for i, v in enumerate(precision_values):
            axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        # Recall
        recall_values = [metrics_dict['per_class'][c]['Recall'] for c in classes]
        axes[1, 1].bar(classes, recall_values, color='plum', edgecolor='black')
        axes[1, 1].set_title('Recall', fontweight='bold')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].tick_params(axis='x', rotation=45)
        for i, v in enumerate(recall_values):
            axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        output_path = self.results_dir / 'per_class_metrics.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Per-class metrics plot saved to {output_path}")
        plt.close()
    
    def plot_overall_metrics(self, metrics_dict):
        """Create overall metrics visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Overall metrics bar chart
        metrics_names = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall', 'F1-Score']
        metrics_values = [
            metrics_dict['mAP50'],
            metrics_dict['mAP50-95'],
            metrics_dict['Precision'],
            metrics_dict['Recall'],
            metrics_dict['F1-Score']
        ]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        bars = ax1.bar(metrics_names, metrics_values, color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_title('Overall Model Performance', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_ylim([0, 1])
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Radar chart for overall metrics
        angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
        metrics_values_radar = metrics_values + [metrics_values[0]]
        angles += angles[:1]
        
        ax2 = plt.subplot(122, projection='polar')
        ax2.plot(angles, metrics_values_radar, 'o-', linewidth=2, color='#FF6B6B')
        ax2.fill(angles, metrics_values_radar, alpha=0.25, color='#FF6B6B')
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(metrics_names, size=10)
        ax2.set_ylim(0, 1)
        ax2.set_title('Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
        ax2.grid(True)
        
        plt.tight_layout()
        output_path = self.results_dir / 'overall_metrics.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Overall metrics plot saved to {output_path}")
        plt.close()
    
    def plot_pr_curve(self):
        """Plot Precision-Recall curve"""
        # Check if PR curve exists from validation
        pr_curve_path = Path('runs/surveillance/train/PR_curve.png')
        if pr_curve_path.exists():
            print(f"[INFO] PR curve available at {pr_curve_path}")
        else:
            print("[INFO] PR curve not found. Run validation first.")
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix"""
        cm_path = Path('runs/surveillance/train/confusion_matrix.png')
        if cm_path.exists():
            print(f"[INFO] Confusion matrix available at {cm_path}")
        else:
            print("[INFO] Confusion matrix not found. Run validation first.")
    
    def generate_report(self, metrics_dict):
        """Generate comprehensive evaluation report"""
        report_path = self.results_dir / 'evaluation_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("SURVEILLANCE SYSTEM - MODEL EVALUATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("Dataset Information:\n")
            f.write("-" * 70 + "\n")
            f.write("Classes: Animal, Forest, Militant, UAV-Drone, Wildfire\n")
            f.write("Total Classes: 5\n")
            f.write("Training Images: 646\n")
            f.write("Test Images: 114\n\n")
            
            f.write("Overall Performance Metrics:\n")
            f.write("-" * 70 + "\n")
            f.write(f"mAP@0.5           : {metrics_dict['mAP50']:.4f}\n")
            f.write(f"mAP@0.5:0.95      : {metrics_dict['mAP50-95']:.4f}\n")
            f.write(f"Precision         : {metrics_dict['Precision']:.4f}\n")
            f.write(f"Recall            : {metrics_dict['Recall']:.4f}\n")
            f.write(f"F1-Score          : {metrics_dict['F1-Score']:.4f}\n\n")
            
            f.write("Per-Class Performance:\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Class':<15} {'AP@0.5':<10} {'AP@0.5:0.95':<15} {'Precision':<12} {'Recall':<10}\n")
            f.write("-" * 70 + "\n")
            
            for class_name, class_metrics in metrics_dict['per_class'].items():
                f.write(f"{class_name:<15} "
                       f"{class_metrics['AP50']:<10.4f} "
                       f"{class_metrics['AP']:<15.4f} "
                       f"{class_metrics['Precision']:<12.4f} "
                       f"{class_metrics['Recall']:<10.4f}\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        print(f"[INFO] Evaluation report saved to {report_path}")

def main():
    print("=" * 60)
    print("Model Evaluation Pipeline")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Run validation
    metrics = evaluator.validate_model()
    
    # Extract metrics
    metrics_dict = evaluator.extract_metrics(metrics)
    
    # Print metrics
    evaluator.print_metrics(metrics_dict)
    
    # Save metrics
    evaluator.save_metrics(metrics_dict)
    
    # Generate visualizations
    print("\n[INFO] Generating visualizations...")
    evaluator.plot_overall_metrics(metrics_dict)
    evaluator.plot_metrics_comparison(metrics_dict)
    
    # Generate report
    evaluator.generate_report(metrics_dict)
    
    print("\n[SUCCESS] Evaluation completed!")
    print(f"[INFO] All results saved in: {evaluator.results_dir}")

if __name__ == "__main__":
    main()
