"""
Advanced Visualization Module
Creates comprehensive graphs and analysis plots
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import json
import cv2
from ultralytics import YOLO
import random

class ResultVisualizer:
    def __init__(self, results_dir='evaluation_results'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.class_names = ['Animal', 'Forest', 'Militant', 'UAV-Drone', 'Wildfire']
        sns.set_style("whitegrid")
        
    def plot_training_curves(self, csv_path='runs/surveillance/train/results.csv'):
        """Plot training and validation curves"""
        if not Path(csv_path).exists():
            print(f"[WARNING] Training results CSV not found at {csv_path}")
            return
        
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
        # Box Loss
        if 'train/box_loss' in df.columns:
            axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train', linewidth=2)
            if 'val/box_loss' in df.columns:
                axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val', linewidth=2)
            axes[0, 0].set_title('Box Loss', fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Class Loss
        if 'train/cls_loss' in df.columns:
            axes[0, 1].plot(df['epoch'], df['train/cls_loss'], label='Train', linewidth=2)
            if 'val/cls_loss' in df.columns:
                axes[0, 1].plot(df['epoch'], df['val/cls_loss'], label='Val', linewidth=2)
            axes[0, 1].set_title('Classification Loss', fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # DFL Loss
        if 'train/dfl_loss' in df.columns:
            axes[0, 2].plot(df['epoch'], df['train/dfl_loss'], label='Train', linewidth=2)
            if 'val/dfl_loss' in df.columns:
                axes[0, 2].plot(df['epoch'], df['val/dfl_loss'], label='Val', linewidth=2)
            axes[0, 2].set_title('DFL Loss', fontweight='bold')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Loss')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Precision
        if 'metrics/precision(B)' in df.columns:
            axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], 
                          color='green', linewidth=2, label='Precision')
            axes[1, 0].set_title('Precision', fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].set_ylim([0, 1])
            axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        if 'metrics/recall(B)' in df.columns:
            axes[1, 1].plot(df['epoch'], df['metrics/recall(B)'], 
                          color='blue', linewidth=2, label='Recall')
            axes[1, 1].set_title('Recall', fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].set_ylim([0, 1])
            axes[1, 1].grid(True, alpha=0.3)
        
        # mAP@0.5
        if 'metrics/mAP50(B)' in df.columns:
            axes[1, 2].plot(df['epoch'], df['metrics/mAP50(B)'], 
                          color='red', linewidth=2, label='mAP@0.5')
            if 'metrics/mAP50-95(B)' in df.columns:
                axes[1, 2].plot(df['epoch'], df['metrics/mAP50-95(B)'], 
                              color='orange', linewidth=2, label='mAP@0.5:0.95')
            axes[1, 2].set_title('mAP Metrics', fontweight='bold')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('mAP')
            axes[1, 2].set_ylim([0, 1])
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.results_dir / 'training_curves.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Training curves saved to {output_path}")
        plt.close()
    
    def plot_class_distribution(self):
        """Plot class distribution in dataset"""
        train_labels_dir = Path('train/labels')
        test_labels_dir = Path('test/labels')
        
        class_counts_train = {name: 0 for name in self.class_names}
        class_counts_test = {name: 0 for name in self.class_names}
        
        # Count training labels
        for label_file in train_labels_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    class_id = int(line.split()[0])
                    class_counts_train[self.class_names[class_id]] += 1
        
        # Count test labels
        for label_file in test_labels_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    class_id = int(line.split()[0])
                    class_counts_test[self.class_names[class_id]] += 1
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Training distribution
        classes = list(class_counts_train.keys())
        counts_train = list(class_counts_train.values())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        
        bars1 = ax1.bar(classes, counts_train, color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_title('Training Set - Class Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Class', fontsize=12)
        ax1.set_ylabel('Number of Instances', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar, count in zip(bars1, counts_train):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}', ha='center', va='bottom', fontweight='bold')
        
        # Test distribution
        counts_test = list(class_counts_test.values())
        bars2 = ax2.bar(classes, counts_test, color=colors, edgecolor='black', linewidth=1.5)
        ax2.set_title('Test Set - Class Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Class', fontsize=12)
        ax2.set_ylabel('Number of Instances', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar, count in zip(bars2, counts_test):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        output_path = self.results_dir / 'class_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Class distribution plot saved to {output_path}")
        plt.close()
        
        return class_counts_train, class_counts_test
    
    def plot_comparison_chart(self, metrics_path='evaluation_results/metrics.json'):
        """Create comparison chart for all metrics"""
        if not Path(metrics_path).exists():
            print(f"[WARNING] Metrics file not found at {metrics_path}")
            return
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Prepare data for heatmap
        classes = list(metrics['per_class'].keys())
        metric_names = ['AP@0.5', 'AP@0.5:0.95', 'Precision', 'Recall']
        
        data = []
        for class_name in classes:
            row = [
                metrics['per_class'][class_name]['AP50'],
                metrics['per_class'][class_name]['AP'],
                metrics['per_class'][class_name]['Precision'],
                metrics['per_class'][class_name]['Recall']
            ]
            data.append(row)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(np.arange(len(metric_names)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(metric_names)
        ax.set_yticklabels(classes)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(len(classes)):
            for j in range(len(metric_names)):
                text = ax.text(j, i, f'{data[i][j]:.3f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('Performance Heatmap - All Classes', fontsize=14, fontweight='bold', pad=20)
        fig.colorbar(im, ax=ax, label='Score')
        
        plt.tight_layout()
        output_path = self.results_dir / 'performance_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Performance heatmap saved to {output_path}")
        plt.close()
    
    def plot_inference_samples(self, model_path='runs/surveillance/train/weights/best.pt', 
                              num_samples=6):
        """Run inference on sample images and visualize"""
        model = YOLO(model_path)
        test_images_dir = Path('test/images')
        
        # Get random sample images
        all_images = list(test_images_dir.glob('*.jpg'))
        sample_images = random.sample(all_images, min(num_samples, len(all_images)))
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, img_path in enumerate(sample_images):
            # Run inference
            results = model(img_path, verbose=False)
            
            # Get annotated image
            annotated = results[0].plot()
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            
            axes[idx].imshow(annotated_rgb)
            axes[idx].axis('off')
            axes[idx].set_title(f'Sample {idx+1}: {img_path.name[:30]}...', 
                              fontsize=10, fontweight='bold')
        
        plt.suptitle('Inference Results on Test Samples', fontsize=16, fontweight='bold')
        plt.tight_layout()
        output_path = self.results_dir / 'inference_samples.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Inference samples saved to {output_path}")
        plt.close()
    
    def create_summary_dashboard(self, metrics_path='evaluation_results/metrics.json'):
        """Create a comprehensive summary dashboard"""
        if not Path(metrics_path).exists():
            print(f"[WARNING] Metrics file not found")
            return
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Overall metrics
        ax1 = fig.add_subplot(gs[0, :])
        overall_metrics = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall', 'F1-Score']
        overall_values = [metrics['mAP50'], metrics['mAP50-95'], 
                         metrics['Precision'], metrics['Recall'], metrics['F1-Score']]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        bars = ax1.barh(overall_metrics, overall_values, color=colors, edgecolor='black', linewidth=2)
        ax1.set_xlim([0, 1])
        ax1.set_title('Overall Model Performance', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Score', fontsize=12)
        for bar, value in zip(bars, overall_values):
            width = bar.get_width()
            ax1.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{value:.3f}', ha='left', va='center', fontweight='bold', fontsize=11)
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Per-class AP@0.5
        ax2 = fig.add_subplot(gs[1, 0])
        classes = list(metrics['per_class'].keys())
        ap50_values = [metrics['per_class'][c]['AP50'] for c in classes]
        ax2.bar(range(len(classes)), ap50_values, color='skyblue', edgecolor='black')
        ax2.set_xticks(range(len(classes)))
        ax2.set_xticklabels(classes, rotation=45, ha='right')
        ax2.set_title('AP@0.5 by Class', fontweight='bold')
        ax2.set_ylabel('Score')
        ax2.set_ylim([0, 1])
        ax2.grid(axis='y', alpha=0.3)
        
        # Per-class Precision
        ax3 = fig.add_subplot(gs[1, 1])
        precision_values = [metrics['per_class'][c]['Precision'] for c in classes]
        ax3.bar(range(len(classes)), precision_values, color='lightgreen', edgecolor='black')
        ax3.set_xticks(range(len(classes)))
        ax3.set_xticklabels(classes, rotation=45, ha='right')
        ax3.set_title('Precision by Class', fontweight='bold')
        ax3.set_ylabel('Score')
        ax3.set_ylim([0, 1])
        ax3.grid(axis='y', alpha=0.3)
        
        # Per-class Recall
        ax4 = fig.add_subplot(gs[1, 2])
        recall_values = [metrics['per_class'][c]['Recall'] for c in classes]
        ax4.bar(range(len(classes)), recall_values, color='lightcoral', edgecolor='black')
        ax4.set_xticks(range(len(classes)))
        ax4.set_xticklabels(classes, rotation=45, ha='right')
        ax4.set_title('Recall by Class', fontweight='bold')
        ax4.set_ylabel('Score')
        ax4.set_ylim([0, 1])
        ax4.grid(axis='y', alpha=0.3)
        
        # Radar chart
        ax5 = fig.add_subplot(gs[2, :], projection='polar')
        angles = np.linspace(0, 2 * np.pi, len(overall_metrics), endpoint=False).tolist()
        values_radar = overall_values + [overall_values[0]]
        angles += angles[:1]
        ax5.plot(angles, values_radar, 'o-', linewidth=3, color='#FF6B6B', markersize=8)
        ax5.fill(angles, values_radar, alpha=0.25, color='#FF6B6B')
        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(overall_metrics, size=11)
        ax5.set_ylim(0, 1)
        ax5.set_title('Performance Radar', fontsize=14, fontweight='bold', pad=20)
        ax5.grid(True)
        
        plt.suptitle('Surveillance System - Performance Dashboard', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        output_path = self.results_dir / 'summary_dashboard.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Summary dashboard saved to {output_path}")
        plt.close()

def main():
    print("=" * 60)
    print("Results Visualization Pipeline")
    print("=" * 60)
    
    visualizer = ResultVisualizer()
    
    print("\n[INFO] Generating visualizations...")
    
    # Training curves
    visualizer.plot_training_curves()
    
    # Class distribution
    visualizer.plot_class_distribution()
    
    # Performance heatmap
    visualizer.plot_comparison_chart()
    
    # Inference samples
    visualizer.plot_inference_samples()
    
    # Summary dashboard
    visualizer.create_summary_dashboard()
    
    print("\n[SUCCESS] All visualizations generated!")
    print(f"[INFO] Results saved in: {visualizer.results_dir}")

if __name__ == "__main__":
    main()
