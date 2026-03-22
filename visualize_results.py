"""
Results Visualization for IEEE Paper Publication
Generates publication-quality figures and analysis
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cv2
import json

class ResultVisualizer:
    def __init__(self):
        """Initialize visualizer"""
        self.results_dir = Path('evaluation_results')
        self.results_dir.mkdir(exist_ok=True)
        self.train_dir = Path('runs/surveillance/train')
        self.class_names = ['Animal', 'Forest', 'Militant', 'UAV-Drone', 'Wildfire']
        
        # Set publication-quality style
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")
        
    def plot_training_curves(self):
        """Plot training and validation curves"""
        results_csv = self.train_dir / 'results.csv'
        
        if not results_csv.exists():
            print("[WARNING] Training results CSV not found")
            return
        
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Training Progress and Metrics Evolution', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(df) + 1)
        
        # Box Loss
        if 'train/box_loss' in df.columns:
            axes[0, 0].plot(epochs, df['train/box_loss'], label='Train', linewidth=2)
            axes[0, 0].plot(epochs, df['val/box_loss'], label='Val', linewidth=2)
            axes[0, 0].set_title('Box Loss', fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Class Loss
        if 'train/cls_loss' in df.columns:
            axes[0, 1].plot(epochs, df['train/cls_loss'], label='Train', linewidth=2)
            axes[0, 1].plot(epochs, df['val/cls_loss'], label='Val', linewidth=2)
            axes[0, 1].set_title('Classification Loss', fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # DFL Loss
        if 'train/dfl_loss' in df.columns:
            axes[0, 2].plot(epochs, df['train/dfl_loss'], label='Train', linewidth=2)
            axes[0, 2].plot(epochs, df['val/dfl_loss'], label='Val', linewidth=2)
            axes[0, 2].set_title('DFL Loss', fontweight='bold')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Loss')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Precision
        if 'metrics/precision(B)' in df.columns:
            axes[1, 0].plot(epochs, df['metrics/precision(B)'], linewidth=2, color='green')
            axes[1, 0].set_title('Precision', fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_ylim([0, 1])
        
        # Recall
        if 'metrics/recall(B)' in df.columns:
            axes[1, 1].plot(epochs, df['metrics/recall(B)'], linewidth=2, color='orange')
            axes[1, 1].set_title('Recall', fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim([0, 1])
        
        # mAP@0.5
        if 'metrics/mAP50(B)' in df.columns:
            axes[1, 2].plot(epochs, df['metrics/mAP50(B)'], linewidth=2, color='red')
            axes[1, 2].set_title('mAP@0.5', fontweight='bold')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('mAP@0.5')
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].set_ylim([0, 1])
        
        plt.tight_layout()
        output_path = self.results_dir / 'training_curves.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Training curves saved to {output_path}")
        plt.close()
    
    def plot_class_distribution(self):
        """Plot dataset class distribution"""
        # Count labels in training set
        train_labels_dir = Path('train/labels')
        
        if not train_labels_dir.exists():
            print("[WARNING] Training labels directory not found")
            return
        
        class_counts = {name: 0 for name in self.class_names}
        
        for label_file in train_labels_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    class_id = int(line.split()[0])
                    if class_id < len(self.class_names):
                        class_counts[self.class_names[class_id]] += 1
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar chart
        colors = plt.cm.Set3(range(len(self.class_names)))
        bars = ax1.bar(self.class_names, class_counts.values(), color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_title('Training Dataset Class Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Class', fontsize=12)
        ax1.set_ylabel('Number of Instances', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        ax2.pie(class_counts.values(), labels=self.class_names, autopct='%1.1f%%',
               colors=colors, startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
        ax2.set_title('Class Distribution Percentage', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        output_path = self.results_dir / 'class_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Class distribution plot saved to {output_path}")
        plt.close()
    
    def plot_comparison_chart(self):
        """Create performance heatmap"""
        metrics_file = self.results_dir / 'metrics.json'
        
        if not metrics_file.exists():
            print("[WARNING] Metrics file not found")
            return
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Create heatmap data
        data = []
        for class_name in self.class_names:
            class_metrics = metrics['per_class'][class_name]
            data.append([
                class_metrics['AP50'],
                class_metrics['AP'],
                class_metrics['Precision'],
                class_metrics['Recall']
            ])
        
        df = pd.DataFrame(data, 
                         columns=['AP@0.5', 'AP@0.5:0.95', 'Precision', 'Recall'],
                         index=self.class_names)
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(df, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Score'},
                   linewidths=0.5, linecolor='gray', vmin=0, vmax=1)
        plt.title('Performance Heatmap Across Classes and Metrics', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Metrics', fontsize=12, fontweight='bold')
        plt.ylabel('Classes', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.results_dir / 'performance_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Performance heatmap saved to {output_path}")
        plt.close()
    
    def plot_inference_samples(self, num_samples=6):
        """Plot sample predictions on test images"""
        model_path = self.train_dir / 'weights' / 'best.pt'
        
        if not model_path.exists():
            print("[WARNING] Best model not found")
            return
        
        model = YOLO(str(model_path))
        test_images_dir = Path('test/images')
        
        if not test_images_dir.exists():
            print("[WARNING] Test images directory not found")
            return
        
        images = list(test_images_dir.glob('*.jpg'))[:num_samples]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Sample Predictions on Test Set', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        for idx, img_path in enumerate(images):
            results = model(str(img_path), verbose=False)
            annotated = results[0].plot()
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            
            axes[idx].imshow(annotated_rgb)
            axes[idx].axis('off')
            axes[idx].set_title(f'{img_path.stem[:30]}', fontsize=10)
        
        plt.tight_layout()
        output_path = self.results_dir / 'inference_samples.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Inference samples saved to {output_path}")
        plt.close()
    
    def create_summary_dashboard(self):
        """Create comprehensive summary dashboard"""
        metrics_file = self.results_dir / 'metrics.json'
        
        if not metrics_file.exists():
            print("[WARNING] Metrics file not found")
            return
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Overall metrics
        ax1 = fig.add_subplot(gs[0, :])
        overall_metrics = ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1-Score']
        overall_values = [metrics[m] for m in overall_metrics]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        bars = ax1.bar(overall_metrics, overall_values, color=colors, edgecolor='black', linewidth=2)
        ax1.set_title('Overall Model Performance Metrics', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_ylim([0, 1])
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        for bar, value in zip(bars, overall_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Per-class AP@0.5
        ax2 = fig.add_subplot(gs[1, 0])
        classes = list(metrics['per_class'].keys())
        ap50_values = [metrics['per_class'][c]['AP50'] for c in classes]
        ax2.barh(classes, ap50_values, color='skyblue', edgecolor='black')
        ax2.set_xlabel('AP@0.5', fontsize=10)
        ax2.set_title('Per-Class AP@0.5', fontsize=12, fontweight='bold')
        ax2.set_xlim([0, 1])
        for i, v in enumerate(ap50_values):
            ax2.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=9)
        
        # Per-class Precision
        ax3 = fig.add_subplot(gs[1, 1])
        precision_values = [metrics['per_class'][c]['Precision'] for c in classes]
        ax3.barh(classes, precision_values, color='lightgreen', edgecolor='black')
        ax3.set_xlabel('Precision', fontsize=10)
        ax3.set_title('Per-Class Precision', fontsize=12, fontweight='bold')
        ax3.set_xlim([0, 1])
        for i, v in enumerate(precision_values):
            ax3.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=9)
        
        # Per-class Recall
        ax4 = fig.add_subplot(gs[1, 2])
        recall_values = [metrics['per_class'][c]['Recall'] for c in classes]
        ax4.barh(classes, recall_values, color='lightcoral', edgecolor='black')
        ax4.set_xlabel('Recall', fontsize=10)
        ax4.set_title('Per-Class Recall', fontsize=12, fontweight='bold')
        ax4.set_xlim([0, 1])
        for i, v in enumerate(recall_values):
            ax4.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=9)
        
        # Radar chart
        ax5 = fig.add_subplot(gs[2, 0], projection='polar')
        angles = np.linspace(0, 2 * np.pi, len(overall_metrics), endpoint=False).tolist()
        values_radar = overall_values + [overall_values[0]]
        angles += angles[:1]
        ax5.plot(angles, values_radar, 'o-', linewidth=2, color='#FF6B6B')
        ax5.fill(angles, values_radar, alpha=0.25, color='#FF6B6B')
        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(overall_metrics, size=9)
        ax5.set_ylim(0, 1)
        ax5.set_title('Performance Radar', fontsize=12, fontweight='bold', pad=20)
        ax5.grid(True)
        
        # Heatmap
        ax6 = fig.add_subplot(gs[2, 1:])
        heatmap_data = []
        for class_name in classes:
            class_metrics = metrics['per_class'][class_name]
            heatmap_data.append([
                class_metrics['AP50'],
                class_metrics['AP'],
                class_metrics['Precision'],
                class_metrics['Recall']
            ])
        df_heatmap = pd.DataFrame(heatmap_data,
                                  columns=['AP@0.5', 'AP@0.5:0.95', 'Precision', 'Recall'],
                                  index=classes)
        sns.heatmap(df_heatmap, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax6,
                   cbar_kws={'label': 'Score'}, vmin=0, vmax=1, linewidths=0.5)
        ax6.set_title('Metrics Heatmap', fontsize=12, fontweight='bold')
        
        fig.suptitle('Surveillance System - Comprehensive Performance Dashboard',
                    fontsize=18, fontweight='bold', y=0.98)
        
        output_path = self.results_dir / 'summary_dashboard.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Summary dashboard saved to {output_path}")
        plt.close()

def main():
    """Main visualization execution"""
    print("=" * 70)
    print("Results Visualization Pipeline")
    print("=" * 70)
    
    visualizer = ResultVisualizer()
    
    print("\n[INFO] Generating visualizations...")
    visualizer.plot_training_curves()
    visualizer.plot_class_distribution()
    visualizer.plot_comparison_chart()
    visualizer.plot_inference_samples()
    visualizer.create_summary_dashboard()
    
    print("\n[SUCCESS] All visualizations generated!")
    print(f"[INFO] Results saved in: {visualizer.results_dir}")

if __name__ == "__main__":
    main()
