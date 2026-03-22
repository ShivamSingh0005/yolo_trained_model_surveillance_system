"""
Generate comprehensive report and visualizations for 4-class model
"""

from ultralytics import YOLO
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def generate_comprehensive_report():
    """Generate complete report with all visualizations"""
    
    print("=" * 80)
    print("4-CLASS SURVEILLANCE SYSTEM - COMPREHENSIVE REPORT")
    print("=" * 80)
    
    # Check if model exists
    model_path = Path('runs/detect/runs/detect/4class_surveillance/weights/best.pt')
    results_csv = Path('runs/detect/runs/detect/4class_surveillance/results.csv')
    
    if not model_path.exists():
        print("\n⚠️  Model not found. Training may still be in progress.")
        return
    
    # Load model and evaluate
    model = YOLO(str(model_path))
    
    print("\nEvaluating 4-class model...")
    metrics = model.val(
        data='data_4class.yaml',
        split='test',
        save_json=True,
        plots=True
    )
    
    # Extract metrics
    results = {
        'model': 'YOLOv8n',
        'classes': 4,
        'class_names': ['Animal', 'Forest', 'Militant', 'UAV-Drone'],
        'mAP50': float(metrics.box.map50),
        'mAP50-95': float(metrics.box.map),
        'Precision': float(metrics.box.mp),
        'Recall': float(metrics.box.mr),
        'F1-Score': 2 * (float(metrics.box.mp) * float(metrics.box.mr)) / (float(metrics.box.mp) + float(metrics.box.mr) + 1e-6)
    }
    
    # Per-class
    class_names = ['Animal', 'Forest', 'Militant', 'UAV-Drone']
    results['per_class'] = {}
    
    for idx, class_name in enumerate(class_names):
        if idx < len(metrics.box.ap50):
            p = float(metrics.box.p[idx]) if hasattr(metrics.box, 'p') and idx < len(metrics.box.p) else 0.0
            r = float(metrics.box.r[idx]) if hasattr(metrics.box, 'r') and idx < len(metrics.box.r) else 0.0
            
            results['per_class'][class_name] = {
                'AP50': float(metrics.box.ap50[idx]),
                'AP': float(metrics.box.ap[idx]) if hasattr(metrics.box, 'ap') else 0.0,
                'Precision': p,
                'Recall': r,
            }
    
    # Save results
    output_dir = Path('4class_final_results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Generate visualizations
    generate_performance_plots(results, output_dir)
    
    if results_csv.exists():
        generate_training_curves(results_csv, output_dir)
    
    # Generate text report
    generate_text_report(results, output_dir)
    
    print(f"\n✓ All results saved to: {output_dir}/")
    
    return results


def generate_performance_plots(results, output_dir):
    """Generate performance visualization plots"""
    
    class_names = results['class_names']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Per-class AP50
    ap50_values = [results['per_class'][c]['AP50'] * 100 for c in class_names]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    bars = axes[0, 0].bar(class_names, ap50_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0, 0].axhline(y=80, color='green', linestyle='--', linewidth=2, label='80% Target')
    axes[0, 0].set_ylabel('AP50 (%)', fontsize=14, fontweight='bold')
    axes[0, 0].set_title('Per-Class AP50 Performance', fontsize=16, fontweight='bold')
    axes[0, 0].legend(fontsize=12)
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].set_ylim([0, 105])
    
    for bar in bars:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Plot 2: Precision vs Recall
    precision_values = [results['per_class'][c]['Precision'] * 100 for c in class_names]
    recall_values = [results['per_class'][c]['Recall'] * 100 for c in class_names]
    
    x = np.arange(len(class_names))
    width = 0.35
    
    bars1 = axes[0, 1].bar(x - width/2, precision_values, width, label='Precision', color='#3498db', alpha=0.8)
    bars2 = axes[0, 1].bar(x + width/2, recall_values, width, label='Recall', color='#e74c3c', alpha=0.8)
    
    axes[0, 1].set_ylabel('Score (%)', fontsize=14, fontweight='bold')
    axes[0, 1].set_title('Precision vs Recall', fontsize=16, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0, 1].legend(fontsize=12)
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].set_ylim([0, 105])
    
    # Plot 3: Overall metrics
    overall_metrics = ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1-Score']
    overall_values = [results[m] * 100 for m in overall_metrics]
    
    bars = axes[1, 0].barh(overall_metrics, overall_values, color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1, 0].axvline(x=80, color='green', linestyle='--', linewidth=2, label='80% Target')
    axes[1, 0].set_xlabel('Score (%)', fontsize=14, fontweight='bold')
    axes[1, 0].set_title('Overall Model Performance', fontsize=16, fontweight='bold')
    axes[1, 0].legend(fontsize=12)
    axes[1, 0].grid(axis='x', alpha=0.3)
    axes[1, 0].set_xlim([0, 105])
    
    for bar in bars:
        width = bar.get_width()
        axes[1, 0].text(width, bar.get_y() + bar.get_height()/2.,
                       f'{width:.1f}%',
                       ha='left', va='center', fontsize=12, fontweight='bold', color='darkgreen')
    
    # Plot 4: Performance summary table
    axes[1, 1].axis('off')
    
    table_data = []
    table_data.append(['Class', 'AP50', 'Precision', 'Recall'])
    for c in class_names:
        cls = results['per_class'][c]
        table_data.append([
            c,
            f"{cls['AP50']*100:.1f}%",
            f"{cls['Precision']*100:.1f}%",
            f"{cls['Recall']*100:.1f}%"
        ])
    table_data.append(['', '', '', ''])
    table_data.append(['Overall', f"{results['mAP50']*100:.1f}%", 
                      f"{results['Precision']*100:.1f}%", f"{results['Recall']*100:.1f}%"])
    
    table = axes[1, 1].table(cellText=table_data, cellLoc='center', loc='center',
                            colWidths=[0.3, 0.2, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style overall row
    for i in range(4):
        table[(len(table_data)-1, i)].set_facecolor('#2ecc71')
        table[(len(table_data)-1, i)].set_text_props(weight='bold')
    
    axes[1, 1].set_title('Performance Summary', fontsize=16, fontweight='bold', pad=20)
    
    plt.suptitle('4-Class Surveillance System Performance', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    plt.savefig(output_dir / 'performance_summary.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/performance_summary.png")
    plt.close()


def generate_training_curves(results_csv, output_dir):
    """Generate training progress curves"""
    
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    epochs = df['epoch'] + 1
    
    # mAP curves
    axes[0, 0].plot(epochs, df['metrics/mAP50(B)'], 'b-', linewidth=2.5, label='mAP50')
    axes[0, 0].plot(epochs, df['metrics/mAP50-95(B)'], 'r-', linewidth=2.5, label='mAP50-95')
    axes[0, 0].axhline(y=0.8, color='g', linestyle='--', linewidth=2, label='80% Target')
    axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('mAP', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('mAP Progress', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss curves
    if 'train/box_loss' in df.columns:
        axes[0, 1].plot(epochs, df['train/box_loss'], 'b-', linewidth=2, label='Box Loss')
        axes[0, 1].plot(epochs, df['train/cls_loss'], 'r-', linewidth=2, label='Cls Loss')
        axes[0, 1].plot(epochs, df['train/dfl_loss'], 'g-', linewidth=2, label='DFL Loss')
        axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Loss', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Training Losses', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)
    
    # Precision & Recall
    if 'metrics/precision(B)' in df.columns:
        axes[1, 0].plot(epochs, df['metrics/precision(B)'], 'b-', linewidth=2.5, label='Precision')
        axes[1, 0].plot(epochs, df['metrics/recall(B)'], 'r-', linewidth=2.5, label='Recall')
        axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Score', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('Precision & Recall', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate
    if 'lr/pg0' in df.columns:
        axes[1, 1].plot(epochs, df['lr/pg0'], 'b-', linewidth=2.5)
        axes[1, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Training Progress - 4-Class Model', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/training_curves.png")
    plt.close()


def generate_text_report(results, output_dir):
    """Generate comprehensive text report"""
    
    report = []
    report.append("=" * 80)
    report.append("4-CLASS SURVEILLANCE SYSTEM - FINAL REPORT")
    report.append("=" * 80)
    report.append("")
    report.append(f"Model: {results['model']}")
    report.append(f"Classes: {', '.join(results['class_names'])}")
    report.append(f"Total Classes: {results['classes']}")
    report.append("")
    
    report.append("OVERALL PERFORMANCE")
    report.append("-" * 80)
    report.append(f"  mAP@0.5: {results['mAP50']*100:.2f}%")
    report.append(f"  mAP@0.5:0.95: {results['mAP50-95']*100:.2f}%")
    report.append(f"  Precision: {results['Precision']*100:.2f}%")
    report.append(f"  Recall: {results['Recall']*100:.2f}%")
    report.append(f"  F1-Score: {results['F1-Score']*100:.2f}%")
    report.append("")
    
    report.append("PER-CLASS PERFORMANCE")
    report.append("-" * 80)
    report.append(f"{'Class':<15} {'AP50':<12} {'Precision':<12} {'Recall':<12}")
    report.append("-" * 80)
    
    for class_name in results['class_names']:
        cls = results['per_class'][class_name]
        report.append(f"{class_name:<15} {cls['AP50']*100:>6.1f}%{'':<5} {cls['Precision']*100:>6.1f}%{'':<5} {cls['Recall']*100:>6.1f}%")
    
    report.append("")
    report.append("=" * 80)
    report.append("ACHIEVEMENT STATUS")
    report.append("=" * 80)
    
    if results['mAP50'] >= 0.80:
        report.append("\n🎉 SUCCESS! Overall mAP50 >= 80%")
    
    all_above_80 = all(results['per_class'][c]['AP50'] >= 0.80 for c in results['class_names'])
    if all_above_80:
        report.append("🎉 EXCELLENT! All classes achieved >= 80% AP50")
    
    report.append("")
    report.append("=" * 80)
    
    report_text = "\n".join(report)
    
    with open(output_dir / 'final_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✓ Saved: {output_dir}/final_report.txt")
    print("\n" + report_text)


if __name__ == "__main__":
    results = generate_comprehensive_report()
