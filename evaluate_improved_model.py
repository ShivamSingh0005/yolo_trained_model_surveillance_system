"""
Evaluate the improved wildfire model and generate comprehensive results
"""

from ultralytics import YOLO
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_improved_model():
    """Evaluate the improved model and compare with original"""
    
    print("=" * 80)
    print("EVALUATING IMPROVED WILDFIRE MODEL")
    print("=" * 80)
    
    # Load the best model
    model_path = 'runs/detect/runs/detect/wildfire_focused_v2/weights/best.pt'
    
    if not Path(model_path).exists():
        print(f"\n⚠️  Model not found: {model_path}")
        return
    
    print(f"\n✓ Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    metrics = model.val(
        data='data.yaml',
        split='test',
        save_json=True,
        plots=True,
        project='runs/detect',
        name='wildfire_improved_eval'
    )
    
    # Extract metrics
    results_dict = {
        'mAP50': float(metrics.box.map50),
        'mAP50-95': float(metrics.box.map),
        'Precision': float(metrics.box.mp),
        'Recall': float(metrics.box.mr),
        'F1-Score': 2 * (float(metrics.box.mp) * float(metrics.box.mr)) / (float(metrics.box.mp) + float(metrics.box.mr) + 1e-6)
    }
    
    # Per-class metrics
    class_names = ['Animal', 'Forest', 'Militant', 'UAV-Drone', 'Wildfire']
    results_dict['per_class'] = {}
    
    for idx, class_name in enumerate(class_names):
        if idx < len(metrics.box.ap50):
            # Calculate precision and recall for this class
            p = float(metrics.box.p[idx]) if hasattr(metrics.box, 'p') and idx < len(metrics.box.p) else 0.0
            r = float(metrics.box.r[idx]) if hasattr(metrics.box, 'r') and idx < len(metrics.box.r) else 0.0
            
            results_dict['per_class'][class_name] = {
                'AP50': float(metrics.box.ap50[idx]),
                'AP': float(metrics.box.ap[idx]) if hasattr(metrics.box, 'ap') else 0.0,
                'Precision': p,
                'Recall': r,
                'F1-Score': 2 * (p * r) / (p + r + 1e-6)
            }
    
    # Save results
    results_dir = Path('wildfire_improved_results')
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / 'improved_metrics.json', 'w') as f:
        json.dump(results_dict, f, indent=4)
    
    print(f"\n✓ Results saved to: {results_dir}/improved_metrics.json")
    
    # Compare with original
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    
    original_metrics_path = Path('evaluation_results/metrics.json')
    if original_metrics_path.exists():
        with open(original_metrics_path, 'r') as f:
            original = json.load(f)
        
        # Overall comparison
        print("\nOVERALL METRICS:")
        print("-" * 80)
        print(f"{'Metric':<20} {'Original':<15} {'Improved':<15} {'Change':<15}")
        print("-" * 80)
        
        for metric in ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1-Score']:
            old_val = original.get(metric, original.get(metric.replace('-', ''), 0)) * 100
            new_val = results_dict.get(metric, 0) * 100
            change = new_val - old_val
            
            status = "✓" if change > 0 else "✗" if change < 0 else "="
            change_str = f"{status} {change:+.1f}%"
            
            print(f"{metric:<20} {old_val:>6.1f}%{'':<8} {new_val:>6.1f}%{'':<8} {change_str:<15}")
        
        # Per-class comparison
        print("\n" + "=" * 80)
        print("PER-CLASS COMPARISON")
        print("=" * 80)
        
        for class_name in class_names:
            if class_name in results_dict['per_class'] and class_name in original['per_class']:
                new_cls = results_dict['per_class'][class_name]
                old_cls = original['per_class'][class_name]
                
                print(f"\n{class_name.upper()}:")
                print("-" * 80)
                print(f"{'Metric':<20} {'Original':<15} {'Improved':<15} {'Change':<15}")
                print("-" * 80)
                
                for metric in ['AP50', 'Precision', 'Recall']:
                    old_val = old_cls.get(metric, 0) * 100
                    new_val = new_cls.get(metric, 0) * 100
                    change = new_val - old_val
                    
                    status = "✓" if change > 0 else "✗" if change < 0 else "="
                    change_str = f"{status} {change:+.1f}%"
                    
                    print(f"{metric:<20} {old_val:>6.1f}%{'':<8} {new_val:>6.1f}%{'':<8} {change_str:<15}")
        
        # Wildfire specific analysis
        print("\n" + "=" * 80)
        print("WILDFIRE CLASS - DETAILED ANALYSIS")
        print("=" * 80)
        
        if 'Wildfire' in results_dict['per_class']:
            wf_new = results_dict['per_class']['Wildfire']
            wf_old = original['per_class']['Wildfire']
            
            print(f"\nOriginal Wildfire Performance:")
            print(f"  AP50: {wf_old['AP50']*100:.1f}%")
            print(f"  Precision: {wf_old['Precision']*100:.1f}%")
            print(f"  Recall: {wf_old['Recall']*100:.1f}%")
            
            print(f"\nImproved Wildfire Performance:")
            print(f"  AP50: {wf_new['AP50']*100:.1f}%")
            print(f"  Precision: {wf_new['Precision']*100:.1f}%")
            print(f"  Recall: {wf_new['Recall']*100:.1f}%")
            
            ap50_improvement = (wf_new['AP50'] - wf_old['AP50']) * 100
            
            print(f"\nImprovement: {ap50_improvement:+.1f}% AP50")
            
            if wf_new['AP50'] >= 0.80:
                print("\n🎉 TARGET ACHIEVED! Wildfire AP50 >= 80%")
            elif wf_new['AP50'] >= 0.70:
                print("\n✓ Good progress! Wildfire AP50 >= 70%")
            elif wf_new['AP50'] >= 0.60:
                print("\n✓ Moderate improvement. Consider additional training.")
            else:
                print("\n⚠️  Further improvement needed.")
    
    print("\n" + "=" * 80)
    
    # Generate comparison visualizations
    generate_comparison_plots(results_dict, original if original_metrics_path.exists() else None)
    
    return results_dict


def generate_comparison_plots(improved, original=None):
    """Generate comparison visualizations"""
    
    print("\nGenerating comparison visualizations...")
    
    results_dir = Path('wildfire_improved_results')
    
    if original is None:
        return
    
    # Prepare data
    class_names = ['Animal', 'Forest', 'Militant', 'UAV-Drone', 'Wildfire']
    metrics_to_plot = ['AP50', 'Precision', 'Recall']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, metric in enumerate(metrics_to_plot):
        original_vals = [original['per_class'][cls].get(metric, 0) * 100 for cls in class_names]
        improved_vals = [improved['per_class'][cls].get(metric, 0) * 100 for cls in class_names]
        
        x = range(len(class_names))
        width = 0.35
        
        bars1 = axes[idx].bar([i - width/2 for i in x], original_vals, width, 
                             label='Original', color='#3498db', alpha=0.8)
        bars2 = axes[idx].bar([i + width/2 for i in x], improved_vals, width,
                             label='Improved', color='#2ecc71', alpha=0.8)
        
        # Highlight Wildfire
        bars1[4].set_color('#e74c3c')
        bars1[4].set_alpha(0.6)
        bars2[4].set_color('#27ae60')
        bars2[4].set_edgecolor('darkgreen')
        bars2[4].set_linewidth(2)
        
        axes[idx].set_xlabel('Class', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel(f'{metric} (%)', fontsize=12, fontweight='bold')
        axes[idx].set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(class_names, rotation=45, ha='right')
        axes[idx].legend()
        axes[idx].grid(axis='y', alpha=0.3)
        axes[idx].set_ylim([0, 105])
        
        # Add value labels
        for bar in bars1 + bars2:
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.1f}',
                          ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Model Performance Comparison: Original vs Improved', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = results_dir / 'performance_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot: {output_path}")
    plt.close()


if __name__ == "__main__":
    results = evaluate_improved_model()
