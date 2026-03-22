"""
Complete End-to-End Pipeline
Surveillance System - Training, Evaluation, and Visualization
"""

import os
import sys
from pathlib import Path
import argparse

def run_training(epochs=100, batch=16):
    """Execute training pipeline"""
    print("\n" + "=" * 70)
    print("STEP 1: MODEL TRAINING")
    print("=" * 70)
    
    from train_pipeline import setup_training, train_model
    
    model = setup_training()
    results = train_model(model, epochs=epochs, batch=batch)
    
    print("\n[SUCCESS] Training completed!")
    return results

def run_evaluation():
    """Execute evaluation pipeline"""
    print("\n" + "=" * 70)
    print("STEP 2: MODEL EVALUATION")
    print("=" * 70)
    
    from evaluate_model import ModelEvaluator
    
    evaluator = ModelEvaluator()
    metrics = evaluator.validate_model()
    metrics_dict = evaluator.extract_metrics(metrics)
    
    evaluator.print_metrics(metrics_dict)
    evaluator.save_metrics(metrics_dict)
    evaluator.plot_overall_metrics(metrics_dict)
    evaluator.plot_metrics_comparison(metrics_dict)
    evaluator.generate_report(metrics_dict)
    
    print("\n[SUCCESS] Evaluation completed!")
    return metrics_dict

def run_visualization():
    """Execute visualization pipeline"""
    print("\n" + "=" * 70)
    print("STEP 3: RESULTS VISUALIZATION")
    print("=" * 70)
    
    from visualize_results import ResultVisualizer
    
    visualizer = ResultVisualizer()
    visualizer.plot_training_curves()
    visualizer.plot_class_distribution()
    visualizer.plot_comparison_chart()
    visualizer.plot_inference_samples()
    visualizer.create_summary_dashboard()
    
    print("\n[SUCCESS] Visualization completed!")

def main():
    parser = argparse.ArgumentParser(description='Surveillance System - Complete Pipeline')
    parser.add_argument('--mode', type=str, default='all', 
                       choices=['train', 'eval', 'viz', 'all'],
                       help='Pipeline mode: train, eval, viz, or all')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size for training')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SURVEILLANCE SYSTEM - COMPLETE PIPELINE")
    print("=" * 70)
    print(f"Mode: {args.mode.upper()}")
    if args.mode in ['train', 'all']:
        print(f"Epochs: {args.epochs}")
        print(f"Batch Size: {args.batch}")
    print("=" * 70)
    
    try:
        if args.mode == 'train':
            run_training(epochs=args.epochs, batch=args.batch)
        
        elif args.mode == 'eval':
            run_evaluation()
        
        elif args.mode == 'viz':
            run_visualization()
        
        elif args.mode == 'all':
            # Run complete pipeline
            run_training(epochs=args.epochs, batch=args.batch)
            run_evaluation()
            run_visualization()
        
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nGenerated Files:")
        print("  - Model: runs/surveillance/train/weights/best.pt")
        print("  - Metrics: evaluation_results/metrics.json")
        print("  - Report: evaluation_results/evaluation_report.txt")
        print("  - Visualizations: evaluation_results/*.png")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
