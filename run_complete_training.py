"""
Complete Training and Analysis Pipeline
Master script for IEEE paper publication
Runs: Training -> Evaluation -> Visualization -> IEEE Analysis
"""

import os
import sys
from pathlib import Path
import argparse
import time

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")

def run_training(epochs=100, batch=16):
    """Execute training pipeline"""
    print_header("STEP 1/4: MODEL TRAINING")
    
    from train_pipeline import setup_training, train_model
    
    start_time = time.time()
    model = setup_training()
    results = train_model(model, epochs=epochs, batch=batch)
    elapsed = time.time() - start_time
    
    print(f"\n[SUCCESS] Training completed in {elapsed/60:.2f} minutes")
    return results

def run_evaluation():
    """Execute evaluation pipeline"""
    print_header("STEP 2/4: MODEL EVALUATION")
    
    from evaluate_model import ModelEvaluator
    
    start_time = time.time()
    evaluator = ModelEvaluator()
    metrics = evaluator.validate_model()
    metrics_dict = evaluator.extract_metrics(metrics)
    
    evaluator.print_metrics(metrics_dict)
    evaluator.save_metrics(metrics_dict)
    evaluator.plot_overall_metrics(metrics_dict)
    evaluator.plot_metrics_comparison(metrics_dict)
    evaluator.generate_report(metrics_dict)
    
    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Evaluation completed in {elapsed:.2f} seconds")
    return metrics_dict

def run_visualization():
    """Execute visualization pipeline"""
    print_header("STEP 3/4: RESULTS VISUALIZATION")
    
    from visualize_results import ResultVisualizer
    
    start_time = time.time()
    visualizer = ResultVisualizer()
    visualizer.plot_training_curves()
    visualizer.plot_class_distribution()
    visualizer.plot_comparison_chart()
    visualizer.plot_inference_samples()
    visualizer.create_summary_dashboard()
    
    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Visualization completed in {elapsed:.2f} seconds")

def run_ieee_analysis():
    """Execute IEEE paper analysis"""
    print_header("STEP 4/4: IEEE PAPER ANALYSIS")
    
    from ieee_paper_analysis import IEEEPaperAnalysis
    
    start_time = time.time()
    analyzer = IEEEPaperAnalysis()
    
    print("[INFO] Generating statistical summary...")
    analyzer.generate_statistical_summary()
    
    print("[INFO] Generating IEEE figures...")
    analyzer.plot_ieee_figure_1()
    analyzer.plot_ieee_figure_2()
    analyzer.plot_ieee_figure_3()
    analyzer.plot_ieee_figure_4()
    analyzer.plot_ieee_figure_5()
    
    print("[INFO] Generating LaTeX tables...")
    analyzer.generate_latex_table()
    
    print("[INFO] Generating comprehensive report...")
    analyzer.generate_complete_report()
    
    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] IEEE analysis completed in {elapsed:.2f} seconds")

def print_summary():
    """Print final summary"""
    print_header("PIPELINE COMPLETED SUCCESSFULLY!")
    
    print("Generated Files and Directories:\n")
    
    print("1. MODEL WEIGHTS:")
    print("   - runs/surveillance/train/weights/best.pt")
    print("   - runs/surveillance/train/weights/last.pt\n")
    
    print("2. EVALUATION RESULTS (evaluation_results/):")
    print("   - metrics.json")
    print("   - evaluation_report.txt")
    print("   - overall_metrics.png")
    print("   - per_class_metrics.png")
    print("   - training_curves.png")
    print("   - class_distribution.png")
    print("   - performance_heatmap.png")
    print("   - inference_samples.png")
    print("   - summary_dashboard.png\n")
    
    print("3. IEEE PAPER RESULTS (ieee_paper_results/):")
    print("   - figure_1_configuration.png")
    print("   - figure_2_training_convergence.png")
    print("   - figure_3_per_class_performance.png")
    print("   - figure_4_confusion_and_samples.png")
    print("   - figure_5_performance_summary.png")
    print("   - latex_tables.tex")
    print("   - statistical_summary.txt")
    print("   - ieee_paper_report.txt\n")
    
    print("4. TRAINING ARTIFACTS (runs/surveillance/train/):")
    print("   - confusion_matrix.png")
    print("   - confusion_matrix_normalized.png")
    print("   - PR_curve.png")
    print("   - F1_curve.png")
    print("   - P_curve.png")
    print("   - R_curve.png")
    print("   - results.csv\n")
    
    print("=" * 80)
    print("All results are publication-ready for IEEE paper submission!")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(
        description='Complete Training and Analysis Pipeline for IEEE Paper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline with default settings
  python run_complete_training.py
  
  # Run with custom epochs and batch size
  python run_complete_training.py --epochs 50 --batch 8
  
  # Skip training (use existing model)
  python run_complete_training.py --skip-training
  
  # Run only specific steps
  python run_complete_training.py --only-eval
  python run_complete_training.py --only-viz
  python run_complete_training.py --only-ieee
        """
    )
    
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size for training (default: 16)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training step (use existing model)')
    parser.add_argument('--only-eval', action='store_true',
                       help='Run only evaluation')
    parser.add_argument('--only-viz', action='store_true',
                       help='Run only visualization')
    parser.add_argument('--only-ieee', action='store_true',
                       help='Run only IEEE analysis')
    
    args = parser.parse_args()
    
    print_header("SURVEILLANCE SYSTEM - COMPLETE TRAINING PIPELINE")
    print(f"Configuration:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch Size: {args.batch}")
    print(f"  - Skip Training: {args.skip_training}")
    
    try:
        total_start = time.time()
        
        # Handle specific step execution
        if args.only_eval:
            run_evaluation()
        elif args.only_viz:
            run_visualization()
        elif args.only_ieee:
            run_ieee_analysis()
        else:
            # Run complete pipeline
            if not args.skip_training:
                run_training(epochs=args.epochs, batch=args.batch)
            else:
                print_header("SKIPPING TRAINING (Using existing model)")
            
            run_evaluation()
            run_visualization()
            run_ieee_analysis()
        
        total_elapsed = time.time() - total_start
        
        print_summary()
        print(f"\nTotal execution time: {total_elapsed/60:.2f} minutes")
        
    except KeyboardInterrupt:
        print("\n\n[WARNING] Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
