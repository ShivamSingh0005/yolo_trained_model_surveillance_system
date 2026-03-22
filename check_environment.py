"""
Environment Check Script
Verifies all dependencies and system requirements
"""

import sys
import subprocess
from pathlib import Path

def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def check_python_version():
    """Check Python version"""
    print_section("Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✓ Python version is compatible")
        return True
    else:
        print("✗ Python 3.8+ required")
        return False

def check_packages():
    """Check required packages"""
    print_section("Required Packages")
    
    packages = {
        'torch': 'PyTorch',
        'ultralytics': 'Ultralytics YOLO',
        'cv2': 'OpenCV',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'yaml': 'PyYAML',
        'PIL': 'Pillow'
    }
    
    all_installed = True
    
    for package, name in packages.items():
        try:
            if package == 'cv2':
                import cv2
                version = cv2.__version__
            elif package == 'yaml':
                import yaml
                version = yaml.__version__ if hasattr(yaml, '__version__') else 'installed'
            elif package == 'sklearn':
                import sklearn
                version = sklearn.__version__
            else:
                module = __import__(package)
                version = module.__version__
            
            print(f"✓ {name:<20} : {version}")
        except ImportError:
            print(f"✗ {name:<20} : NOT INSTALLED")
            all_installed = False
    
    return all_installed

def check_cuda():
    """Check CUDA availability"""
    print_section("CUDA and GPU")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✓ CUDA Available: Yes")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print("⚠ CUDA Available: No")
            print("  Training will use CPU (slower)")
            return False
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        return False

def check_dataset():
    """Check dataset structure"""
    print_section("Dataset Structure")
    
    required_paths = [
        'data.yaml',
        'train/images',
        'train/labels',
        'test/images',
        'test/labels'
    ]
    
    all_exist = True
    
    for path_str in required_paths:
        path = Path(path_str)
        if path.exists():
            if path.is_dir():
                count = len(list(path.glob('*')))
                print(f"✓ {path_str:<20} : {count} files")
            else:
                print(f"✓ {path_str:<20} : exists")
        else:
            print(f"✗ {path_str:<20} : NOT FOUND")
            all_exist = False
    
    return all_exist

def check_disk_space():
    """Check available disk space"""
    print_section("Disk Space")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)
        used_percent = (used / total) * 100
        
        print(f"Total Space: {total_gb:.2f} GB")
        print(f"Used: {used_percent:.1f}%")
        print(f"Free: {free_gb:.2f} GB")
        
        if free_gb >= 5:
            print("✓ Sufficient disk space available")
            return True
        else:
            print("⚠ Low disk space (5GB+ recommended)")
            return False
    except Exception as e:
        print(f"✗ Error checking disk space: {e}")
        return False

def check_scripts():
    """Check if all required scripts exist"""
    print_section("Required Scripts")
    
    scripts = [
        'train_pipeline.py',
        'evaluate_model.py',
        'visualize_results.py',
        'ieee_paper_analysis.py',
        'run_complete_training.py',
        'complete_pipeline.py',
        'quick_start.py'
    ]
    
    all_exist = True
    
    for script in scripts:
        path = Path(script)
        if path.exists():
            print(f"✓ {script}")
        else:
            print(f"✗ {script} : NOT FOUND")
            all_exist = False
    
    return all_exist

def print_recommendations():
    """Print recommendations based on system"""
    print_section("Recommendations")
    
    import torch
    
    if torch.cuda.is_available():
        print("✓ GPU detected - Use default settings:")
        print("  python run_complete_training.py --epochs 100 --batch 16")
    else:
        print("⚠ No GPU detected - Use reduced settings:")
        print("  python run_complete_training.py --epochs 50 --batch 4")
    
    print("\nFor quick testing (10 epochs):")
    print("  python quick_start.py train")
    
    print("\nFor step-by-step execution:")
    print("  1. python train_pipeline.py")
    print("  2. python evaluate_model.py")
    print("  3. python visualize_results.py")
    print("  4. python ieee_paper_analysis.py")

def main():
    """Main environment check"""
    print("=" * 70)
    print(" SURVEILLANCE SYSTEM - ENVIRONMENT CHECK")
    print("=" * 70)
    
    checks = {
        'Python Version': check_python_version(),
        'Required Packages': check_packages(),
        'CUDA/GPU': check_cuda(),
        'Dataset Structure': check_dataset(),
        'Disk Space': check_disk_space(),
        'Required Scripts': check_scripts()
    }
    
    print_section("Summary")
    
    for check_name, result in checks.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{check_name:<25} : {status}")
    
    all_passed = all(checks.values())
    
    if all_passed:
        print("\n" + "=" * 70)
        print(" ✓ ALL CHECKS PASSED - READY TO TRAIN!")
        print("=" * 70)
        print_recommendations()
        return 0
    else:
        print("\n" + "=" * 70)
        print(" ✗ SOME CHECKS FAILED - PLEASE FIX ISSUES ABOVE")
        print("=" * 70)
        print("\nTo install missing packages:")
        print("  pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
