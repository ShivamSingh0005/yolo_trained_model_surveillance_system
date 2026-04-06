"""
Quick test script for Raspberry Pi setup
Tests camera, model loading, and basic inference
"""

import sys
import time

def test_imports():
    """Test if all required packages are installed"""
    print("=" * 60)
    print("Testing Package Imports")
    print("=" * 60)
    
    packages = {
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'torch': 'PyTorch',
        'ultralytics': 'Ultralytics YOLO'
    }
    
    failed = []
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError as e:
            print(f"✗ {name}: {e}")
            failed.append(name)
    
    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        print("Run: pip3 install ultralytics opencv-python numpy torch")
        return False
    
    print("\n✓ All packages imported successfully")
    return True


def test_camera():
    """Test camera connection"""
    print("\n" + "=" * 60)
    print("Testing Camera")
    print("=" * 60)
    
    try:
        import cv2
        
        # Try to open camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Failed to open camera")
            print("Check:")
            print("  1. Camera is connected")
            print("  2. Camera is enabled: sudo raspi-config")
            print("  3. Try: ls /dev/video*")
            return False
        
        # Try to capture frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("❌ Failed to capture frame")
            return False
        
        print(f"✓ Camera working: {frame.shape[1]}x{frame.shape[0]}")
        return True
        
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False


def test_model():
    """Test model loading"""
    print("\n" + "=" * 60)
    print("Testing Model Loading")
    print("=" * 60)
    
    try:
        from ultralytics import YOLO
        import os
        
        model_paths = [
            'FINAL_4CLASS_MODEL/best_4class_model.pt',
            'best_4class_model.pt',
            '../FINAL_4CLASS_MODEL/best_4class_model.pt'
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            print("❌ Model file not found")
            print("Expected locations:")
            for path in model_paths:
                print(f"  - {path}")
            print("\nTransfer model using:")
            print("  scp FINAL_4CLASS_MODEL/best_4class_model.pt pi@<ip>:~/")
            return False
        
        print(f"Found model: {model_path}")
        print("Loading model...")
        
        start = time.time()
        model = YOLO(model_path)
        load_time = time.time() - start
        
        print(f"✓ Model loaded in {load_time:.2f}s")
        print(f"  Classes: {model.names}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False


def test_inference():
    """Test inference on dummy image"""
    print("\n" + "=" * 60)
    print("Testing Inference")
    print("=" * 60)
    
    try:
        from ultralytics import YOLO
        import numpy as np
        import os
        
        # Find model
        model_paths = [
            'FINAL_4CLASS_MODEL/best_4class_model.pt',
            'best_4class_model.pt',
            '../FINAL_4CLASS_MODEL/best_4class_model.pt'
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            print("⚠️  Skipping inference test (model not found)")
            return True
        
        # Load model
        model = YOLO(model_path)
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        print("Running inference on dummy image...")
        start = time.time()
        results = model(dummy_image, verbose=False)
        inference_time = time.time() - start
        
        fps = 1.0 / inference_time
        print(f"✓ Inference successful")
        print(f"  Time: {inference_time:.3f}s")
        print(f"  FPS: {fps:.1f}")
        
        if fps < 1:
            print("\n⚠️  Warning: FPS is low (<1)")
            print("Consider:")
            print("  - Using lower resolution (416x416)")
            print("  - Exporting to ONNX format")
            print("  - Adding cooling (heatsink/fan)")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return False


def test_system_info():
    """Display system information"""
    print("\n" + "=" * 60)
    print("System Information")
    print("=" * 60)
    
    try:
        import platform
        import os
        
        print(f"Platform: {platform.platform()}")
        print(f"Python: {platform.python_version()}")
        
        # Check if Raspberry Pi
        if os.path.exists('/proc/cpuinfo'):
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                if 'Raspberry Pi' in cpuinfo:
                    # Extract model
                    for line in cpuinfo.split('\n'):
                        if 'Model' in line:
                            print(f"Device: {line.split(':')[1].strip()}")
                            break
        
        # Check temperature (Raspberry Pi specific)
        try:
            import subprocess
            temp = subprocess.check_output(['vcgencmd', 'measure_temp']).decode()
            print(f"Temperature: {temp.strip()}")
        except:
            pass
        
        # Memory info
        try:
            import subprocess
            mem = subprocess.check_output(['free', '-h']).decode()
            lines = mem.split('\n')[1].split()
            print(f"Memory: {lines[2]} used / {lines[1]} total")
        except:
            pass
        
        return True
        
    except Exception as e:
        print(f"⚠️  Could not get system info: {e}")
        return True


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("RASPBERRY PI SURVEILLANCE SYSTEM - TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("System Info", test_system_info),
        ("Package Imports", test_imports),
        ("Camera", test_camera),
        ("Model Loading", test_model),
        ("Inference", test_inference)
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print("\nYou can now run:")
        print("  python3 raspberry_pi_surveillance.py")
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 60)
        print("\nPlease fix the issues above before running surveillance system")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
