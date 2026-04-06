"""
Raspberry Pi Real-time Surveillance System
4-Class YOLO Detection: Animal, Forest, Militant, UAV-Drone
Optimized for Raspberry Pi 4/5 with camera support
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime
import json
import os
from pathlib import Path
import argparse

class RaspberryPiSurveillance:
    def __init__(self, model_path='FINAL_4CLASS_MODEL/best_4class_model.pt', 
                 conf_threshold=0.5, camera_id=0, save_detections=True):
        """
        Initialize Raspberry Pi Surveillance System
        
        Args:
            model_path: Path to trained YOLO model
            conf_threshold: Confidence threshold for detections
            camera_id: Camera device ID (0 for default, or RTSP URL)
            save_detections: Whether to save detection images
        """
        print("=" * 60)
        print("RASPBERRY PI SURVEILLANCE SYSTEM - 4-CLASS YOLO")
        print("=" * 60)
        
        # Load model
        print(f"\n[1/4] Loading model from {model_path}...")
        self.model = YOLO(model_path)
        print("✓ Model loaded successfully")
        
        # Configuration
        self.conf_threshold = conf_threshold
        self.camera_id = camera_id
        self.save_detections = save_detections
        
        # Class names and colors
        self.class_names = ['Animal', 'Forest', 'Militant', 'UAV-Drone']
        self.colors = {
            'Animal': (0, 255, 0),      # Green
            'Forest': (34, 139, 34),    # Forest Green
            'Militant': (0, 0, 255),    # Red
            'UAV-Drone': (255, 165, 0)  # Orange
        }
        
        # Threat levels
        self.threat_levels = {
            'Animal': 'LOW',
            'Forest': 'LOW',
            'Militant': 'HIGH',
            'UAV-Drone': 'MEDIUM'
        }
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'detections': 0,
            'fps_history': [],
            'class_counts': {name: 0 for name in self.class_names}
        }
        
        # Create output directories
        if self.save_detections:
            self.output_dir = Path('raspberry_pi_detections')
            self.output_dir.mkdir(exist_ok=True)
            (self.output_dir / 'images').mkdir(exist_ok=True)
            (self.output_dir / 'logs').mkdir(exist_ok=True)
            print(f"✓ Output directory: {self.output_dir}")
        
        print(f"✓ Confidence threshold: {self.conf_threshold}")
        print(f"✓ Camera ID: {self.camera_id}")
        
    def initialize_camera(self):
        """Initialize camera with optimized settings for Raspberry Pi"""
        print("\n[2/4] Initializing camera...")
        
        # Try to open camera
        if isinstance(self.camera_id, str) and self.camera_id.startswith('rtsp'):
            # RTSP stream
            self.cap = cv2.VideoCapture(self.camera_id)
        else:
            # USB/CSI camera
            self.cap = cv2.VideoCapture(self.camera_id)
            
            # Optimize camera settings for Raspberry Pi
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")
        
        # Test frame capture
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture test frame")
        
        print(f"✓ Camera initialized: {frame.shape[1]}x{frame.shape[0]}")
        return True
    
    def detect_threats(self, frame):
        """
        Run YOLO detection on frame
        
        Returns:
            results: YOLO detection results
            detections: List of detection dictionaries
        """
        # Run inference
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        # Parse detections
        detections = []
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                box = boxes[i]
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                class_name = self.class_names[cls_id]
                threat_level = self.threat_levels[class_name]
                
                detection = {
                    'class': class_name,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2],
                    'threat_level': threat_level,
                    'timestamp': datetime.now().isoformat()
                }
                detections.append(detection)
                
                # Update statistics
                self.stats['class_counts'][class_name] += 1
        
        if detections:
            self.stats['detections'] += 1
        
        return results, detections
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on frame"""
        annotated_frame = frame.copy()
        
        for det in detections:
            class_name = det['class']
            conf = det['confidence']
            x1, y1, x2, y2 = det['bbox']
            threat_level = det['threat_level']
            
            # Get color
            color = self.colors[class_name]
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f"{class_name} {conf:.2f}"
            threat_label = f"[{threat_level}]"
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            threat_size, _ = cv2.getTextSize(threat_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 25), 
                         (x1 + max(label_size[0], threat_size[0]) + 10, y1), color, -1)
            
            # Draw text
            cv2.putText(annotated_frame, label, (x1 + 5, y1 - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated_frame, threat_label, (x1 + 5, y1 - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated_frame
    
    def add_overlay(self, frame, fps, detections):
        """Add information overlay to frame"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # System info
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"Raspberry Pi Surveillance | {timestamp}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # FPS and detection count
        cv2.putText(frame, f"FPS: {fps:.1f} | Detections: {len(detections)}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Threat alert
        if detections:
            high_threats = [d for d in detections if d['threat_level'] == 'HIGH']
            if high_threats:
                cv2.putText(frame, "!!! HIGH THREAT DETECTED !!!", (10, 75),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def save_detection(self, frame, detections):
        """Save detection image and log"""
        if not self.save_detections or not detections:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save image
        img_path = self.output_dir / 'images' / f'detection_{timestamp}.jpg'
        cv2.imwrite(str(img_path), frame)
        
        # Save log
        log_path = self.output_dir / 'logs' / f'detection_{timestamp}.json'
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'detections': detections,
            'image_path': str(img_path)
        }
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def run(self, display=True, max_frames=None):
        """
        Run surveillance system
        
        Args:
            display: Whether to display video feed (requires X server)
            max_frames: Maximum frames to process (None for infinite)
        """
        print("\n[3/4] Starting surveillance...")
        print("Press 'q' to quit, 's' to save current frame")
        print("-" * 60)
        
        try:
            self.initialize_camera()
            
            frame_count = 0
            start_time = time.time()
            
            while True:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                frame_count += 1
                self.stats['total_frames'] += 1
                
                # Run detection
                inference_start = time.time()
                results, detections = self.detect_threats(frame)
                inference_time = time.time() - inference_start
                
                # Calculate FPS
                fps = 1.0 / inference_time if inference_time > 0 else 0
                self.stats['fps_history'].append(fps)
                
                # Draw detections
                annotated_frame = self.draw_detections(frame, detections)
                annotated_frame = self.add_overlay(annotated_frame, fps, detections)
                
                # Save detections
                if detections:
                    self.save_detection(annotated_frame, detections)
                    
                    # Print detection info
                    print(f"\n[Frame {frame_count}] {len(detections)} detection(s):")
                    for det in detections:
                        print(f"  - {det['class']}: {det['confidence']:.2f} [{det['threat_level']}]")
                
                # Display frame
                if display:
                    cv2.imshow('Raspberry Pi Surveillance', annotated_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nQuitting...")
                        break
                    elif key == ord('s'):
                        save_path = f'manual_save_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
                        cv2.imwrite(save_path, annotated_frame)
                        print(f"Saved: {save_path}")
                
                # Check max frames
                if max_frames and frame_count >= max_frames:
                    break
                
                # Print progress every 30 frames
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    avg_fps = frame_count / elapsed
                    print(f"[Progress] Frames: {frame_count} | Avg FPS: {avg_fps:.1f}")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            self.cleanup()
            self.print_statistics()
    
    def cleanup(self):
        """Release resources"""
        print("\n[4/4] Cleaning up...")
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
        print("✓ Resources released")
    
    def print_statistics(self):
        """Print final statistics"""
        print("\n" + "=" * 60)
        print("SURVEILLANCE STATISTICS")
        print("=" * 60)
        print(f"Total Frames Processed: {self.stats['total_frames']}")
        print(f"Frames with Detections: {self.stats['detections']}")
        print(f"Detection Rate: {self.stats['detections']/max(self.stats['total_frames'], 1)*100:.1f}%")
        
        if self.stats['fps_history']:
            avg_fps = np.mean(self.stats['fps_history'])
            print(f"Average FPS: {avg_fps:.2f}")
        
        print("\nDetections by Class:")
        for class_name, count in self.stats['class_counts'].items():
            print(f"  - {class_name}: {count}")
        
        if self.save_detections:
            print(f"\nDetections saved to: {self.output_dir}")
        
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Raspberry Pi Surveillance System')
    parser.add_argument('--model', type=str, 
                       default='FINAL_4CLASS_MODEL/best_4class_model.pt',
                       help='Path to YOLO model')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold (0-1)')
    parser.add_argument('--camera', type=str, default='0',
                       help='Camera ID or RTSP URL')
    parser.add_argument('--no-display', action='store_true',
                       help='Run without display (headless mode)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save detections')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames to process')
    
    args = parser.parse_args()
    
    # Convert camera ID
    camera_id = args.camera
    if camera_id.isdigit():
        camera_id = int(camera_id)
    
    # Initialize system
    surveillance = RaspberryPiSurveillance(
        model_path=args.model,
        conf_threshold=args.conf,
        camera_id=camera_id,
        save_detections=not args.no_save
    )
    
    # Run surveillance
    surveillance.run(
        display=not args.no_display,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()
