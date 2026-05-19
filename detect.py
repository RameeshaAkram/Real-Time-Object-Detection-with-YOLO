"""
Computer Vision Lab Final - Real-Time Object Detection with YOLOv8
Student: [Your Name]
Class: BSAI Sem 5th
Date: 22-April-2026
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
from collections import defaultdict
import argparse
import os


class RealTimeObjectDetector:
    """
    Real-time object detection system using YOLOv8
    Detects 80+ COCO classes, displays bounding boxes, confidence scores, and FPS
    """
    
    # COCO class names (80 categories - exceeds the 15 requirement)
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
    
    def __init__(self, model_name='yolov8n.pt', conf_threshold=0.5, device='auto'):
        """
        Initialize the detector with YOLO model
        
        Args:
            model_name: YOLO model variant (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
            conf_threshold: Minimum confidence score for detections
            device: 'cpu', 'cuda', or 'auto'
        """
        self.conf_threshold = conf_threshold
        
        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"[INFO] Using device: {self.device}")
        
        # Load YOLO model
        print(f"[INFO] Loading model: {model_name}")
        self.model = YOLO(model_name)
        
        # Move model to device
        if self.device == 'cuda':
            self.model.to('cuda')
            
        # For FPS calculation
        self.fps_buffer = []
        self.last_time = time.time()
        
        # Detection history for counting
        self.detection_history = defaultdict(list)
        
    def calculate_fps(self):
        """Calculate real-time FPS"""
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time)
        self.last_time = current_time
        
        # Keep buffer of last 30 FPS values for smoothing
        self.fps_buffer.append(fps)
        if len(self.fps_buffer) > 30:
            self.fps_buffer.pop(0)
            
        return np.mean(self.fps_buffer)
    
    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes, labels, and confidence scores on frame
        
        Args:
            frame: Input image frame
            detections: List of detection results from YOLO
        """
        if detections[0].boxes is None:
            return frame
            
        boxes = detections[0].boxes.xyxy.cpu().numpy()
        confidences = detections[0].boxes.conf.cpu().numpy()
        class_ids = detections[0].boxes.cls.cpu().numpy().astype(int)
        
        for box, conf, class_id in zip(boxes, confidences, class_ids):
            if conf < self.conf_threshold:
                continue
                
            x1, y1, x2, y2 = map(int, box)
            class_name = self.COCO_CLASSES[class_id] if class_id < len(self.COCO_CLASSES) else f'class_{class_id}'
            
            # Dynamic color based on class (hash for consistency)
            color = self._get_color(class_id)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label with class name and confidence
            label = f"{class_name}: {conf:.2f}"
            
            # Draw label background
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame,
                (x1, y1 - label_height - 10),
                (x1 + label_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            
            # Update detection history for counting
            self.detection_history[class_name].append(time.time())
            
        return frame
    
    def _get_color(self, class_id):
        """Generate consistent color for each class"""
        np.random.seed(class_id)
        return tuple(map(int, np.random.randint(0, 255, 3)))
    
    def draw_info_panel(self, frame, fps, total_detections, current_classes):
        """
        Draw information panel with FPS, detection count, and active classes
        
        Args:
            frame: Input image frame
            fps: Current FPS
            total_detections: Total number of detections in current frame
            current_classes: Set of classes detected in current frame
        """
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # FPS counter
        cv2.putText(
            frame, f"FPS: {fps:.1f}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        
        # Total detections
        cv2.putText(
            frame, f"Objects: {total_detections}", (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
        )
        
        # Active classes
        classes_text = f"Classes: {', '.join(list(current_classes)[:3])}"
        if len(current_classes) > 3:
            classes_text += f" +{len(current_classes) - 3}"
            
        cv2.putText(
            frame, classes_text, (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2
        )
        
        # Model info
        cv2.putText(
            frame, f"Device: {self.device.upper()}", (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )
        
        return frame
    
    def process_frame(self, frame):
        """
        Process a single frame: run inference and draw results
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            Processed frame with detections and info panel
        """
        # Run YOLO inference
        results = self.model(frame, verbose=False)
        
        # Draw detections
        frame = self.draw_detections(frame, results)
        
        # Calculate FPS
        fps = self.calculate_fps()
        
        # Get current detections info
        if results[0].boxes is not None:
            total_detections = len(results[0].boxes)
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            current_classes = set(
                self.COCO_CLASSES[cid] for cid in class_ids 
                if cid < len(self.COCO_CLASSES)
            )
        else:
            total_detections = 0
            current_classes = set()
        
        # Draw info panel
        frame = self.draw_info_panel(frame, fps, total_detections, current_classes)
        
        return frame
    
    def run_webcam(self, camera_id=0):
        """
        Run real-time detection on webcam feed
        
        Args:
            camera_id: Webcam device ID (default: 0)
        """
        print("[INFO] Starting webcam detection...")
        print("[INFO] Press 'q' to quit")
        print("[INFO] Press 's' to save screenshot")
        print("[INFO] Press 'r' to reset detection history")
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("[ERROR] Could not open webcam")
            return
            
        # Set resolution for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        screenshot_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame")
                break
                
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Display
            cv2.imshow('YOLOv8 Real-Time Object Detection', processed_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("[INFO] Quitting...")
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = f"screenshot_{screenshot_count}.jpg"
                cv2.imwrite(filename, processed_frame)
                print(f"[INFO] Saved screenshot: {filename}")
            elif key == ord('r'):
                self.detection_history.clear()
                print("[INFO] Detection history reset")
                
        cap.release()
        cv2.destroyAllWindows()
        
    def run_video(self, video_path):
        """
        Run detection on recorded video file
        
        Args:
            video_path: Path to video file
        """
        if not os.path.exists(video_path):
            print(f"[ERROR] Video file not found: {video_path}")
            return
            
        print(f"[INFO] Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties for output
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output video writer
        output_path = "output_detection.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Write to output
            out.write(processed_frame)
            
            # Display
            cv2.imshow('YOLOv8 Video Detection', processed_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                print(f"[INFO] Processed {frame_count} frames ({frame_count/elapsed:.1f} FPS)")
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        elapsed_total = time.time() - start_time
        print(f"\n[INFO] Video processing complete!")
        print(f"[INFO] Total frames: {frame_count}")
        print(f"[INFO] Average FPS: {frame_count/elapsed_total:.1f}")
        print(f"[INFO] Output saved to: {output_path}")
        
    def run_image(self, image_path):
        """
        Run detection on a single image
        
        Args:
            image_path: Path to image file
        """
        if not os.path.exists(image_path):
            print(f"[ERROR] Image file not found: {image_path}")
            return
            
        print(f"[INFO] Processing image: {image_path}")
        
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            print("[ERROR] Could not read image")
            return
            
        # Process image
        processed_frame = self.process_frame(frame)
        
        # Save output
        output_path = "output_detection.jpg"
        cv2.imwrite(output_path, processed_frame)
        print(f"[INFO] Output saved to: {output_path}")
        
        # Display
        cv2.imshow('YOLOv8 Image Detection', processed_frame)
        print("[INFO] Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def benchmark_performance(detector, num_frames=100):
    """
    Benchmark detection performance on a test video
    
    Args:
        detector: RealTimeObjectDetector instance
        num_frames: Number of frames to benchmark
    """
    print("\n" + "="*50)
    print("PERFORMANCE BENCHMARK")
    print("="*50)
    
    # Create a dummy frame
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Warm up
    for _ in range(10):
        detector.process_frame(dummy_frame)
        
    # Benchmark
    times = []
    detector.last_time = time.time()
    
    for i in range(num_frames):
        start = time.time()
        detector.process_frame(dummy_frame)
        end = time.time()
        times.append(end - start)
        
    avg_time = np.mean(times) * 1000  # Convert to ms
    fps = 1000 / avg_time
    
    print(f"\n[Benchmark Results]")
    print(f"  Device: {detector.device.upper()}")
    print(f"  Model: {detector.model.model_name}")
    print(f"  Frames tested: {num_frames}")
    print(f"  Average inference time: {avg_time:.2f} ms")
    print(f"  Average FPS: {fps:.1f}")
    
    # Check requirements
    if fps >= 15:
        print(f"  ✅ FPS requirement met: {fps:.1f} >= 15")
    else:
        print(f"  ⚠️  FPS requirement NOT met: {fps:.1f} < 15 (try using 'yolov8n.pt' on CPU or GPU)")
        
    return fps


def main():
    parser = argparse.ArgumentParser(description='Real-Time Object Detection with YOLOv8')
    parser.add_argument('--source', type=str, default='webcam',
                        help='Source: webcam, video path, or image path')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='YOLO model (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold (default: 0.5)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, cuda')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run performance benchmark')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("COMPUTER VISION LAB FINAL - REAL-TIME OBJECT DETECTION")
    print("Class: BSAI Sem 5th | Date: 22-April-2026")
    print("="*60 + "\n")
    
    # Initialize detector
    detector = RealTimeObjectDetector(
        model_name=args.model,
        conf_threshold=args.conf,
        device=args.device
    )
    
    # Run benchmark if requested
    if args.benchmark:
        benchmark_performance(detector)
        return
    
    # Run detection based on source
    if args.source == 'webcam':
        detector.run_webcam()
    elif args.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        detector.run_video(args.source)
    elif args.source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        detector.run_image(args.source)
    else:
        print(f"[ERROR] Unknown source type: {args.source}")
        print("Usage: python detect.py --source webcam")
        print("       python detect.py --source video.mp4")
        print("       python detect.py --source image.jpg")
        print("       python detect.py --benchmark")


if __name__ == "__main__":
    main()