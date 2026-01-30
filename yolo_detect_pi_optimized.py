#!/usr/bin/env python3
"""
Optimized YOLO detection for Raspberry Pi 4 with Camera Module 3
Faster inference with reduced resolution and frame skipping
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file', required=True)
parser.add_argument('--thresh', help='Confidence threshold (default: 0.5)', default=0.5, type=float)
parser.add_argument('--resolution', help='Resolution WxH (default: 640x480)', default='640x480')
parser.add_argument('--frame-skip', help='Process every Nth frame (default: 2)', default=2, type=int)
parser.add_argument('--device', help='Device (default: cpu, use "0" for GPU if available)', default='cpu')

args = parser.parse_args()

if not os.path.exists(args.model):
    print(f'ERROR: Model not found: {args.model}')
    sys.exit(1)

# Load model with optimization for Pi
print('Loading YOLO model...')
try:
    # Load YOLO model (Ultralytics auto-detects format)
    model_path = args.model
    model = YOLO(model_path, task='detect')

    # IMPORTANT: only move device for PyTorch (.pt) models
    if str(model_path).lower().endswith(".pt"):
        model.to(args.device)
        print(f'  ✓ PyTorch model loaded')
    else:
        print(f"  ✓ Exported model detected (NCNN/ONNX/etc) → running on CPU")

    labels = model.names
    print(f'  ✓ Classes: {list(labels.values())}')

except Exception as e:
    print(f'ERROR: Failed to load model: {e}')
    print('  Make sure the model path is correct and compatible with ultralytics')
    print('  For NCNN: ensure both .param and .bin files are in the same directory')
    sys.exit(1)

# Parse resolution
resW, resH = map(int, args.resolution.split('x'))

print('Initializing Raspberry Pi Camera...')
from picamera2 import Picamera2

cap = Picamera2()
config = cap.create_video_configuration(
    main={"format": 'RGB888', "size": (resW, resH)},
    controls={"FrameRate": 30}
)
cap.configure(config)
cap.start()

# Color scheme for bboxes
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

frame_buffer = []
fps_window = 30
frame_count = 0
process_count = 0

print(f'✓ Camera ready. Running inference...')
print(f'  Resolution: {resW}x{resH}')
print(f'  Frame skip: {args.frame_skip} (process every {args.frame_skip} frames)')
print(f'  Device: {args.device}')
print(f'Press "q" to quit, "s" to pause\n')

try:
    while True:
        t_start = time.perf_counter()
        
        frame = cap.capture_array()
        if frame is None:
            break
        
        frame_count += 1
        
        # Skip frames for faster processing
        if frame_count % args.frame_skip != 0:
            t_stop = time.perf_counter()
            frame_rate = 1 / (t_stop - t_start) if (t_stop - t_start) > 0 else 0
            frame_buffer.append(frame_rate)
            if len(frame_buffer) > fps_window:
                frame_buffer.pop(0)
            continue
        
        process_count += 1
        
        # Run inference
        results = model(frame, verbose=False, conf=args.thresh)
        detections = results[0].boxes
        
        object_count = 0
        
        # Draw detections
        for i in range(len(detections)):
            xyxy_tensor = detections[i].xyxy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()
            if len(xyxy.shape) == 1:  # Handle single detection
                xmin, ymin, xmax, ymax = xyxy.astype(int)
            else:
                continue
            
            classidx = int(detections[i].cls.item())
            classname = labels[classidx]
            conf = detections[i].conf.item()
            
            if conf > args.thresh:
                color = bbox_colors[classidx % 10]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                
                label = f'{classname}: {int(conf*100)}%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), 
                            (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin-7), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                object_count += 1
        
        # Calculate FPS
        t_stop = time.perf_counter()
        frame_rate = 1 / (t_stop - t_start) if (t_stop - t_start) > 0 else 0
        frame_buffer.append(frame_rate)
        if len(frame_buffer) > fps_window:
            frame_buffer.pop(0)
        
        avg_fps = np.mean(frame_buffer)
        
        # Draw info
        cv2.putText(frame, f'FPS: {avg_fps:.1f}', (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f'Objects: {object_count}', (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f'Frame: {frame_count} (Processed: {process_count})', (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        cv2.imshow('YOLO Detection - Raspberry Pi', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.waitKey()
        elif key == ord('p'):
            cv2.imwrite('capture.png', frame)
            print('✓ Image saved: capture.png')

except KeyboardInterrupt:
    print('\n✓ Interrupted by user')

finally:
    print(f'\nShutting down...')
    print(f'  Total frames: {frame_count}')
    print(f'  Processed: {process_count}')
    print(f'  Avg FPS: {np.mean(frame_buffer):.1f}')
    cap.stop()
    cv2.destroyAllWindows()
    print('✓ Done')
