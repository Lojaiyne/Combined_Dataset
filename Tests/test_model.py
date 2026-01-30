#!/usr/bin/env python3
"""
Test YOLO model with images or video
Usage: python test_model.py --source <image/video/folder>
"""

import argparse
import cv2
from ultralytics import YOLO
import os

# Configuration
MODEL_PATH = "my_model/my_model.pt"
CONFIDENCE_THRESHOLD = 0.5

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, help='Path to image, video, or folder')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    args = parser.parse_args()
    
    print("Loading YOLO model...")
    try:
        model = YOLO(MODEL_PATH)
        print(f"✓ Model loaded successfully from {MODEL_PATH}")
        print(f"✓ Classes: {model.names}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    print(f"\nRunning inference on: {args.source}")
    print("=" * 50)
    
    # Run prediction
    results = model.predict(
        source=args.source,
        conf=args.conf,
        save=True,
        show=True
    )
    
    print(f"\n✓ Results saved to: runs/detect/predict/")
    print("✓ Done!")

if __name__ == "__main__":
    main()
