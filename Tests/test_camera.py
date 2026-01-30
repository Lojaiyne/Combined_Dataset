#!/usr/bin/env python3
"""
Simple script to test YOLO model with camera
No Anaconda required - just run with: python test_camera.py
"""

import cv2
from ultralytics import YOLO

# Configuration
MODEL_PATH = "my_model/my_model.pt"  # Path to your trained model
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to display detections
CAMERA_INDEX = 0  # 0 for default camera, 1 for second camera, etc.

def main():
    print("Loading YOLO model...")
    try:
        model = YOLO(MODEL_PATH)
        print(f"✓ Model loaded successfully from {MODEL_PATH}")
        print(f"✓ Classes: {model.names}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print(f"  Make sure '{MODEL_PATH}' exists")
        return
    
    print(f"\nTrying to open camera...")
    cap = None
    
    # Try multiple camera indices and backends
    backends = [cv2.CAP_ANY, cv2.CAP_V4L2, cv2.CAP_DSHOW]
    for backend in backends:
        for cam_idx in range(10):
            print(f"  Trying camera {cam_idx} with backend {backend}...", end=" ")
            test_cap = cv2.VideoCapture(cam_idx, backend)
            if test_cap.isOpened():
                # Verify we can actually read a frame
                ret, frame = test_cap.read()
                if ret:
                    print("✓ Found and working!")
                    cap = test_cap
                    break
                else:
                    print("✗ (opened but can't read)")
                    test_cap.release()
            else:
                print("✗")
                test_cap.release()
        if cap is not None:
            break
    
    if cap is None:
        print(f"\n✗ Error: No working camera found")
        print("  Make sure the webcam is properly connected and not in use by another program")
        return
    
    print("✓ Camera opened successfully")
    print("\nPress 'q' to quit")
    print("=" * 50)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            break
        
        # Run YOLO detection
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        # Draw results on frame
        annotated_frame = results[0].plot()
        
        # Display the frame
        cv2.imshow('YOLO Detection - Press Q to quit', annotated_frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Camera test completed")

if __name__ == "__main__":
    main()
