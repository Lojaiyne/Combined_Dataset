#!/usr/bin/env python3
"""
YOLOv8 Training Script for Waste Classification Dataset
This script trains a YOLOv8 model on your combined waste classification dataset
"""

import subprocess
import sys
from pathlib import Path

def install_dependencies():
    """Install required packages if not already installed"""
    packages = ['ultralytics', 'torch', 'torchvision']
    for package in packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

def main():
    # Install dependencies
    print("Checking dependencies...")
    install_dependencies()
    
    from ultralytics import YOLO
    
    # Get the dataset path - using the Combined Dataset subdirectory with absolute path
    data_yaml = r'C:\Users\logy1\OneDrive - Aston University\Final Year Project\Combined_Dataset\Combined Dataset\data.yaml'
    
    print(f"\n{'='*60}")
    print("YOLOv8 Training Configuration")
    print(f"{'='*60}")
    print(f"Dataset YAML: {data_yaml}")
    print(f"Dataset exists: {Path(data_yaml).exists()}")
    
    # Load a pretrained YOLOv8 model
    model = YOLO('yolov8m.pt')  # medium model (you can use 'yolov8n.pt' for nano, 'yolov8l.pt' for large, etc.)
    
    # Train the model
    print(f"\n{'='*60}")
    print("Starting YOLOv8 Training...")
    print(f"{'='*60}\n")
    
    results = model.train(
        data=str(data_yaml),
        epochs=200,              # Number of epochs (increased for better convergence)
        imgsz=640,              # Image size
        batch=16,               # Batch size (reduced for CPU training)
        patience=30,            # Early stopping patience
        device='cpu',           # Use CPU (no CUDA GPU available)
        workers=4,              # Number of dataloader workers
        save=True,              # Save model checkpoints
        verbose=True,           # Print training progress
        project='runs/detect',  # Project directory
        name='waste_detector',  # Run name
        # Augmentation for better generalization
        augment=True,           # Enable augmentation
        mosaic=1.0,             # Mosaic augmentation
        mixup=0.1,              # Mixup augmentation
        # Better logging
        save_json=True,         # Save results as JSON
        plots=True,             # Save training plots
        # Learning rate scheduling
        lr0=0.01,               # Initial learning rate
        lrf=0.01,               # Final learning rate
    )
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Model saved in: {Path('runs/detect/waste_detector')}")
    print(f"Best model: {Path('runs/detect/waste_detector/weights/best.pt')}")

if __name__ == '__main__':
    main()
