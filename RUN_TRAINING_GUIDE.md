# How to Run Training - Quick Reference

## The Correct Way

**Directory to run from:**
```
c:\Users\logy1\OneDrive - Aston University\Final Year Project\Combined_Dataset
```

**Command to use:**
```powershell
cd "c:\Users\logy1\OneDrive - Aston University\Final Year Project\Combined_Dataset"
.\yolo_venv\Scripts\python.exe train_yolo.py
```

Or the full path:
```powershell
cd "c:\Users\logy1\OneDrive - Aston University\Final Year Project\Combined_Dataset"
& "C:/Users/logy1/OneDrive - Aston University/Final Year Project/Combined_Dataset/yolo_venv/Scripts/python.exe" train_yolo.py
```

## Folder Structure

```
Combined_Dataset/              ← RUN FROM HERE
├── train_yolo.py             ← This is what we run
├── yolo_venv/                ← Virtual environment with Python
│   └── Scripts/
│       └── python.exe        ← Python interpreter
├── Combined Dataset/          ← Your dataset
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── data.yaml
└── runs/                      ← Where results will be saved
    └── detect/
        └── waste_detector/    ← Training results go here
```

## What's Happening Now

Training is running **on CPU** (no GPU available on your system):
- **Epochs**: 200 (will take several hours)
- **Batch size**: 16 (reduced for CPU)
- **Image size**: 640x640
- **Augmentation**: Enabled (mosaic, mixup)

## Monitor Progress

Watch the terminal output - you'll see:
```
Epoch 1/200  loss: 2.34  val_loss: 1.89  mAP: 0.45
Epoch 2/200  loss: 2.10  val_loss: 1.67  mAP: 0.52
...
```

Loss should **decrease** over time, mAP should **increase** over time.

## When Training Completes

Check these files:
1. `runs/detect/waste_detector/confusion_matrix.png` - Shows paper/plastic confusion
2. `runs/detect/waste_detector/results.png` - Shows loss/accuracy curves
3. `runs/detect/waste_detector/weights/best.pt` - Your trained model

## If You Need to Stop Training

Press `Ctrl+C` in the terminal. The best model weights are saved automatically.

## Notes

- Training will take 2-6 hours on CPU (depending on your processor)
- Don't close the terminal while training is running
- You can check progress by looking at the printed epochs
