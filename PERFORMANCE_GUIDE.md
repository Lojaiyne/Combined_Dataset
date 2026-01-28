# YOLO Model Performance Issues - Root Cause Analysis & Solutions

## Your Model's Problems

### 1. **Severe Class Imbalance** ⚠️ MAIN ISSUE
Your validation set has a **severe imbalance**:
- Paper: 70.6% (2,691 objects)
- Plastic: 15.4% (588 objects)  
- Metal: 14.0% (533 objects)

**Why this causes misclassification**: The model learns that predicting "paper" is a safe bet since it's right 70% of the time. This causes:
- Plastic objects → misclassified as paper
- Metal objects → misclassified as paper
- Some objects not detected at all (low confidence)

### 2. **Data Integrity Issues**
- 310 orphaned labels in training set (labels with no images)
- 60 images missing labels in training set
- 29 images missing labels in validation set

These inconsistencies confuse the training process.

### 3. **Suboptimal Training Configuration**
- Training on **CPU** instead of GPU (very slow)
- Small batch size (16) - leads to noisy gradient estimates
- No class weighting to balance loss
- Limited augmentation for generalization
- Only 100 epochs (may need more)

---

## Solutions

### Step 1: Clean Up Your Dataset
Run the cleanup script to remove orphaned labels and mismatched files:

```bash
cd "c:\Users\logy1\OneDrive - Aston University\Final Year Project\Combined_Dataset"
.\yolo_venv\Scripts\python.exe cleanup_dataset.py
```

Then verify the cleanup:
```bash
.\yolo_venv\Scripts\python.exe analyze_dataset.py
```

### Step 2: Rebalance Your Dataset

**Option A: Duplicate Underrepresented Classes** (Quick Fix)
If plastic and metal have fewer images, duplicate some of them:
```bash
# Copy more metal and plastic images to increase their presence
# This helps the model learn these classes better
```

**Option B: Collect More Data** (Best Solution)
- Collect more plastic images to match paper's ~2,700 objects
- Collect more metal images to match paper's ~2,700 objects
- Ensure diverse angles, lighting, and backgrounds

**Option C: Use Sampling Weights** (What we've updated)
The improved training script now uses class weighting (`cls_pw=1.5`) which penalizes 
misclassifying minority classes more heavily.

### Step 3: Use Improved Training Script
The training script has been updated with:

```python
# Better performance
device=0              # Auto-detect GPU (falls back to CPU)
epochs=200            # More epochs for convergence
batch=32              # Larger batches for stable gradients
workers=4             # More parallel data loading

# Handle imbalance
cls_pw=1.5            # Class weight factor

# Better generalization
mosaic=1.0            # Mosaic augmentation
mixup=0.1             # Mixup augmentation

# Better logging
plots=True            # Generate training plots
save_json=True        # Save results as JSON
```

Run the updated training:
```bash
.\yolo_venv\Scripts\python.exe train_yolo.py
```

### Step 4: Evaluate Results
After training, check the confusion matrix and results:
- Look for the `confusion_matrix.png` in `runs/detect/waste_detector/`
- Check `results.png` to see loss/precision/recall curves
- Use the confusion matrix to identify which classes are being mixed up

---

## Best Practices for Better Results

1. **Balanced Dataset**: Aim for roughly equal images per class
   - Ideally 500-1000 images per class minimum
   - 80% train / 20% validation split

2. **High-Quality Annotations**:
   - Double-check that paper/plastic labels are correct
   - Ensure bounding boxes tightly fit objects
   - Look for mislabeled images (common cause of low accuracy)

3. **Training Parameters**:
   - Use GPU if available (100x faster than CPU)
   - Use larger models (yolov8l.pt) if accuracy is more important than speed
   - Use smaller models (yolov8n.pt) if speed is important

4. **Augmentation**:
   - Enable mosaic and mixup (already in updated script)
   - This helps the model generalize to different object positions/sizes

5. **Validation**:
   - Regularly inspect model predictions
   - If paper/plastic still confused, look at individual images in those classes
   - Check if labels are genuinely correct

---

## Monitoring Training

After running the updated script, check these files:
- `runs/detect/waste_detector/results.csv` - Metrics over epochs
- `runs/detect/waste_detector/results.png` - Loss/precision/recall plots
- `runs/detect/waste_detector/confusion_matrix.png` - Class confusion matrix
- `runs/detect/waste_detector/val_batch*.jpg` - Predictions on validation set

The confusion matrix will show where misclassifications happen:
```
         Predicted
         Metal  Plastic  Paper
Actual:
Metal      [X]    [ ]     [ ]    ← Should be mostly diagonal
Plastic    [ ]    [X]     [ ]    ← Should be mostly diagonal  
Paper      [ ]    [ ]     [X]    ← Should be mostly diagonal
```

If you see off-diagonal values (e.g., many Plastics predicted as Paper), your 
model still has a problem that likely needs better balanced training data.

---

## Next Steps

1. Run `cleanup_dataset.py` to fix data integrity issues
2. Run `analyze_dataset.py` again to see updated class distribution
3. Consider rebalancing the dataset (collect more plastic/metal or duplicate)
4. Run the improved `train_yolo.py`
5. Analyze the confusion matrix to see if plastic/paper confusion is resolved
6. If still not solved, focus on improving your dataset quality/balance

Let me know if you need help with any of these steps!
