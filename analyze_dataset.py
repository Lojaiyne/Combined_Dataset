#!/usr/bin/env python3
"""
Dataset Analysis Script
Checks for class imbalance, missing labels, annotation errors
"""

from pathlib import Path
from collections import defaultdict
import json

def analyze_dataset():
    base_path = Path(r'C:\Users\logy1\OneDrive - Aston University\Final Year Project\Combined_Dataset\Combined Dataset')
    
    print("="*70)
    print("DATASET ANALYSIS REPORT")
    print("="*70)
    
    # Analyze train and val splits
    for split in ['train', 'valid']:
        split_path = base_path / split
        if not split_path.exists():
            print(f"\n⚠ {split.upper()} folder not found")
            continue
            
        images_dir = split_path / 'images'
        labels_dir = split_path / 'labels'
        
        image_files = list(images_dir.glob('*.*')) if images_dir.exists() else []
        label_files = list(labels_dir.glob('*.txt')) if labels_dir.exists() else []
        
        print(f"\n{split.upper()} SET:")
        print(f"  Images: {len(image_files)}")
        print(f"  Labels: {len(label_files)}")
        
        # Check for missing labels
        image_stems = {img.stem for img in image_files}
        label_stems = {lbl.stem for lbl in label_files}
        
        missing_labels = image_stems - label_stems
        orphaned_labels = label_stems - image_stems
        
        if missing_labels:
            print(f"  ⚠ Missing labels ({len(missing_labels)}): {list(missing_labels)[:5]}")
        if orphaned_labels:
            print(f"  ⚠ Orphaned labels ({len(orphaned_labels)}): {list(orphaned_labels)[:5]}")
        
        # Analyze class distribution in labels
        class_counts = defaultdict(int)
        object_counts = defaultdict(int)
        
        for label_file in label_files:
            with open(label_file, 'r') as f:
                for line in f.readlines():
                    if line.strip():
                        class_id = int(line.split()[0])
                        class_counts[class_id] += 1
                        object_counts[label_file.stem] += 1
        
        if class_counts:
            print(f"\n  Class Distribution:")
            classes = ['metal', 'plastic', 'paper']
            for class_id, count in sorted(class_counts.items()):
                class_name = classes[class_id] if class_id < len(classes) else f"Unknown ({class_id})"
                pct = (count / sum(class_counts.values())) * 100
                print(f"    {class_name}: {count} objects ({pct:.1f}%)")
        
        # Statistics
        if object_counts:
            avg_objects = sum(object_counts.values()) / len(object_counts)
            max_objects = max(object_counts.values())
            min_objects = min(object_counts.values())
            
            print(f"\n  Label Statistics:")
            print(f"    Avg objects per image: {avg_objects:.2f}")
            print(f"    Max objects in image: {max_objects}")
            print(f"    Min objects in image: {min_objects}")
            
            # Images with no objects
            zero_object_images = [img for img, count in object_counts.items() if count == 0]
            if zero_object_images:
                print(f"    ⚠ Images with NO objects: {len(zero_object_images)}")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS:")
    print("="*70)
    print("""
1. CHECK CLASS IMBALANCE:
   - If one class dominates, the model will struggle with other classes
   - Consider collecting more data for underrepresented classes
   
2. CHECK LABEL QUALITY:
   - Verify paper/plastic annotations are correct
   - Mixed up labels will cause misclassification
   - Use a validation set to spot-check annotations
   
3. VERIFY ANNOTATIONS:
   - Manually inspect some label files to ensure format is correct
   - Check that bounding boxes match object locations
   
4. DATASET SIZE:
   - Aim for 500+ images per class for reliable detection
   - More diverse lighting and angles = better generalization
    """)

if __name__ == '__main__':
    analyze_dataset()
