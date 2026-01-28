#!/usr/bin/env python3
"""
Check exact image and label counts in train/valid folders
"""

from pathlib import Path

base_path = Path(r'C:\Users\logy1\OneDrive - Aston University\Final Year Project\Combined_Dataset\Combined Dataset')

for split in ['train', 'valid']:
    split_path = base_path / split
    
    images_dir = split_path / 'images'
    labels_dir = split_path / 'labels'
    
    # Count files
    image_files = list(images_dir.glob('*')) if images_dir.exists() else []
    label_files = list(labels_dir.glob('*.txt')) if labels_dir.exists() else []
    
    print(f"\n{split.upper()}:")
    print(f"  Images: {len(image_files)}")
    print(f"  Labels: {len(label_files)}")
    
    # Find mismatches
    image_stems = {img.stem for img in image_files}
    label_stems = {lbl.stem for lbl in label_files}
    
    missing_labels = image_stems - label_stems
    orphaned_labels = label_stems - image_stems
    
    if missing_labels:
        print(f"  ❌ IMAGES WITHOUT LABELS ({len(missing_labels)}):")
        for stem in sorted(list(missing_labels)[:10]):
            print(f"     - {stem}")
        if len(missing_labels) > 10:
            print(f"     ... and {len(missing_labels) - 10} more")
    
    if orphaned_labels:
        print(f"  ❌ ORPHANED LABELS ({len(orphaned_labels)}):")
        for stem in sorted(list(orphaned_labels)[:10]):
            print(f"     - {stem}")
        if len(orphaned_labels) > 10:
            print(f"     ... and {len(orphaned_labels) - 10} more")
    
    if not missing_labels and not orphaned_labels:
        print(f"  ✅ All files matched!")
