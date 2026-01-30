#!/usr/bin/env python3
"""
Data Cleanup Script
Removes orphaned labels and images without labels
"""

from pathlib import Path
import shutil

def cleanup_dataset():
    base_path = Path(r'C:\Users\logy1\OneDrive - Aston University\Final Year Project\Combined_Dataset\Combined Dataset')
    
    print("="*70)
    print("DATASET CLEANUP")
    print("="*70)
    
    for split in ['train', 'valid']:
        split_path = base_path / split
        if not split_path.exists():
            continue
            
        images_dir = split_path / 'images'
        labels_dir = split_path / 'labels'
        
        image_files = list(images_dir.glob('*.*')) if images_dir.exists() else []
        label_files = list(labels_dir.glob('*.txt')) if labels_dir.exists() else []
        
        image_stems = {img.stem for img in image_files}
        label_stems = {lbl.stem for lbl in label_files}
        
        # Find problematic files
        missing_labels = image_stems - label_stems
        orphaned_labels = label_stems - image_stems
        
        print(f"\n{split.upper()} SET:")
        
        # Remove images without labels
        if missing_labels:
            print(f"  Removing {len(missing_labels)} images without labels...")
            for stem in missing_labels:
                for img_file in images_dir.glob(f"{stem}.*"):
                    img_file.unlink()
                    print(f"    - {img_file.name}")
        
        # Remove orphaned labels
        if orphaned_labels:
            print(f"  Removing {len(orphaned_labels)} orphaned label files...")
            for stem in orphaned_labels:
                for label_file in labels_dir.glob(f"{stem}.txt"):
                    label_file.unlink()
                    print(f"    - {label_file.name}")
        
        # Summary
        image_files = list(images_dir.glob('*.*'))
        label_files = list(labels_dir.glob('*.txt'))
        print(f"\n  After cleanup:")
        print(f"    Images: {len(image_files)}")
        print(f"    Labels: {len(label_files)}")

if __name__ == '__main__':
    cleanup_dataset()
    print("\n" + "="*70)
    print("Cleanup complete! Run analyze_dataset.py to verify.")
    print("="*70)
