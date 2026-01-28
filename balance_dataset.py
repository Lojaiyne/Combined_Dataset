#!/usr/bin/env python3
"""
Dataset Balancing Script
Duplicates underrepresented classes to balance the validation set
"""

from pathlib import Path
import shutil
import random
from collections import defaultdict

def balance_dataset():
    base_path = Path(r'C:\Users\logy1\OneDrive - Aston University\Final Year Project\Combined_Dataset\Combined Dataset')
    
    print("="*70)
    print("DATASET BALANCING - VALIDATION SET")
    print("="*70)
    
    valid_path = base_path / 'valid'
    images_dir = valid_path / 'images'
    labels_dir = valid_path / 'labels'
    
    # Read class distribution
    class_counts = defaultdict(int)
    image_classes = defaultdict(list)  # stem -> class_id (majority class in image)
    
    label_files = list(labels_dir.glob('*.txt'))
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            classes_in_image = []
            for line in f.readlines():
                if line.strip():
                    class_id = int(line.split()[0])
                    classes_in_image.append(class_id)
                    class_counts[class_id] += 1
            
            # Store majority class for this image
            if classes_in_image:
                majority_class = max(set(classes_in_image), key=classes_in_image.count)
                image_classes[label_file.stem] = majority_class
    
    print("\nCurrent validation set class distribution:")
    total_objects = sum(class_counts.values())
    classes = ['metal', 'plastic', 'paper']
    
    class_info = {}
    for class_id, count in sorted(class_counts.items()):
        class_name = classes[class_id]
        pct = (count / total_objects) * 100
        class_info[class_id] = count
        print(f"  {class_name}: {count} objects ({pct:.1f}%)")
    
    # Calculate target: balance to have roughly equal objects
    # Target is to have each class at ~33%
    target_count = total_objects // 3
    
    print(f"\nTarget balance: ~{target_count} objects per class")
    print(f"(Total objects: {total_objects})")
    
    # Find images to duplicate for each underrepresented class
    duplicates_to_create = {}
    
    for class_id in [0, 1, 2]:  # metal, plastic, paper
        current_count = class_info.get(class_id, 0)
        if current_count < target_count:
            shortage = target_count - current_count
            duplicates_to_create[class_id] = shortage
            print(f"\n  {classes[class_id]}: Need {shortage} more objects")
    
    # Get images for each class
    images_by_class = defaultdict(list)
    for stem, class_id in image_classes.items():
        images_by_class[class_id].append(stem)
    
    # Duplicate images to balance classes
    total_duplicated = 0
    
    for class_id, shortage in duplicates_to_create.items():
        if class_id not in images_by_class:
            print(f"  ⚠ No images found for class {classes[class_id]}")
            continue
        
        candidate_images = images_by_class[class_id]
        print(f"\n  Duplicating {classes[class_id]} images...")
        
        # Calculate how many copies of each image needed
        num_images = len(candidate_images)
        copies_needed = shortage // class_info[class_id] + 1
        
        duplicated_count = 0
        for i in range(copies_needed):
            for stem in candidate_images:
                if duplicated_count >= shortage:
                    break
                
                # Find image file (could be jpg, png, etc)
                image_files = list(images_dir.glob(f"{stem}.*"))
                label_file = labels_dir / f"{stem}.txt"
                
                if image_files and label_file.exists():
                    for img_file in image_files:
                        # Create a new name with suffix
                        new_stem = f"{stem}_dup{i}_{duplicated_count}"
                        new_img_path = images_dir / f"{new_stem}{img_file.suffix}"
                        new_label_path = labels_dir / f"{new_stem}.txt"
                        
                        # Copy files
                        shutil.copy2(img_file, new_img_path)
                        shutil.copy2(label_file, new_label_path)
                        
                        duplicated_count += 1
                        total_duplicated += 1
                        
                        if duplicated_count % 50 == 0:
                            print(f"    Created {duplicated_count} duplicates...")
            
            if duplicated_count >= shortage:
                break
        
        print(f"    ✓ Created {duplicated_count} duplicate images for {classes[class_id]}")
    
    print(f"\n{'='*70}")
    print(f"✓ Created {total_duplicated} duplicate images total")
    print(f"{'='*70}")
    
    # Show new distribution
    print("\nNew validation set statistics:")
    class_counts_new = defaultdict(int)
    
    label_files = list(labels_dir.glob('*.txt'))
    for label_file in label_files:
        with open(label_file, 'r') as f:
            for line in f.readlines():
                if line.strip():
                    class_id = int(line.split()[0])
                    class_counts_new[class_id] += 1
    
    total_new = sum(class_counts_new.values())
    for class_id in range(3):
        count = class_counts_new.get(class_id, 0)
        pct = (count / total_new * 100) if total_new > 0 else 0
        print(f"  {classes[class_id]}: {count} objects ({pct:.1f}%)")
    
    print(f"\nTotal objects: {total_new}")
    print(f"Total images: {len(list(images_dir.glob('*.*')))}")
    
    print("\n" + "="*70)
    print("Run analyze_dataset.py to verify the new distribution!")
    print("="*70)

if __name__ == '__main__':
    balance_dataset()
