import os

def remove_orphans(images_dir, labels_dir, image_exts=(".jpg", ".jpeg", ".png")):
    images = set([os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.lower().endswith(image_exts)])
    labels = set([os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.lower().endswith('.txt')])

    # Images without labels
    orphaned_images = images - labels
    # Labels without images
    orphaned_labels = labels - images

    for img in orphaned_images:
        for ext in image_exts:
            img_path = os.path.join(images_dir, img + ext)
            if os.path.exists(img_path):
                print(f"Removing orphaned image: {img_path}")
                os.remove(img_path)
    for lbl in orphaned_labels:
        lbl_path = os.path.join(labels_dir, lbl + '.txt')
        if os.path.exists(lbl_path):
            print(f"Removing orphaned label: {lbl_path}")
            os.remove(lbl_path)

# Set your train/validation paths
train_images = os.path.join("Combined Dataset", "train", "images")
train_labels = os.path.join("Combined Dataset", "train", "labels")
val_images = os.path.join("Combined Dataset", "valid", "images")
val_labels = os.path.join("Combined Dataset", "valid", "labels")

remove_orphans(train_images, train_labels)
remove_orphans(val_images, val_labels)
print("Orphaned files removed. Now images and labels should match.")
