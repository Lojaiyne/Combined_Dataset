import os
import random
import cv2
import matplotlib.pyplot as plt

# Set these paths to your dataset
IMAGES_DIR = os.path.join('Combined Dataset', 'train', 'images')
LABELS_DIR = os.path.join('Combined Dataset', 'train', 'labels')
CLASSES_PATH = os.path.join('Combined Dataset', 'train', 'classes.txt')

# Load class names
def load_classes(classes_path):
    with open(classes_path, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

# Parse YOLO label file
def parse_label(label_path):
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls, x, y, w, h = map(float, parts)
                boxes.append((int(cls), x, y, w, h))
    return boxes

# Draw bounding boxes on image
def draw_boxes(img, boxes, class_names):
    h, w = img.shape[:2]
    for cls, x, y, bw, bh in boxes:
        x1 = int((x - bw/2) * w)
        y1 = int((y - bh/2) * h)
        x2 = int((x + bw/2) * w)
        y2 = int((y + bh/2) * h)
        color = (0, 255, 0) if cls == 0 else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = class_names[cls] if cls < len(class_names) else str(cls)
        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img

# Visualize random images for a specific class
def visualize_class_images(class_name, num_images=10):
    class_names = load_classes(CLASSES_PATH)
    if class_name not in class_names:
        print(f'Class "{class_name}" not found in classes.txt')
        return
    class_idx = class_names.index(class_name)
    label_files = os.listdir(LABELS_DIR)
    files = []
    for label_file in label_files:
        label_path = os.path.join(LABELS_DIR, label_file)
        boxes = parse_label(label_path)
        for cls, *_ in boxes:
            if cls == class_idx:
                files.append(label_file)
                break
    print(f'Class: {class_name} ({class_idx}) - {len(files)} images')
    if not files:
        print('No images found for this class.')
        return
    sample_files = random.sample(files, min(num_images, len(files)))
    for label_file in sample_files:
        img_file = label_file.replace('.txt', '.jpg')
        img_path = os.path.join(IMAGES_DIR, img_file)
        if not os.path.exists(img_path):
            img_file = label_file.replace('.txt', '.png')
            img_path = os.path.join(IMAGES_DIR, img_file)
            if not os.path.exists(img_path):
                print(f'Image not found for label: {label_file}')
                continue
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = parse_label(os.path.join(LABELS_DIR, label_file))
        img = draw_boxes(img, boxes, class_names)
        plt.figure(figsize=(6,6))
        plt.imshow(img)
        plt.title(f'{class_name} - {img_file}')
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    visualize_class_images('metal', num_images=10)
