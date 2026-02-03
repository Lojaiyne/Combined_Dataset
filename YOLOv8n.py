import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------------
# Args
# ----------------------------
parser = argparse.ArgumentParser()

parser.add_argument('--model', required=True,
                    help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")')

parser.add_argument('--source', required=True,
                    help='Image source: image file, folder, video file, "usb0", or "picamera0"')

parser.add_argument('--thresh', default=0.5,
                    help='Minimum confidence threshold (example: "0.4")')

parser.add_argument('--resolution', default=None,
                    help='Display/capture resolution WxH (example: "640x480"). If not set: use safe defaults.')

parser.add_argument('--record', action='store_true',
                    help='Record results to demo1.avi (requires --resolution).')

# SPEED OPTIONS
parser.add_argument('--imgsz', default=320, type=int,
                    help='YOLO inference image size (smaller = faster). Try 320 or 256.')

parser.add_argument('--stride', default=1, type=int,
                    help='Process every Nth frame (2 = every 2nd frame).')

parser.add_argument('--draw', action='store_true',
                    help='Draw boxes/labels. If omitted, FPS is higher (still runs detection).')

args = parser.parse_args()

model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record
imgsz = int(args.imgsz)
stride = max(1, int(args.stride))
DRAW = bool(args.draw)

# ----------------------------
# Validate model
# ----------------------------
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or model was not found.')
    sys.exit(0)

# ----------------------------
# Speed: OpenCV tuning (helps on Pi)
# ----------------------------
cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(1)
except Exception:
    pass

# ----------------------------
# Load model
# ----------------------------
model = YOLO(model_path, task='detect')
labels = model.names
try:
    model.fuse()  # small free speed-up
except Exception:
    pass

# ----------------------------
# Determine source type
# ----------------------------
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

# ----------------------------
# Resolution handling
# - If user doesn't set it, choose fast Pi-friendly defaults
# ----------------------------
resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.split('x'))
else:
    # SAFE DEFAULTS (fast)
    if source_type in ['usb', 'video', 'picamera']:
        resW, resH = 640, 480
        resize = True  # keep pipeline consistent
    else:
        resW, resH = None, None

# ----------------------------
# Recording setup
# ----------------------------
if record:
    if source_type not in ['video', 'usb', 'picamera']:
        print('Recording only works for video/camera sources.')
        sys.exit(0)
    if not user_res:
        print('Please specify --resolution to record video at.')
        sys.exit(0)

    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'),
                               record_fps, (resW, resH))

# ----------------------------
# Load/init source
# ----------------------------
if source_type == 'image':
    imgs_list = [img_source]

elif source_type == 'folder':
    imgs_list = []
    for file in glob.glob(img_source + '/*'):
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)

elif source_type in ['video', 'usb']:
    cap_arg = img_source if source_type == 'video' else usb_idx
    cap = cv2.VideoCapture(cap_arg)

    # reduce latency a bit (may not work on all backends)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    # Set camera/video resolution if available
    if resize:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)

elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    # Ensure resW/resH exist even if --resolution not passed
    cap.configure(cap.create_video_configuration(
        main={"format": "RGB888", "size": (resW, resH)}
    ))
    cap.start()

# ----------------------------
# Colors
# ----------------------------
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
               (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# ----------------------------
# FPS tracking
# ----------------------------
avg_frame_rate = 0.0
frame_rate_buffer = []
fps_avg_len = 100
img_count = 0
frame_id = 0

# ----------------------------
# Loop
# ----------------------------
while True:
    t_start = time.perf_counter()

    # Grab frame
    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            print('All images have been processed. Exiting program.')
            sys.exit(0)
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1

    elif source_type == 'video':
        ret, frame = cap.read()
        if not ret:
            print('Reached end of the video file. Exiting program.')
            break

    elif source_type == 'usb':
        ret, frame = cap.read()
        if (frame is None) or (not ret):
            print('Unable to read frames from the camera. Exiting program.')
            break

    elif source_type == 'picamera':
        frame = cap.capture_array()
        if frame is None:
            print('Unable to read frames from Picamera. Exiting program.')
            break

    # Resize (fast, consistent)
    if resize and (resW is not None) and (resH is not None):
        frame = cv2.resize(frame, (resW, resH), interpolation=cv2.INTER_LINEAR)

    # Frame skipping
    frame_id += 1
    do_infer = (frame_id % stride == 0)

    object_count = 0

    if do_infer:
        # Inference (imgsz is a big speed knob)
        results = model.predict(frame, imgsz=imgsz, conf=min_thresh, verbose=False)
        detections = results[0].boxes

        if detections is not None and len(detections) > 0:
            # Use numpy arrays directly (less python overhead)
            xyxy = detections.xyxy.cpu().numpy().astype(int)
            cls = detections.cls.cpu().numpy().astype(int)
            conf = detections.conf.cpu().numpy()

            object_count = int((conf > min_thresh).sum())

            if DRAW:
                for j in range(len(conf)):
                    if conf[j] <= min_thresh:
                        continue

                    xmin, ymin, xmax, ymax = xyxy[j]
                    classidx = int(cls[j])
                    classname = labels[classidx]
                    color = bbox_colors[classidx % 10]

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

                    label = f'{classname}: {int(conf[j]*100)}%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_ymin = max(ymin, labelSize[1] + 10)
                    cv2.rectangle(frame,
                                  (xmin, label_ymin - labelSize[1] - 10),
                                  (xmin + labelSize[0], label_ymin + baseLine - 10),
                                  color, cv2.FILLED)
                    cv2.putText(frame, label, (xmin, label_ymin - 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Overlay text (optional but cheap)
    if source_type in ['video', 'usb', 'picamera']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(frame, f'Objects: {object_count}', (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow('YOLO detection results', frame)
    if record:
        recorder.write(frame)

    # Wait key
    if source_type in ['image', 'folder']:
        key = cv2.waitKey()
    else:
        key = cv2.waitKey(1)

    if key in (ord('q'), ord('Q')):
        break
    elif key in (ord('s'), ord('S')):
        cv2.waitKey()
    elif key in (ord('p'), ord('P')):
        cv2.imwrite('capture.png', frame)

    # FPS calc
    t_stop = time.perf_counter()
    dt = max(1e-6, (t_stop - t_start))
    frame_rate_calc = 1.0 / dt

    frame_rate_buffer.append(frame_rate_calc)
    if len(frame_rate_buffer) > fps_avg_len:
        frame_rate_buffer.pop(0)
    avg_frame_rate = float(np.mean(frame_rate_buffer))

# ----------------------------
# Cleanup
# ----------------------------
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type in ['video', 'usb']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record:
    recorder.release()
cv2.destroyAllWindows()
