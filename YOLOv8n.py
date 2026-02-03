import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

# Optional: reduce OpenCV thread contention on Raspberry Pi
try:
    cv2.setNumThreads(1)
except Exception:
    pass

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), index of USB camera ("usb0"), or index of Picamera ("picamera0")',
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.5)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise, match source resolution',
                    default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')

# ⭐ NEW speed / stability controls
parser.add_argument('--imgsz', help='YOLO inference image size (e.g. 256, 320). Smaller = faster on Pi.',
                    default=256)
parser.add_argument('--interval', help='Run YOLO every N seconds (e.g. 0.5). Camera stays smooth between.',
                    default=0.5)

args = parser.parse_args()

# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

IMG_SZ = int(args.imgsz)
DETECTION_INTERVAL = float(args.interval)

# Check if model file exists and is valid
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# Load the model into memory and get label map
model = YOLO(model_path, task='detect')
labels = model.names

# Parse input to determine if image source is a file, folder, video, or USB camera
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

# Parse user-specified display resolution (make safe for Picamera)
resize = False
resW, resH = None, None
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])
else:
    # Picamera2 needs a configured size; safe default
    if source_type == 'picamera':
        resize = True
        resW, resH = 640, 480

# Check if recording is valid and set up recording
if record:
    if source_type not in ['video','usb','picamera']:
        print('Recording only works for video and camera sources. Please try again.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at.')
        sys.exit(0)

    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

# Load or initialize image source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(img_source + '/*')
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
elif source_type == 'video' or source_type == 'usb':
    cap_arg = img_source if source_type == 'video' else usb_idx
    cap = cv2.VideoCapture(cap_arg)

    # Set camera or video resolution if specified by user
    if user_res:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)

elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
    cap.start()

# Set bounding box colors (Tableau 10)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 60
img_count = 0

# ⭐ NEW: last detection cache so we don't infer every frame
last_detection_time = 0.0
last_detections = None
last_infer_ms = 0.0

# Begin inference loop
while True:
    t_start = time.perf_counter()

    # Load frame from image source
    if source_type == 'image' or source_type == 'folder':
        if img_count >= len(imgs_list):
            print('All images have been processed. Exiting program.')
            sys.exit(0)
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
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
            print('Unable to read frames from the Picamera. Exiting program.')
            break

        # ✅ CRITICAL FIX: Picamera2 gives RGB; OpenCV/YOLO pipeline expects BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Resize frame to desired display resolution
    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # ✅ Run inference only every DETECTION_INTERVAL seconds
    now = time.time()
    if (now - last_detection_time) >= DETECTION_INTERVAL:
        infer_start = time.perf_counter()

        # ✅ CRITICAL FIX: stronger NMS + cap max detections to avoid 300 boxes
        results = model.predict(
            frame,
            imgsz=IMG_SZ,
            conf=min_thresh,
            iou=0.35,
            max_det=20,
            verbose=False
        )

        last_detections = results[0].boxes if results and len(results) else None
        last_detection_time = now
        last_infer_ms = (time.perf_counter() - infer_start) * 1000.0

    detections = last_detections

    # Initialize object counter
    object_count = 0

    # Go through detections
    if detections is not None and len(detections) > 0:
        for i in range(len(detections)):
            xyxy = detections[i].xyxy.cpu().numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)

            classidx = int(detections[i].cls.item())
            classname = labels[classidx]
            conf = detections[i].conf.item()

            if conf > min_thresh:
                color = bbox_colors[classidx % 10]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

                label = f'{classname}: {int(conf*100)}%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(frame,
                              (xmin, label_ymin-labelSize[1]-10),
                              (xmin+labelSize[0], label_ymin+baseLine-10),
                              color, cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin-7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                object_count += 1

    # Draw framerate + debug info
    if source_type in ['video', 'usb', 'picamera']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
        cv2.putText(frame, f'YOLO interval: {DETECTION_INTERVAL:.2f}s  imgsz: {IMG_SZ}', (10,45),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,255), 2)
        cv2.putText(frame, f'Infer: {last_infer_ms:.0f} ms', (10,70),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,255), 2)

    # Display detection results
    cv2.putText(frame, f'Number of objects: {object_count}', (10,95),
                cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)

    cv2.imshow('YOLO detection results', frame)
    if record:
        recorder.write(frame)

    # Key handling
    if source_type == 'image' or source_type == 'folder':
        key = cv2.waitKey()
    else:
        key = cv2.waitKey(1)

    if key == ord('q') or key == ord('Q'):
        break
    elif key == ord('s') or key == ord('S'):
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'):
        cv2.imwrite('capture.png', frame)

    # FPS calc
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))

    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(frame_rate_calc)
    avg_frame_rate = np.mean(frame_rate_buffer)

# Clean up
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type == 'video' or source_type == 'usb':
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record:
    recorder.release()
cv2.destroyAllWindows()
