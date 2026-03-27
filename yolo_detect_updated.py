import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO
from gpiozero import OutputDevice

# TB6612FNG I2C motor driver library
try:
    from raspberry_i2c_tb6612fng import MotorDriverTB6612FNG, TB6612FNGMotors
except ImportError:
    print('ERROR: Missing motor driver library.')
    print('Install it with: python3 -m pip install raspberry-i2c-tb6612fng')
    sys.exit(1)

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

# solenoid control arguments
parser.add_argument('--solenoid_pins', default="17,27,22",
                    help='BCM GPIO pins for solenoids, comma-separated. Example: "17,27,22"')
parser.add_argument('--solenoid_map', default="plastic:0,metal:1,paper:2",
                    help='Map class->solenoid index. Example: "plastic:0,metal:1,paper:2"')
parser.add_argument('--pulse_ms', default=120, type=int,
                    help='How long to fire the solenoid (milliseconds).')
parser.add_argument('--cooldown_ms', default=700, type=int,
                    help='Minimum time between firings for the same solenoid (milliseconds).')
parser.add_argument('--min_area', default=0, type=int,
                    help='Ignore tiny detections by bbox area in pixels. 0 disables.')
parser.add_argument('--trigger_x', type=float, default=0.7,
                    help='Fire when bbox center is past this fraction of frame width (0..1).')

# motor driver arguments
parser.add_argument('--motor_enable', action='store_true',
                    help='Enable conveyor motor through TB6612FNG I2C motor driver.')
parser.add_argument('--motor_channel', default='A', choices=['A', 'B'],
                    help='Motor driver channel to use: A or B.')
parser.add_argument('--motor_speed', type=int, default=120,
                    help='Motor speed 0..255. Start around 80-140.')
parser.add_argument('--motor_addr', type=lambda x: int(x, 0), default=0x14,
                    help='I2C address of motor driver. Default is 0x14.')
parser.add_argument('--motor_start_delay', type=float, default=1.0,
                    help='Seconds to wait after starting motor before inference.')

args = parser.parse_args()

# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

# Check if model file exists and is valid
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# Load the model into memory and get label map
model = YOLO(model_path, task='detect')
labels = model.names
print("MODEL LABELS:", labels)

# -----------------------------
# Solenoid setup
# -----------------------------
solenoid_pins = [int(x.strip()) for x in args.solenoid_pins.split(',') if x.strip()]
solenoids = [OutputDevice(pin, active_high=True, initial_value=False) for pin in solenoid_pins]

# Parse "plastic:0,metal:1,paper:2" into dict
solenoid_map = {}
for pair in args.solenoid_map.split(','):
    pair = pair.strip()
    if not pair:
        continue
    cls, idx = pair.split(':')
    solenoid_map[cls.strip()] = int(idx.strip())

pulse_s = args.pulse_ms / 1000.0
cooldown_s = args.cooldown_ms / 1000.0

# For each solenoid: when it should turn OFF, and last time it fired
sol_off_at = [0.0] * len(solenoids)
sol_last_fire = [0.0] * len(solenoids)

def request_fire(sol_idx: int, now: float):
    if sol_idx < 0 or sol_idx >= len(solenoids):
        print("Bad solenoid index:", sol_idx)
        return
    if (now - sol_last_fire[sol_idx]) < cooldown_s:
        print("Cooldown blocking solenoid", sol_idx)
        return
    print("Solenoid ON idx", sol_idx)
    solenoids[sol_idx].on()
    sol_off_at[sol_idx] = now + pulse_s
    sol_last_fire[sol_idx] = now

# -----------------------------
# Motor setup (TB6612FNG I2C)
# -----------------------------
motor_driver = None
motor_channel = None
motor_speed = max(0, min(255, args.motor_speed))
belt_stopped = False
sort_done = False   # stops repeated triggering for same item

def motor_run():
    global belt_stopped
    if motor_driver is not None and motor_channel is not None:
        motor_driver.dc_motor_run(motor_channel, motor_speed)
        belt_stopped = False
        print("Motor RUN")

def motor_stop():
    global belt_stopped
    if motor_driver is not None and motor_channel is not None:
        motor_driver.dc_motor_run(motor_channel, 0)
        belt_stopped = True
        print("Motor STOP")

if args.motor_enable:
    try:
        motor_driver = MotorDriverTB6612FNG(address=args.motor_addr)
        motor_channel = TB6612FNGMotors.MOTOR_CHA if args.motor_channel == 'A' else TB6612FNGMotors.MOTOR_CHB

        print(f"Starting motor on channel {args.motor_channel} at speed {motor_speed}, I2C addr {hex(args.motor_addr)}")
        motor_run()
        time.sleep(args.motor_start_delay)
    except Exception as e:
        print("ERROR: Could not start motor driver:", e)
        print("Check: I2C enabled, wiring correct, board detected at the right address.")
        sys.exit(1)

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

# Parse user-specified display resolution
resize = False
resW, resH = None, None
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# Check if recording is valid and set up recording
if record:
    if source_type not in ['video','usb']:
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

    if source_type == 'video':
        cap_arg = img_source
    elif source_type == 'usb':
        cap_arg = usb_idx
    cap = cv2.VideoCapture(cap_arg)

    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)

elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    if resize:
        cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
    else:
        cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (640, 480)}))
    cap.start()

# Set bounding box colors
bbox_colors = [
    (164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
    (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)
]

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

try:
    while True:
        now = time.perf_counter()
        t_start = now

        # Turn OFF any solenoids whose pulse time ended
        for si in range(len(solenoids)):
            if sol_off_at[si] != 0.0 and now >= sol_off_at[si]:
                solenoids[si].off()
                sol_off_at[si] = 0.0

        # Load frame from image source
        if source_type == 'image' or source_type == 'folder':
            if img_count >= len(imgs_list):
                print('All images have been processed. Exiting program.')
                break
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

        # Resize frame to desired display resolution
        if resize:
            frame = cv2.resize(frame, (resW, resH))

        frame_h, frame_w = frame.shape[:2]
        trigger_line_x = int(frame_w * args.trigger_x)

        # Draw trigger line
        cv2.line(frame, (trigger_line_x, 0), (trigger_line_x, frame_h), (0, 255, 0), 2)
        cv2.putText(frame, 'Trigger line', (trigger_line_x + 5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # Run inference on frame
        results = model(frame, verbose=False)
        detections = results[0].boxes
        object_count = 0

        # Reset trigger lock if no objects are detected anymore
        if len(detections) == 0:
            sort_done = False

        # Go through each detection
        for i in range(len(detections)):
            xyxy_tensor = detections[i].xyxy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)

            classidx = int(detections[i].cls.item())
            classname = labels[classidx]
            conf = detections[i].conf.item()

            if conf > min_thresh:
                color = bbox_colors[classidx % 10]
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

                label = f'{classname}: {int(conf*100)}%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10),
                              (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin-7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                object_count += 1

                bbox_area = (xmax - xmin) * (ymax - ymin)
                if args.min_area and bbox_area < args.min_area:
                    continue

                cx = int((xmin + xmax) / 2)

                # Stop belt when object crosses trigger line
                if (classname in solenoid_map) and (cx >= trigger_line_x) and (not sort_done):
                    print(f"TRIGGERED: {classname} crossed line at x={cx}")

                    if args.motor_enable and not belt_stopped:
                        motor_stop()

                    # Fire the correct solenoid after stopping
                    request_fire(solenoid_map[classname], now)

                    # Prevent repeated triggering for the same item
                    sort_done = True

        # Calculate and draw framerate
        if source_type in ['video', 'usb', 'picamera']:
            cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)

        motor_status = "STOPPED" if belt_stopped else ("ON" if args.motor_enable else "OFF")
        cv2.putText(frame, f'Motor: {motor_status}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)

        cv2.putText(frame, f'Number of objects: {object_count}', (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
        cv2.imshow('YOLO detection results', frame)

        if record:
            recorder.write(frame)

        # Key handling
        if source_type in ['image', 'folder']:
            key = cv2.waitKey()
        else:
            key = cv2.waitKey(5)

        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            cv2.waitKey()
        elif key == ord('p') or key == ord('P'):
            cv2.imwrite('capture.png', frame)
        elif key == ord('r') or key == ord('R'):
            # press R to restart belt manually
            if args.motor_enable:
                motor_run()
                sort_done = False

        # Calculate FPS
        t_stop = time.perf_counter()
        frame_rate_calc = float(1 / (t_stop - t_start))

        if len(frame_rate_buffer) >= fps_avg_len:
            frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)

        avg_frame_rate = np.mean(frame_rate_buffer)

finally:
    print(f'Average pipeline FPS: {avg_frame_rate:.2f}')

    if source_type in ['video', 'usb']:
        cap.release()
    elif source_type == 'picamera':
        cap.stop()

    if record:
        recorder.release()

    for s in solenoids:
        try:
            s.off()
            s.close()
        except Exception:
            pass

    if motor_driver is not None and motor_channel is not None:
        try:
            motor_driver.dc_motor_run(motor_channel, 0)
        except Exception:
            pass

    cv2.destroyAllWindows()