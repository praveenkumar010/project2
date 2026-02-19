from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os

def process_video(input_video_path):

    model = YOLO("yolov8x.pt")   # local model file

    cap = cv2.VideoCapture(input_video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = "output_speed.mp4"

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    LINE_Y1 = int(h * 0.38)
    LINE_Y2 = int(h * 0.60)

    DISTANCE_METERS = 15
    SPEED_LIMIT = 50

    entry_frame = {}
    vehicle_speed = {}
    prev_y = {}

    frame_no = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1

        results = model.track(
            frame,
            persist=True,
            conf=0.35,
            classes=[2,3,5,7],
            tracker="bytetrack.yaml"
        )

        if results[0].boxes is not None:
            boxes = results[0].boxes
            ids = boxes.id if boxes.id is not None else [None]*len(boxes)

            for box, track_id in zip(boxes, ids):

                if track_id is None:
                    continue

                track_id = int(track_id)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = int((x1+x2)/2)
                cy = int((y1+y2)/2)

                # Right lane only
                if cx < w * 0.45:
                    continue

                if track_id not in prev_y:
                    prev_y[track_id] = cy
                    continue

                movement = cy - prev_y[track_id]
                prev_y[track_id] = cy

                if movement < 2:
                    continue

                if track_id not in entry_frame:
                    if cy > LINE_Y1:
                        entry_frame[track_id] = frame_no

                elif track_id not in vehicle_speed:
                    if cy > LINE_Y2:
                        time_taken = (frame_no - entry_frame[track_id]) / fps
                        if time_taken > 0:
                            speed = (DISTANCE_METERS / time_taken) * 3.6
                            if 10 < speed < 120:
                                vehicle_speed[track_id] = speed

                color = (0,255,0)
                label = f"ID {track_id}"

                if track_id in vehicle_speed:
                    spd = vehicle_speed[track_id]
                    if spd > SPEED_LIMIT:
                        color = (0,0,255)
                        label = f"{spd:.1f} km/h SPEEDING"
                    else:
                        label = f"{spd:.1f} km/h"

                cv2.rectangle(frame,(x1,y1),(x2,y2),color,3)
                cv2.putText(frame,label,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

        cv2.line(frame,(0,LINE_Y1),(w,LINE_Y1),(255,0,255),2)
        cv2.line(frame,(0,LINE_Y2),(w,LINE_Y2),(0,255,255),2)

        out.write(frame)

    cap.release()
    out.release()

    return output_path