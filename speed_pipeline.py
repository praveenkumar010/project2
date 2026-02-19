from ultralytics import YOLO
import cv2
import torch
import numpy as np
import pandas as pd

# -------------------------------------------------
# LOAD HEAVY MODEL (accurate > fast)
# -------------------------------------------------
model = YOLO("yolov8x.pt")

device = 0 if torch.cuda.is_available() else "cpu"

# -------------------------------------------------
# MAIN FUNCTION CALLED FROM STREAMLIT
# -------------------------------------------------
def process_video(video_path):

    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = "output_speed.mp4"

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w,h)
    )

    # -------------------------------------------------
    # SPEED LINE SETTINGS (RIGHT LANE ONLY)
    # -------------------------------------------------
    LINE_Y1 = int(h * 0.35)
    LINE_Y2 = int(h * 0.60)

    DISTANCE_METERS = 10
    SPEED_LIMIT = 50

    entry_frame = {}
    vehicle_speed = {}
    speed_done = {}

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
            iou=0.5,
            device=device,
            classes=[2,3,5,7]   # car, bike, bus, truck
        )

        if results[0].boxes is not None:

            boxes = results[0].boxes
            ids = boxes.id if boxes.id is not None else [None]*len(boxes)

            for box, track_id in zip(boxes, ids):
                if track_id is None:
                    continue

                track_id = int(track_id)
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                cx = int((x1+x2)/2)
                cy = int((y1+y2)/2)

                # 🚫 IGNORE LEFT LANE VEHICLES
                if cx < w * 0.45:
                    continue

                # -------- ENTRY LINE --------
                if track_id not in entry_frame:
                    if cy > LINE_Y1:
                        entry_frame[track_id] = frame_no

                # -------- EXIT LINE --------
                elif track_id not in speed_done:
                    if cy > LINE_Y2:
                        time_taken = (frame_no - entry_frame[track_id]) / fps

                        if time_taken > 0:
                            speed = (DISTANCE_METERS / time_taken) * 3.6
                            vehicle_speed[track_id] = speed
                            speed_done[track_id] = True

                # -------- DRAW BOX --------
                color = (0,255,0)
                label = f"ID {track_id}"

                if track_id in vehicle_speed:
                    spd = vehicle_speed[track_id]
                    label = f"{spd:.1f} km/h"

                    if spd > SPEED_LIMIT:
                        color = (0,0,255)
                        label = f"FAST {spd:.1f} km/h"

                cv2.rectangle(frame,(x1,y1),(x2,y2),color,3)
                cv2.putText(frame,label,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

        # draw speed lines
        cv2.line(frame,(0,LINE_Y1),(w,LINE_Y1),(255,0,255),3)
        cv2.line(frame,(0,LINE_Y2),(w,LINE_Y2),(0,255,255),3)

        out.write(frame)

    cap.release()
    out.release()

    # -------------------------------------------------
    # CREATE VIOLATION TABLE
    # -------------------------------------------------
    records = []

    for vid, spd in vehicle_speed.items():
        if spd > SPEED_LIMIT:
            status = "OverSpeed 🚨"
        else:
            status = "Normal ✅"

        records.append([vid, round(spd,2), status])

    df = pd.DataFrame(records, columns=["Vehicle ID","Speed (km/h)","Violation"])

    return output_path, df
