from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Open the video file
video_path = "crosstrafic.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                annotated_frame,
                [points],
                isClosed=False,
                color=(230, 230, 230),
                thickness=10,
            )

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()


# import threading

# import cv2
# from ultralytics import YOLO


# def run_tracker_in_thread(filename, model):
#     video = cv2.VideoCapture(filename)
#     frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#     for _ in range(frames):
#         ret, frame = video.read()
#         if ret:
#             results = model.track(source=frame, persist=True)
#             res_plotted = results[0].plot()
#             cv2.imshow("p", res_plotted)
#             if cv2.waitKey(1) == ord("q"):
#                 break


# # Load the models
# model1 = YOLO("yolov8n.pt")
# model2 = YOLO("yolov8n-seg.pt")

# # Define the video files for the trackers
# video_file1 = "trafic.mp4"
# video_file2 = "crosstrafic.mp4"

# # Create the tracker threads
# tracker_thread1 = threading.Thread(
#     target=run_tracker_in_thread, args=(video_file1, model1), daemon=True
# )
# tracker_thread2 = threading.Thread(
#     target=run_tracker_in_thread, args=(video_file2, model2), daemon=True
# )

# # Start the tracker threads
# tracker_thread1.start()
# tracker_thread2.start()

# # Wait for the tracker threads to finish
# tracker_thread1.join()
# tracker_thread2.join()

# # Clean up and close windows
# cv2.destroyAllWindows()