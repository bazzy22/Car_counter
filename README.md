# Car Counter Project (YOLOv8 + SORT)
This project implements an automatic vehicle counting system (cars, trucks, buses, and motorcycles) in a video stream. It uses the YOLOv8 object detection model to identify vehicles and the SORT (Simple Online and Realtime Tracking) algorithm to keep track of detected objects and count them only once as they cross a defined boundary line.

<p align="center">
  <img src="https://github.com/user-attachments/assets/c6dbc88c-297d-4947-87fc-d807c0c9b4e1" alt="unknown_2025 09 29-14 57-ezgif com-optimize">
</p>

## Technologies Used
Python: Main programming language.

OpenCV (cv2): Video management, masking, and visualization.

Ultralytics YOLO: Framework for object detection (YOLOv8n).

Numpy: Array manipulation for detection and tracking.

cvzone: Used to simplify the display of rectangles and text.

SORT: Implementation of a multi-object tracker. https://github.com/abewley/sort
