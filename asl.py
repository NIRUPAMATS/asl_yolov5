from matplotlib import pyplot as plt
import numpy as np
import torch
import cv2

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

# Open the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Webcam frame settings
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():

    # Capture frame, if all went good then 'ret==True'
    ret, frame = cap.read()

    # Make detections
    results = model(frame)

    # Plot detections
    cv2.imshow('YOLO', np.squeeze(results.render()))

    # If we press the exit-buttom 'q' we end the webcam caption
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close everything in the end
cap.release()
cv2.destroyAllWindows()