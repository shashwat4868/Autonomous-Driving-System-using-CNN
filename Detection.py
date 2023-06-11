import cv2
import numpy as np
import time

# Step 1: Download the YOLO model weights and configuration files
model_weights = "YOLO/yolov3.weights"
model_config = "YOLO/yolov3.cfg"
coco_labels = "YOLO/coco.names"

# Step 2: Load the YOLO model and labels
net = cv2.dnn.readNetFromDarknet(model_config, model_weights)
labels = open("YOLO/coco.names").read().strip().split("\n")


# Step 3: Load the video file
video_file = "resources/street5.mp4"
cap = cv2.VideoCapture(video_file)

# Step 4: Process the video frames
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Step 5: Perform object detection on the frame
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    start = time.time()
    outs = net.forward(output_layers)
    end = time.time()

    # Step 6: Process the detection results
    boxes = []
    confidences = []
    class_ids = []
    (h, w) = frame.shape[:2]

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                box = detection[0:4] * np.array([w, h, w, h])
                (center_x, center_y, width, height) = box.astype("int")
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Step 7: Apply non-maxima suppression to remove redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Step 8: Draw the bounding boxes and labels on the frame
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = labels[class_ids[i]]
            confidence = confidences[i]
            color = (0, 255, 0)  # Green
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Step 9: Display the frame with bounding boxes
    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Step 10: Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()
