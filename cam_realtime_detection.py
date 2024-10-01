import cv2
import numpy as np

# Load the COCO class labels the model was trained on
labels = open("cocomodel/coco-labels-paper.txt").read().strip().split(",")

# Load the pre-trained MobileNet SSD model from disk
net = cv2.dnn.readNetFromCaffe("cocomodel/deploy.prototxt", "cocomodel/mobilenet_iter_73000.caffemodel")

# Access the webcam feed
cap = cv2.VideoCapture(0)

# Loop through each frame captured from the webcam
while True:
    # Capture frame-by-frame from the camera
    ret, frame = cap.read()
    
    if not ret:
        break

    # Get the dimensions of the frame
    (h, w) = frame.shape[:2]

    # Prepare the frame for object detection by creating a blob
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # Set the blob as input to the network
    net.setInput(blob)

    # Perform forward pass and get the detections
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the confidence is above a threshold
        if confidence > 0.2:  # Confidence threshold can be tuned for accuracy
            # Extract the index of the class label from the detections
            idx = int(detections[0, 0, i, 1])

            # Compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the bounding box and the label text on the frame
            label = "{}: {:.2f}%".format(labels[idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the output frame
    cv2.imshow("Object Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
