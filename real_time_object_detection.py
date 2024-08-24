# import packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import pickle
import face_recognition
import pyttsx3

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
    help="minimum probability to filter weak predictions")
args = vars(ap.parse_args())

# Directory to save detected objects (current working directory)
save_dir = os.getcwd()

# Ensure the save directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# File to store saved object features and data
features_file = os.path.join(save_dir, "saved_objects.pkl")

# Load saved objects if available
if os.path.exists(features_file):
    with open(features_file, "rb") as f:
        saved_objects = pickle.load(f)
        # Debugging: Print the type of the loaded data
        print(f"[DEBUG] Loaded saved_objects type: {type(saved_objects)}")
        # Print the first item to see its structure
        if len(saved_objects) > 0:
            print(f"[DEBUG] First item in saved_objects: {saved_objects[0]}")
else:
    saved_objects = []

# Initialising the list of the 21 class labels MobileNet SSD was trained to.
CLASSES = ["aeroplane", "background", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Assigning random colors to each of the classes
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Function to speak the detected class
def speak_detection(label):
    engine.say(label)
    engine.runAndWait()

# load our serialized model
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Start the FPS timer
fps = FPS().start()

# Previous detection announcement time
last_announcement_time = time.time()

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    resized_image = cv2.resize(frame, (300, 300))

    # Creating the blob
    blob = cv2.dnn.blobFromImage(resized_image, (1/127.5), (300, 300), 127.5, swapRB=True)

    # pass the blob through the network and obtain the predictions
    net.setInput(blob)
    predictions = net.forward()

    # loop over the predictions
    for i in np.arange(0, predictions.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = predictions[0, 0, i, 2]

        # Filter out predictions lesser than the minimum confidence level
        if confidence > 0.51:  # Save objects with confidence > 51%
            # extract the index of the class label from the 'predictions'
            idx = int(predictions[0, 0, i, 1])

            # compute the (x, y)-coordinates of the bounding box for the object
            box = predictions[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the coordinates are within the frame boundaries
            startX = max(startX, 0)
            startY = max(startY, 0)
            endX = min(endX, w)
            endY = min(endY, h)

            # Create a feature representation for the object (cropped image)
            detected_object = frame[startY:endY, startX:endX]

            # Check if detected_object is not empty
            if detected_object.size > 0:
                try:
                    # Convert detected object to RGB
                    rgb_image = cv2.cvtColor(detected_object, cv2.COLOR_BGR2RGB)

                    # Encode the detected object using face_recognition
                    encodings = face_recognition.face_encodings(rgb_image)
                    if len(encodings) > 0:
                        encoding = encodings[0]
                    else:
                        encoding = None

                    # Check if the object feature is already saved
                    if encoding is not None:
                        # Ensure saved_objects is a list of dictionaries
                        if isinstance(saved_objects, list):
                            saved_features = [obj['feature'] for obj in saved_objects if 'feature' in obj]  # Extract feature only
                            if not any(np.allclose(encoding, feature) for feature in saved_features):
                                # Save the detected object
                                save_path = os.path.join(save_dir, f"object_{len(saved_objects)+1}.png")
                                cv2.imwrite(save_path, detected_object)

                                # Append object feature and image path to saved_objects
                                saved_objects.append({'feature': encoding, 'path': save_path})
                                with open(features_file, "wb") as f:
                                    pickle.dump(saved_objects, f)

                                print(f"Saved detected object at {save_path}")

                except Exception as e:
                    print(f"[ERROR] Failed to process detected object: {e}")

            # Draw a rectangle around the object
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15

            # Put a label outside the rectangular detection
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

            # Announce the detection if enough time has passed
            current_time = time.time()
            if current_time - last_announcement_time >= 3:  # 3-second delay
                speak_detection(CLASSES[idx])
                last_announcement_time = current_time

    # show the output frame
    cv2.imshow("Frame", frame)

    # break the loop if 'q' key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] Elapsed Time: {:.2f}".format(fps.elapsed()))
print("[INFO] Approximate FPS: {:.2f}".format(fps.fps()))

# Cleanup
cv2.destroyAllWindows()
vs.stop()
