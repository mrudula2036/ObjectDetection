import streamlit as st
import cv2
import numpy as np
import os
import pickle
import face_recognition
import pyttsx3
from imutils.video import VideoStream
from imutils.video import FPS
import time
import imutils

# Initialize variables and model
save_dir = os.getcwd()
features_file = os.path.join(save_dir, "saved_objects.pkl")

# Load saved objects if available
if os.path.exists(features_file):
    with open(features_file, "rb") as f:
        saved_objects = pickle.load(f)
else:
    saved_objects = []

CLASSES = ["aeroplane", "background", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Initialize the text-to-speech engine
engine = pyttsx3.init()

def speak_detection(label):
    engine.say(label)
    engine.runAndWait()

def load_model(prototxt, model):
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    return net

def detect_objects(net, frame, confidence_threshold=0.51):
    (h, w) = frame.shape[:2]
    resized_image = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(resized_image, (1/127.5), (300, 300), 127.5, swapRB=True)
    net.setInput(blob)
    predictions = net.forward()

    detections = []
    for i in np.arange(0, predictions.shape[2]):
        confidence = predictions[0, 0, i, 2]
        if confidence > confidence_threshold:
            idx = int(predictions[0, 0, i, 1])
            box = predictions[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            startX = max(startX, 0)
            startY = max(startY, 0)
            endX = min(endX, w)
            endY = min(endY, h)

            detected_object = frame[startY:endY, startX:endX]

            if detected_object.size > 0:
                try:
                    rgb_image = cv2.cvtColor(detected_object, cv2.COLOR_BGR2RGB)
                    encodings = face_recognition.face_encodings(rgb_image)
                    if len(encodings) > 0:
                        encoding = encodings[0]

                        # Check if the encoding is already saved
                        saved_features = [obj['feature'] for obj in saved_objects if 'feature' in obj]
                        # Use np.any() to avoid ambiguity in boolean expressions
                        if not any(np.allclose(encoding, feature) for feature in saved_features):
                            # Append the encoding to saved_objects without saving an image
                            saved_objects.append({'feature': encoding})
                            with open(features_file, "wb") as f:
                                pickle.dump(saved_objects, f)

                except Exception as e:
                    # Log error without stopping the loop
                    st.error(f"Error processing detected object: {e}")

            detections.append((startX, startY, endX, endY, idx, confidence))

    return detections

def draw_detections(frame, detections):
    for (startX, startY, endX, endY, idx, confidence) in detections:
        cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
        cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

# Streamlit interface
st.title("Real-Time Object Detection Controller")

prototxt_path = st.text_input("Prototxt Path", "MobileNetSSD_deploy.prototxt.txt")
model_path = st.text_input("Model Path", "MobileNetSSD_deploy.caffemodel")

start_detection = st.button("Start Detection")

# Use a session state to maintain detection status
if "detection_running" not in st.session_state:
    st.session_state.detection_running = False

if start_detection:
    st.session_state.detection_running = True

stop_detection = st.button("Stop Detection")
if stop_detection:
    st.session_state.detection_running = False

if st.session_state.detection_running:
    net = load_model(prototxt_path, model_path)
    
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    fps = FPS().start()

    last_announcement_time = time.time()

    # Start a separate loop for video capture
    while st.session_state.detection_running:
        try:
            frame = vs.read()
            frame = imutils.resize(frame, width=400)

            detections = detect_objects(net, frame)
            draw_detections(frame, detections)

            current_time = time.time()
            for (_, _, _, _, idx, _) in detections:
                if current_time - last_announcement_time >= 3:
                    speak_detection(CLASSES[idx])
                    last_announcement_time = current_time

            # Display the frame using OpenCV
            cv2.imshow("Frame", frame)

            # Check for 'q' key press to stop manually
            if cv2.waitKey(1) & 0xFF == ord('q'):
                st.session_state.detection_running = False
                break

            fps.update()

        except Exception as e:
            # Catch any exceptions to ensure the loop continues
            st.error(f"Error during detection loop: {e}")

    fps.stop()
    vs.stop()
    cv2.destroyAllWindows()

    st.write(f"Elapsed Time: {fps.elapsed():.2f}")
    st.write(f"Approximate FPS: {fps.fps():.2f}")
