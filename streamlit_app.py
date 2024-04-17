import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import time
from collections import Counter
# Load the custom model
import os
os.environ['TORCH_HOME'] = '/tmp/torch'
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best_weight/best_4.pt', force_reload=True)



def detect_objects(image):
    """
    Perform object detection on an image and return detailed detections.
    """
    # Convert PIL image to CV2 format
    image = np.array(image)
    # Perform detection
    results = model(image)
    # Extract detection details
    detections = results.pandas().xyxy[0]  # xyxy format: xmin, ymin, xmax, ymax
    return detections

def draw_boxes(detections, image):
    """
    Draw bounding boxes on the image for each detection, with special emphasis on specific objects.
    """
    # Specify objects of interest
    special_objects = ['pen', 'bottle', 'glasses']
    # Loop through detections and draw boxes
    for _, row in detections.iterrows():
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = row['name']
        confidence = round(row['confidence'], 2)
        color = (0, 255, 0) if label in special_objects else (255, 0, 0)  # Green for special objects, red for others
        # Draw the rectangle
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        # Prepare the label text
        text = f"{label} {confidence}"
        # Calculate text width & height to draw the text background
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Draw the text background
        image[ymin - text_height - 10:ymin, xmin:xmin + text_width, :] = color
        # Put the text on the image
        cv2.putText(image, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    st.title("Real-time Object Detection with Custom YOLOv5")

    # Initialize session states for webcam and stop flag
    if 'capture' not in st.session_state:
        st.session_state['capture'] = False
    if 'stop_pressed' not in st.session_state:
        st.session_state['stop_pressed'] = False

    # Sidebar for displaying static class names and counts
    st.sidebar.markdown("**Detected Object Counts**")
    class_names = ['person', 'bottle', 'glasses', 'pen']
    class_counters = {class_name: 0 for class_name in class_names}
    class_counters_session = {class_name: 0 for class_name in class_names}
    class_count_placeholders = {class_name: st.sidebar.text(f"{class_name}: 0") for class_name in class_names}

    # Toggle button for the webcam
    start_button, stop_button = st.columns(2)
    with start_button:
        if st.button('Start Webcam'):
            st.session_state['capture'] = True
            st.session_state['stop_pressed'] = False

    with stop_button:
        if st.button('Stop Webcam'):
            st.session_state['stop_pressed'] = True

    frame_placeholder = st.empty()
    info_placeholder = st.empty()

    if st.session_state['capture']:
        cap = cv2.VideoCapture(0)
        frame_count = 0
        start_time = time.time()

        while not st.session_state['stop_pressed']:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image.")
                break

            frame_count += 1
            elapsed_time = time.time() - start_time
            
            # Perform object detection
            detections = detect_objects(frame)

            # Draw bounding boxes on the frame
            draw_boxes(detections, frame)

            # Convert frame to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the frame
            frame_placeholder.image(frame)

            # Update the frame count and elapsed time display
            info_placeholder.markdown(f"**Frame Count**: {frame_count}, **Elapsed Time**: {elapsed_time:.2f} seconds")

            # Count the occurrences of each class
            if not detections.empty:
                detected_classes = detections['class'].value_counts().to_dict()
                print('detected_classes',detected_classes)
                for class_name in class_names:
                    if 0 in detected_classes:
                        class_counters_session['person'] = detected_classes[0]
                        class_count_placeholders[class_name].markdown(f"{class_name}: {class_counters[class_name]}")

                    if 1 in detected_classes:
                        class_counters_session['glasses'] = detected_classes[1]
                        class_count_placeholders[class_name].markdown(f"{class_name}: {class_counters[class_name]}")

                    if 2 in detected_classes:
                        class_counters_session['bottle'] = detected_classes[2]
                        class_count_placeholders[class_name].markdown(f"{class_name}: {class_counters[class_name]}")

                    if 3 in detected_classes:
                        class_counters_session['pen'] = detected_classes[3]
                        class_count_placeholders[class_name].markdown(f"{class_name}: {class_counters[class_name]}")

            for classes in class_names:
                class_count_placeholders[class_name].markdown(f"{classes}: {0}")
                class_counters[classes]+=class_counters_session[classes]
                class_counters_session[classes] = 0

            #class_count_placeholders = {class_name: st.sidebar.text(f"{class_name}: 0") for class_name in class_names}

        cap.release()
        st.session_state['capture'] = False

if __name__ == '__main__':
    main()
