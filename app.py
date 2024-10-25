import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import function.helper as helper
import function.utils_rotate as utils_rotate

# Load models
helmet_model = YOLO("best.pt")
yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr.pt', force_reload=True, source='local')
yolo_license_plate.conf = 0.60

# Streamlit UI setup
st.title("Helmet Detection and License Plate Extraction")

# Upload video
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

# Helper function to convert BGR to RGB
def convert_bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Function to check proximity of two bounding boxes
def is_close(bbox1, bbox2, threshold=50):
    """Check if the centers of two bounding boxes are within a certain distance."""
    center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
    center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
    distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    return distance < threshold

if uploaded_file is not None:
    # Handle uploaded video file
    temp_file = "temp_video.mp4"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Open the saved temporary video file using OpenCV
    cap = cv2.VideoCapture(temp_file)
    if not cap.isOpened():
        st.error("Error: Could not open video.")
        st.stop()

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames_to_process_per_second = 10

    detected_plates = set()  # Set to keep track of unique detected plates

    while cap.isOpened():
        frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        if not ret:
            break

        # Process frames at uniform intervals per second
        if frame_index % (fps // frames_to_process_per_second) == 0:
            results = helmet_model(frame)

            for result in results:
                labels = [result.names[int(box.cls[0])] for box in result.boxes]
                if "without helmet" in labels:
                    for helmet_box in result.boxes:
                        helmet_bbox = helmet_box.xyxy[0].cpu().numpy()  # Get helmet bbox coordinates

                        # Check license plate in this frame
                        plates = yolo_LP_detect(frame, size=640)
                        list_plates = plates.pandas().xyxy[0].values.tolist()

                        for plate in list_plates:
                            lp_bbox = [plate[0], plate[1], plate[2], plate[3]]
                            if is_close(helmet_bbox, lp_bbox, threshold=50):
                                x, y, w, h = int(lp_bbox[0]), int(lp_bbox[1]), int(lp_bbox[2] - lp_bbox[0]), int(lp_bbox[3] - lp_bbox[1])
                                crop_img = frame[y:y + h, x:x + w]

                                # Try to read the license plate in the cropped image
                                for cc in range(2):
                                    for ct in range(2):
                                        rotated_crop = utils_rotate.deskew(crop_img, cc, ct)
                                        lp = helper.read_plate(yolo_license_plate, rotated_crop)
                                        if lp != "unknown":
                                            detected_plates.add(lp)
                                            rgb_frame = convert_bgr_to_rgb(frame)  # Convert frame to RGB before display
                                            st.image(rgb_frame, caption=f"License Plate Detected: {lp}", use_column_width=True)
                                            st.write(f"License Plate: {lp}")
                                            break
                                    if lp != "unknown":
                                        break

    cap.release()
    st.text("Processing complete.")
    st.write("Detected License Plates:")
    for plate in detected_plates:
        st.write(plate)
