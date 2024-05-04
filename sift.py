import cv2
import numpy as np
import os

# Global variables for storing ROI coordinates
roi_coords = []

def load_templates(template_folder):
    """Load template images from the specified folder."""
    template_files = [os.path.join(template_folder, f) for f in os.listdir(template_folder) if os.path.isfile(os.path.join(template_folder, f))]
    templates = [cv2.imread(template_file) for template_file in template_files]
    return templates

def custom_match_template(master_image, target_image, threshold=0.5):
    """Match the template image to the target image and return the result."""
    sift = cv2.SIFT_create()
    kp_1, desc_1 = sift.detectAndCompute(master_image, None)
    kp_2, desc_2 = sift.detectAndCompute(target_image, None)

    # Define FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc_1, desc_2, k=2)

    good_points = []
    match_coords = []

    for m, n in matches:
        if m.distance < threshold * n.distance:
            good_points.append(m)
            pt2 = kp_2[m.trainIdx].pt
            match_coords.append((int(pt2[0]), int(pt2[1])))

    num_good_matches = len(good_points)

    if num_good_matches > 0:
        top_left = match_coords[0]
        bottom_right = (top_left[0] + master_image.shape[1], top_left[1] + master_image.shape[0])
        target_image = cv2.rectangle(target_image, top_left, bottom_right, (0, 255, 0), 2)

    return target_image

def select_roi(event, x, y, flags, param):
    """Select the region of interest (ROI) in the first frame."""
    global roi_coords
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_coords = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        roi_coords.append((x, y))
        cv2.rectangle(param, roi_coords[0], roi_coords[1], (0, 255, 0), 2)
        cv2.imshow('Video', param)

def template_matching_video(video_path, template_folder, threshold=0.5):
    """Perform template matching on the video and display the result."""
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Load templates
    templates = load_templates(template_folder)

    global roi_coords
    if not roi_coords:  # If ROI coordinates are not already selected
        # Read the first frame
        ret, frame = cap.read()
        if not ret:
            print("Error reading video frame")
            return

        # Select the region of interest (ROI) in the first frame
        cv2.imshow('Video', frame)
        cv2.setMouseCallback('Video', select_roi, frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Extract the ROI coordinates
        if len(roi_coords) != 2:
            print("ROI selection error")
            return
        x1, y1 = roi_coords[0]
        x2, y2 = roi_coords[1]
    else:
        x1, y1 = roi_coords[0]
        x2, y2 = roi_coords[1]

    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop the frame to the ROI
        roi = frame[y1:y2, x1:x2].copy()

        # Match templates to the ROI
        for template in templates:
            roi = custom_match_template(template, roi, threshold)

        # Display the cropped ROI with bounding boxes
        cv2.imshow('Cropped ROI', roi)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Release the video capture object and close the output file
    cap.release()
    cv2.destroyAllWindows()

# Path to the folder containing videos
video_folder = r"D:\comb_icons\k_series"

# Template folder
template_folder = r"D:\comb_icons\temp_1"

# Perform template matching on all videos in the folder using the same ROI
for video_file in os.listdir(video_folder):
    video_path = os.path.join(video_folder, video_file)
    template_matching_video(video_path, template_folder)