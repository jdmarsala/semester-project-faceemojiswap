import cv2
import mediapipe as mp

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection

# Load the image from the specified path
image_path = r'C:\Users\dillo\github-classroom\SSOE-ECE1390\semester-project-faceemojiswap\Data\TIger woods.jpeg'
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print("Error loading image. Please check the image path and file.")
    exit()

# Resize the image to shrink it
scale_percent = 50  # Adjust this value to shrink the image (e.g., 50 means 50% of the original size)
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
# Resize the image
image_resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# Convert the resized image to RGB as Mediapipe requires
image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

# Initialize the Face Detection model
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    # Perform face detection on the resized image
    results = face_detection.process(image_rgb)

    # Draw face detections on the resized image
    if results.detections:
        print(f"Detected {len(results.detections)} face(s).")

        for detection in results.detections:
            # Get bounding box coordinates in relative units
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = image_resized.shape
            x_min = int(bbox.xmin * w)
            y_min = int(bbox.ymin * h)
            box_width = int(bbox.width * w)
            box_height = int(bbox.height * h)

            # Ensure coordinates are within image bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_min + box_width)
            y_max = min(h, y_min + box_height)

            # Print coordinates for debugging
            print(f"x_min: {x_min}, y_min: {y_min}, x_max: {x_max}, y_max: {y_max}")

            # Draw rectangle around the face (blue color)
            cv2.rectangle(image_resized, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    else:
        print("No faces detected.")

# Display the resized image with the face detection rectangle
cv2.imshow('Mediapipe Face Detection', image_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
