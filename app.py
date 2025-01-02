import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
from imutils import face_utils
import dlib

# Path to the folder where student images are stored
path = r'C:\Users\bhavana.r\Desktop\2024December Face recognition\Fr_photos'

# List to hold images and class names (student names)
images = []
classNames = []

# Get all the image filenames in the specified path
mylist = os.listdir(path)

# Loop through the images and load them into the list
for cl in mylist:
    img_path = f'{path}/{cl}'
    
    # Read the image from the file path
    curImg = cv2.imread(img_path)
    
    if curImg is None:
        print(f"Error loading image: {img_path}")  # Debug message if image is not loaded
        continue  # Skip to the next image if the current one failed to load
    
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])  # Get the name without extension

# Function to find face encodings
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert the image to RGB
        encoded_face = face_recognition.face_encodings(img)
        if encoded_face:  # Check if any faces are detected in the image
            encodeList.append(encoded_face[0])  # Get the first face encoding
        else:
            print("No face found in image.")  # Handle case where no face is detected
    return encodeList

# Get the face encodings for all images
encoded_face_train = findEncodings(images)

# Function to mark attendance in a CSV file
def markAttendance(name):
    with open('Attendance.csv', 'a+') as f:
        myDataList = f.readlines()
        nameList = [entry.split(',')[0] for entry in myDataList]
        if name not in nameList:
            now = datetime.now()
            time = now.strftime('%I:%M:%S:%p')  # Time in 12-hour format
            date = now.strftime('%d-%B-%Y')  # Date in day-month-year format
            f.writelines(f'{name}, {time}, {date}\n')

# Define a threshold for face distance (this can be adjusted based on your needs)
THRESHOLD = 0.6  # Set a threshold for the face distance

# Start the webcam feed
cap = cv2.VideoCapture(0)

# Load dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Make sure to download this file

# Function to compute Eye Aspect Ratio (EAR) to detect blinks
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])  # Vertical distance between two eye landmarks
    B = np.linalg.norm(eye[2] - eye[4])  # Vertical distance between two eye landmarks
    C = np.linalg.norm(eye[0] - eye[3])  # Horizontal distance between two eye landmarks
    ear = (A + B) / (2.0 * C)
    return ear

# Define EAR threshold for blink detection
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 3  # Number of consecutive frames for blink detection

# Initialize blink counter
blink_counter = 0

# Define a variable to track if the person is live or not
is_live = False

while True:
    success, img = cap.read()  # Read a frame from the webcam
    
    if not success:
        print("Error reading webcam frame")  # Debug message if the webcam is not working
        break  # Exit the loop if the webcam frame is not read properly
    
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # Resize the image to speed up processing
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)  # Convert the image to RGB (required by face_recognition)

    # Detect faces in the current frame
    faces_in_frame = face_recognition.face_locations(imgS)
    encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)

    for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
        # Compare the detected face with the trained faces
        matches = face_recognition.compare_faces(encoded_face_train, encode_face)
        faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
        
        # Get the index of the face with the minimum distance
        matchIndex = np.argmin(faceDist)

        if faceDist[matchIndex] < THRESHOLD:  # If the face distance is lower than the threshold
            name = classNames[matchIndex].upper()  # Get the name of the recognized person
            is_live = True  # The person is live if they are recognized
        else:
            name = "Unknown"  # If no match is found, label as Unknown
            is_live = False  # If unknown, assume not live

        y1, x2, y2, x1 = faceloc  # Get face coordinates

        # Since the image was resized, we need to scale the coordinates back
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

        # Draw a rectangle around the face
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw a filled rectangle for the name label
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        # Put the name text on top of the rectangle
        cv2.putText(img, name, (x1 + 6, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        # Apply liveness detection using blink
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)  # Detect landmarks
            
            # Get coordinates of the left and right eye
            left_eye = face_utils.shape_to_np(landmarks)[42:48]
            right_eye = face_utils.shape_to_np(landmarks)[36:42]
            
            # Calculate EAR for both eyes
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # If EAR is below the threshold, consider it as a blink
            if ear < EAR_THRESHOLD:
                blink_counter += 1
            else:
                if blink_counter >= CONSEC_FRAMES:
                    print("Blink detected!")
                blink_counter = 0

        # If no blink is detected or face is static, treat it as non-live
        if blink_counter < CONSEC_FRAMES:
            is_live = False  # Mark as non-live if no blink detected

        # Mark attendance for the recognized person only if live
        if name != "Unknown" and is_live:
            markAttendance(name)

    # Display the resulting image
    cv2.imshow('Webcam', img)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
