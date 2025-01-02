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
        now = datetime.now()
        time = now.strftime('%I:%M:%S:%p')  # Time in 12-hour format
        date = now.strftime('%d-%B-%Y')  # Date in day-month-year format
        f.writelines(f'{name}, {time}, {date}\n')

# Define EAR threshold and other blink-related variables
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 3
blink_counter = 0

# Start the webcam feed with DirectShow to avoid MSMF warning
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

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

# Initialize a dictionary to count attendance markings for each person (case-insensitive)
attendance_counter = {name.upper(): 0 for name in classNames}  # Ensure the name is upper case
MAX_ATTENDANCE = 5  # Maximum number of times a person's attendance can be marked

# Function to handle multiple detected faces
def process_multiple_faces(encoded_faces, faces_in_frame, img, encoded_face_train, attendance_counter):
    for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
        # Compare the detected face with the trained faces
        matches = face_recognition.compare_faces(encoded_face_train, encode_face)
        faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
        
        matchIndex = np.argmin(faceDist)

        if faceDist[matchIndex] < 0.6:  # If the face distance is lower than the threshold
            name = classNames[matchIndex].upper()  # Convert the name to uppercase for consistency
        else:
            name = "UNKNOWN"  # Mark the face as unknown if distance is too high

        # Ensure the name is in the dictionary (case-insensitive) and attendance can be marked
        if name != "UNKNOWN" and attendance_counter.get(name, 0) < MAX_ATTENDANCE:
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Mark attendance if the count is below the maximum for a recognized person
            if attendance_counter.get(name, 0) < MAX_ATTENDANCE:
                markAttendance(name)
                attendance_counter[name] += 1  # Increment attendance counter for the recognized person
                print(f"Attendance marked for {name}.")
        else:
            # For unknown faces
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Mark unknown face with red color
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # Label as UNKNOWN

# Main loop for webcam feed and detection
while True:
    success, img = cap.read()  # Read a frame from the webcam
    
    if not success:
        print("Error reading webcam frame")  # Debug message if the webcam is not working
        break  # Exit the loop if the webcam frame is not read properly
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale (needed for dlib)
    faces = detector(gray)  # Detect faces using dlib

    # Loop through each detected face
    for face in faces:
        landmarks = predictor(gray, face)  # Detect landmarks for each face
        
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
                print("Blink detected!")  # Blink detected
            blink_counter = 0

    # If no blink detected, prompt for a valid blink
    if blink_counter == 0:
        cv2.putText(img, "Please blink your eyes to continue", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue  # Skip to the next frame if no blink

    # Once a blink is detected, proceed with face recognition
    if blink_counter >= CONSEC_FRAMES:
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # Resize the image to speed up processing
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)  # Convert the image to RGB (required by face_recognition)

        faces_in_frame = face_recognition.face_locations(imgS)
        encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)

        process_multiple_faces(encoded_faces, faces_in_frame, img, encoded_face_train, attendance_counter)

        # Display the final image with detected faces and attendance marked
        cv2.imshow('Webcam', img)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
