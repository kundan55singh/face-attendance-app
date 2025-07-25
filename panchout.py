import cv2
import pickle
import numpy as np
import csv
import os
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier

# Load Haar cascade for face detection
facedetect = cv2.CascadeClassifier("D:\\machine leaning\\opencv\\project\\new project\\data\\haarcascade_frontalface_default.xml")

# Load face encodings and labels
with open("data/faces_data.pkl", "rb") as f:
    faces_data = pickle.load(f)
with open("data/names.pkl", "rb") as f:
    labels = pickle.load(f)

# Ensure equal length
min_len = min(len(faces_data), len(labels))
faces_data = faces_data[:min_len]
labels = labels[:min_len]

# Train classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(faces_data, labels)

# Ensure attendance folder exists
if not os.path.exists("attendance"):
    os.makedirs("attendance")

# Start webcam
cap = cv2.VideoCapture(0)

# Column names for CSV
COL_NAMES = ['Name', 'Punch-Out Time']

print("Press 'p' to punch out, 'q' to quit.")

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_crop = frame[y:y+h, x:x+w]
        resized_face = cv2.resize(face_crop, (50, 50)).flatten().reshape(1, -1)
        prediction = knn.predict(resized_face)[0]

        # Draw face and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, prediction, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        key = cv2.waitKey(1) & 0xFF

        # Punch-out on pressing 'p'
        if key == ord('p'):
            now = datetime.now()
            date_str = now.strftime("%d-%m-%Y")
            time_str = now.strftime("%H:%M:%S")
            filename = f"attendance/punch_out_{date_str}.csv"
            already_punched_out = False

            # Check for duplicates
            if os.path.exists(filename):
                with open(filename, "r") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if row and row[0] == prediction:
                            already_punched_out = True
                            break

            if not already_punched_out:
                with open(filename, "a", newline="") as f:
                    writer = csv.writer(f)
                    if os.stat(filename).st_size == 0:
                        writer.writerow(COL_NAMES)
                    writer.writerow([prediction, time_str])
                print(f"{prediction} punched out at {time_str}")
            else:
                print(f"{prediction} already punched out today!")

    cv2.imshow("🔒 Punch-Out Face Verification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
