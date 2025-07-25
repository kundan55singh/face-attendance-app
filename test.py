# from sklearn.neighbors import KNeighborsClassifier



# import cv2
# import pickle
# import numpy as np
# import os
# video = cv2.VideoCapture(0)
# facedetect = cv2.CascadeClassifier('D:\\machine leaning\\opencv\\project\\new project\\data\\haarcascade_frontalface_default.xml')

# with open ('data/names.pkl', 'rb') as f:
#        LEBLES= pickle.load(f)
# with open ('data/faces_data.pkl', 'rb') as f:
#        FACES= pickle.load(f)

# knn =KNeighborsClassifier(n_neighbors=5)
# knn.fit(FACES, LEBLES)


# while True:
#     ret, frame = video.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = facedetect.detectMultiScale(gray, 1.3, 5)
#     for (x,y,w,h) in faces:
#         crop_img= frame[y:y+h, x:x+w, :]
#         resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1,-1)
#         output = knn.predict(resized_img)
#         cv2.putText(frame, str(output[0]), (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
    
#         cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255),1)
#     cv2.imshow("Frame", frame)
#     k = cv2.waitKey(1)
#     if k == ord('q'):
#         break

# video.release()
# cv2.destroyAllWindows
import cv2
import pickle
import numpy as np
import csv
import os
import time
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from win32com.client import Dispatch

def speak(text):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.speak(text)

# Open webcam
video = cv2.VideoCapture(0)

# Load face detection model
face_detect = cv2.CascadeClassifier(
    'D:/machine leaning/opencv/project/new project/data/haarcascade_frontalface_default.xml'
)

# Load face data
with open('data/faces_data.pkl', 'rb') as f:
    faces_data = pickle.load(f)
with open('data/names.pkl', 'rb') as f:
    labels = pickle.load(f)

# Truncate if mismatched
min_len = min(len(faces_data), len(labels))
faces_data = faces_data[:min_len]
labels = labels[:min_len]

# Train model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(faces_data, labels)

# Load and resize background
bg_image = cv2.imread("bg_image.png")  # Optional background
bg_width, bg_height = 1366, 768
bg_image = cv2.resize(bg_image, (bg_width, bg_height))

# Attendance folder
os.makedirs("attendance", exist_ok=True)

COL_NAMES = ['Name', 'Time']
recognized_person = None  # To hold current prediction

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame = cv2.resize(frame, (bg_width, bg_height))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, 1.3, 5)

    recognized_person = None

    for (x, y, w, h) in faces:
        crop = frame[y:y+h, x:x+w]
        resized_face = cv2.resize(crop, (50, 50)).flatten().reshape(1, -1)
        prediction = knn.predict(resized_face)[0]
        recognized_person = prediction

        # Draw box and name
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (0, 0, 255), -1)
        cv2.putText(frame, prediction, (x+5, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Combine background
    blended = cv2.addWeighted(bg_image, 0.2, frame, 0.8, 0)
    cv2.imshow("Face Recognition - Punch In", blended)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('o') and recognized_person:
        now = datetime.now()
        date_str = now.strftime("%d-%m-%Y")
        time_str = now.strftime("%H:%M:%S")
        file_path = f"attendance/attendance_{date_str}.csv"

        # Check if person already punched in
        already_marked = False
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and row[0] == recognized_person:
                        already_marked = True
                        break

        if not already_marked:
            with open(file_path, "a", newline="") as f:
                writer = csv.writer(f)
                if os.stat(file_path).st_size == 0:
                    writer.writerow(COL_NAMES)
                writer.writerow([recognized_person, time_str])
            speak(f"{recognized_person}, your attendance has been recorded.")
            print(f"{recognized_person} punched in at {time_str}")
        else:
            speak(f"{recognized_person}, you have already punched in today.")
            print(f"{recognized_person} already punched in today.")

    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
