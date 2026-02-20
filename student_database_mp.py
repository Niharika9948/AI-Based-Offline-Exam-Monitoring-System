import os
import cv2
import numpy as np

STUDENT_FOLDER = "students"

def load_student_database():
    encodings = []
    details = []

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    for file in os.listdir(STUDENT_FOLDER):
        if file.endswith((".jpg", ".png")):
            path = os.path.join(STUDENT_FOLDER, file)
            image = cv2.imread(path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) == 0:
                continue
            x, y, w, h = faces[0]
            face_img = cv2.resize(gray[y:y+h, x:x+w], (100,100)).flatten()
            encodings.append(face_img)
            hall, name = file.split(".")[0].split("_")
            details.append({"hallticket": hall, "name": name})

    return encodings, details
