import numpy as np
import cv2
import os
import pickle
from fbchat import Client
from fbchat.models import *
from password import pw


client = Client('lucadavidstefan21@yahoo.com', pw)


filename = 'video.mp4'
fps = 24.0
resolution = '480p'

STD_DIMENSIONS = {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}

VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    # 'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}


def set_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)


def get_dims(cap, resolution):
    width, height = STD_DIMENSIONS["480p"]
    if resolution in STD_DIMENSIONS:
        width, height = STD_DIMENSIONS[resolution]
    set_res(cap, width, height)
    return width, height


def video_type(filename):
    # used to split the file name so it gets the ext type
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
        return VIDEO_TYPE[ext]
    return VIDEO_TYPE['mp4']


face_cascade = cv2.CascadeClassifier(
    'D:\Programe\Python_Stuff\FaceRecognition\Haar Cascade\haarcascade_frontalface_alt2.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("D:\Programe\Python_Stuff\FaceRecognition\TrainedData.yml")

labels = {"person_name": 1}
with open("D:\Programe\Python_Stuff\FaceRecognition\labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}


cap = cv2.VideoCapture(0)
recorder = cv2.VideoWriter(filename, video_type(
    filename), fps, get_dims(cap, resolution))

warning = 0
while(True):

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=6)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y + h, x:x + w]

        id_, conf = recognizer.predict(roi_gray)

        if conf > 35 and conf < 70:
            print(conf)
            font = cv2.FONT_HERSHEY_COMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1,
                        color, stroke, cv2.LINE_AA)
        else:
            print(conf)
            font = cv2.FONT_HERSHEY_COMPLEX
            name = "not david"
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1,
                        color, stroke, cv2.LINE_AA)

        if warning == 0:
            while warning <= 3:
                warning = warning + 1
                message_id = client.send(Message(text="Warning!Someone has entered your room!"),
                                         thread_id="100004488721512", thread_type=ThreadType.USER)
        color = (255, 0, 0)  # BGR 0-255
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    recorder.write(frame)
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break


cap.release()
output.release()
cv2.destroyAllWindows()
Client.logout()
