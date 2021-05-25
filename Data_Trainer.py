import os
import numpy as np
from PIL import Image
import cv2
import pickle

face_cascade = cv2.CascadeClassifier(
    'D:\Programe\Python_Stuff\FaceRecognition\Haar Cascade\haarcascade_frontalface_alt2.xml')
directory = os.path.dirname(os.path.abspath(__file__))
Persons_Data_Path = os.path.join(directory, "Persons_Data")
recognizer = cv2.face.LBPHFaceRecognizer_create()


label_id = {}
current_id = 0
labels_train = []
image_train = []

for root, dirs, files in os.walk(Persons_Data_Path):
    for file in files:
        if file.endswith("png") or file.endswith("jpeg"):
            img_path = os.path.join(root, file)  # image path
            label = os.path.basename(root).lower()  # the label path
            if not label in label_id:
                label_id[label] = current_id
                current_id += 1
            id_ = label_id[label]
            pil_image = Image.open(img_path).convert(
                "L")  # converting the image to grayscale
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            # 3D np array for each picture
            img_array = np.array(pil_image, "uint8")
            faces = face_cascade.detectMultiScale(
                img_array, scaleFactor=1.1, minNeighbors=2)
            for (x, y, w, h) in faces:
                roi = img_array[y:y + h, x:x + w]
                print(img_array[y:y + h, x:x + w])
                image_train.append(roi)
                labels_train.append(id_)

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_id, f)

recognizer.train(image_train, np.array(labels_train))
recognizer.save("TrainedData.yml")
