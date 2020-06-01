import numpy as np
import cv2
import matplotlib.image as img
import matplotlib.pyplot as plt
import datetime
from os import listdir
import os


forder = "data_detection/Long_Lan_image"   # forder chứa ảnh cần xác định face
wrong_detects = 0               # biến đếm ảnh mà mtcnn k phát hiện được face
total_image = 0
arr = []
print("start time")
print(datetime.datetime.now().time())
for filename in listdir(forder):
    total_image += 1
    path = forder + "/" + filename
    face_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
    image = plt.imread(path)

    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    if len(faces) == 0:
        wrong_detects += 1
        arr.append(filename)
        cv2.imwrite(os.path.join("wrong" , filename), image)

        continue
    elif len(faces) == 1:
        wrong_detects += 1

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255,0 , 0), 2)
            # cv2.imwrite(filename, image)
            cv2.imwrite(os.path.join("miss_1" , filename), image)

    else:
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255,0 , 0), 2)
            # cv2.imwrite(filename, image)
            cv2.imwrite(os.path.join("right" , filename), image)
print("end time")
print(datetime.datetime.now().time())
print(arr)
print("wrong detects:",wrong_detects)
print("total images:", total_image)
print("accurance:",1-wrong_detects/total_image)
