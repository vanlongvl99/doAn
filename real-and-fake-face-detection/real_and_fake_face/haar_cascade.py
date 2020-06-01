import numpy as np
import cv2
import matplotlib.image as img
import matplotlib.pyplot as plt
import datetime
from os import listdir



forder = "training_fake"   # forder chứa ảnh cần xác định face
count = 0               # biến đếm ảnh mà mtcnn k phát hiện được face
cnt = 586               # chỉ là biến đặt tên ảnh cần lưu
arr = []
empty = np.array([])
print("start time")
print(datetime.datetime.now().time())
for filename in listdir(forder):
    cnt += 1
    path = forder + "/" + filename
    face_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
    image = img.imread(path)

    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    if faces == empty:
        count += 1
        arr.append(cnt)
        continue
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # cv2.imwrite(str(cnt) + '.jpg', image)
print("end time")
print(datetime.datetime.now().time())
print(arr)
print(count)
