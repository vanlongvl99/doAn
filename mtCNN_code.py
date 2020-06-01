import cv2 
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as img
from os import listdir
import datetime
import os

detector = MTCNN()
forder = "data_detection/fake_faces"   # forder chứa ảnh cần xác định face
wrong_detects = 0               # biến đếm ảnh mà mtcnn k phát hiện được face
total_image = 0               # chỉ là biến đặt tên ảnh cần lưu
wrong = []
print("start time")
print(datetime.datetime.now().time())  # xác định thời gian bắt đầu của model
for filename in listdir(forder):
  total_image += 1
  path = forder + "/" + filename
  image = plt.imread(path)
  result = detector.detect_faces(image)
  if len(result) == 0:
    wrong_detects += 1
    wrong.append(filename)
    cv2.imwrite(os.path.join("right" , filename), image)
    continue
  else:
    for person in result:
      bounding_box = person['box']
      keypoints = person['keypoints']
      cv2.rectangle(image,(bounding_box[0], bounding_box[1]),(bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),(0,155,255),2)
      cv2.imwrite(os.path.join("wrong" , filename), image)

print("end time")
print(datetime.datetime.now().time())
print("wrong", wrong)
print("wrong detects:",wrong_detects)
print("total images:", total_image)
print("accurance:",1-wrong_detects/total_image)