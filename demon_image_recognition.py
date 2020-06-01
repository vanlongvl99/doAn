import tensorflow as tf
import keras
import os
import matplotlib.pyplot as plt
import numpy as np
from mtcnn.mtcnn import MTCNN
import cv2
from tensorflow.keras.models import load_model
from keras.models import model_from_json
from os import listdir

mode = load_model("/home/vanlong/vanlong/ky6/doAn/modelacbad.h5")
mode.load_weights("model.h5")


path = "1252.jpg"

# image = plt.imread(path)
# label_names = {0 :'van_long',1 : "tran_thanh", 2: "Bich_lan", 3: "hoai_linh"}
label_names = {}
i = 0
for forder_name in listdir("data/progress"):
    label_names[i] = forder_name
    i += 1
print(label_names)

detector = MTCNN()
image = plt.imread(path)
face = detector.detect_faces(image)
for person in face:
    bounding_box = person['box']
    keypoints = person['keypoints']
    im_crop = image[bounding_box[1]: bounding_box[1] + bounding_box[3], bounding_box[0]: bounding_box[0]+bounding_box[2] ]
    im_crop = tf.convert_to_tensor(im_crop)
    im_crop = tf.cast(im_crop, tf.float32)
    im_crop = (im_crop/255)           #normalize
    im_crop = tf.image.resize(im_crop, (160, 160))
    im_crop = [im_crop]
    im_crop = tf.convert_to_tensor(im_crop)
    prediction = mode.predict(im_crop)
    max_index = int(np.argmax(prediction))
    cv2.rectangle(image,(bounding_box[0], bounding_box[1]),(bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),(0,155,255),2)
    cv2.putText(image, label_names[max_index], (bounding_box[0]+20, bounding_box[1]-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(path + "abc.jpg",image)
