import pandas as pd
import cv2 

train_csv = pd.read_csv("data/famous/train.csv")
path_x_train =train_csv.values[:,0]
y_label = train_csv.values[:, 1]
print(len(y_label))
z = set(y_label)
print(len(z))
x_train = []
for i in range(len(z)):
    x_train.append([])

for i in range(len(path_x_train)):
    image = cv2.imread("/data/famous/train/" + path_x_train[i])
    x_train[y_label[i]].append(image)

for i in range(20):
    print(len(x_train[i]))