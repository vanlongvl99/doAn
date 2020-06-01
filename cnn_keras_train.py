from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from os import listdir
import numpy as np
from mtcnn.mtcnn import MTCNN



label_names = {}
forder = "data/raw"
index_label = 0
for forder_name in listdir(forder):
    label_names[forder_name] = index_label 
    index_label += 1
print(label_names)       
# {'van_long': 0, 'tran_thanh': 1, 'Bich_lan': 2, 'hoai_linh': 3}

y_labels = []
x_data = []
for forder_name in listdir(forder):
    for filename in listdir(forder + "/" + forder_name):
        y_labels.append(label_names[forder_name])
        path = forder + "/" + forder_name + "/" + filename
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32)
        image = (image/255)           #normalize
        image = tf.image.resize(image, (160, 160))
        x_data.append(image)

# print(y_labels)
x_data = tf.convert_to_tensor(x_data)
# print(x_data.shape)
# print(type(x_data[0]))
y_labels = tf.convert_to_tensor(y_labels)
# x_data = np.array(x_data)
# y_labels = np.array(y_labels)

print(type(x_data))


X_train, X_test, y_train, y_test = train_test_split(x_data.numpy(), y_labels.numpy(), test_size=0.5)


# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# print(type(X_train))
# print(X_train[:3])
X_train = tf.convert_to_tensor(X_train)
X_test =    tf.convert_to_tensor(X_test)
y_train = tf.convert_to_tensor(y_train)
y_test = tf.convert_to_tensor(y_test)

# print(y_test[:20])


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(20, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    # tf.keras.layers.Conv2D(20, 15, activation='relu'),
    # tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    # TODO: fill suitable activations
    tf.keras.layers.Dense(units=40, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(units=40, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(units=5, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
            optimizer = tf.keras.optimizers.Adam(),
            metrics = ['accuracy'])

model.fit(X_train, y_train, epochs=15, batch_size = 3)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
# model.save_weights("model.h5")
model.save("modelacbad.h5")

# model.load_weights("model.h5")

path = "1253.jpg"

# image = plt.imread(path)

detector = MTCNN()
image = plt.imread(path)
face = detector.detect_faces(image)
x_test_abc = []

fordef_test = "abcdef"
for filename in listdir(fordef_test):
    path = fordef_test + "/" + filename
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = (image/255)           #normalize
    image = tf.image.resize(image, (160, 160))
    x_test_abc.append(image)

im = [x_test_abc[0]]
im = tf.convert_to_tensor(im)
# x_test_abc = tf.convert_to_tensor(x_test_abc)

prediction = model.predict(im)

print(prediction)