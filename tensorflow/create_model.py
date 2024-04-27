import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential

from tkinter import filedialog
import tkinter as tk

data_dir = 'Data'     # ruta de la carpeta con las imagenes

batch_size = 4
img_height = 128
img_width = 128
epochs = 50


train_data_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_data_gen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')


model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(img_height, img_width ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(
    train_generator,
    epochs=epochs
)

# 保存模型
model.save("face_recognition_model")
model = tf.keras.models.load_model("face_recognition_model")


def predict_face(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    for class_name, class_index in train_generator.class_indices.items():
        if class_index == predicted_class:
            return class_name


root = tk.Tk()
root.withdraw()  # Hide the root window
test_image_path = filedialog.askopenfilename()
predicted_face = predict_face(test_image_path)
print("Predicted face:", predicted_face)

#打开图片 并且显示名字
import cv2
import matplotlib.pyplot as plt
img = cv2.imread(test_image_path)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(predicted_face)
plt.show()

