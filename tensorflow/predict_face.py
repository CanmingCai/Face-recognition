import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential

from tkinter import filedialog
import tkinter as tk
import cv2


batch_size = 4
img_height = 128
img_width = 128
epochs = 50

data_dir = 'Data'     # ruta de la carpeta con las imagenes

train_data_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_data_gen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

model = tf.keras.models.load_model("face_recognition_model")

def predict_face(img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    for class_name, class_index in train_generator.class_indices.items():
        if class_index == predicted_class:
            return class_name


def realtime_face_prediction():
    # Define a video capture object
    vid = cv2.VideoCapture(0)

    while True:
        # Capture the video frame by frame
        ret, frame = vid.read()

        # Convert the frame to RGB for prediction
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize the frame to match the input size of the model
        resized_frame = cv2.resize(rgb_frame, (img_width, img_height))

        # Perform face prediction on the resized frame
        predicted_face = predict_face(resized_frame)

        # Display the frame with predicted face name
        cv2.putText(frame, predicted_face, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)

        # Check if the 'q' button is pressed to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    vid.release()
    cv2.destroyAllWindows()

# Perform real-time face prediction using the webcam
realtime_face_prediction()
