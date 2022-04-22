from turtle import pos
import tensorflow as tf
import cv2
import numpy as np
import torch
import torchvision

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="/home/sara/Documents/Master-thesis/TFLite/models/class_models/test16.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("INPUT DETAILS: ", input_details)
print("OUTPUT DETAILS: ", output_details)

# Evaluate
image = cv2.imread("/home/sara/Documents/Master-thesis/TFLite/data/boat4.jpg")
image_input = image.copy().astype(np.float32)
image_input = np.reshape(image_input, [1,768,768,3])

interpreter.set_tensor(input_details[0]['index'], image_input)
interpreter.invoke()

pred = interpreter.get_tensor(output_details[0]['index'])[0]
print("OUTPUT_DATA: ", pred)
if pred[0] > pred[1]:
    print(f"Predicted boat: {pred[0]:.3f}")
else:
    print(f"Predicted no boat: {pred[1]:.3f}")