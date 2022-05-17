import tensorflow as tf
import cv2
import numpy as np


class Detector():
    def __init__(self, model_path, ):
        pass
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="/home/sara/Documents/Master-thesis/TFLite/models/yolo_models/768-fp16.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("INPUT DETAILS: ", input_details)
print("OUTPUT DETAILS: ", output_details)

# Evaluate
image = cv2.imread("/home/sara/Documents/Master-thesis/TFLite/data/yolo.jpg")
image_input = image.copy().astype(np.float32)/255.
image_input = np.reshape(image_input, [1,768,768,3])

interpreter.set_tensor(input_details[0]['index'], image_input)
interpreter.invoke()
img_size = 768

pred = interpreter.get_tensor(output_details[0]['index'])[0]
