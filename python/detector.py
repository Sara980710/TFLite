import tensorflow as tf
import cv2
import argparse
import numpy as np
import time

class Detector():

    def __init__(self, model_path) -> None:
        
        self.interpreter = tf.lite.Interpreter(model_path)
        self.interpreter.allocate_tensors()

        details = self.interpreter.get_input_details()
        self.shape = details[0]["shape"]
        self.input_index = details[0]['index']

        self.current_tile = 0

    def load_image(self, image_path):
        self.image = cv2.imread(image_path)

    def tile_image(self):
        self.tiles = []
        for x in range(0, self.image.shape[0], self.shape[1]):
            for y in range(0, self.image.shape[1], self.shape[2]):

                if x + self.shape[2] >= self.image.shape[0] and y + self.shape[1] >= self.image.shape[1]:

                    self.tiles.append((self.image.shape[0] - self.shape[1], self.image.shape[1] - self.shape[2], self.shape[1], self.shape[2]))
                else:
                    if x + self.shape[1] >= self.image.shape[0]:
                        self.tiles.append((self.image.shape[0] - self.shape[1], y, self.shape[1], self.shape[2]))
                    elif y + self.shape[2] >= self.image.shape[1]:
                        self.tiles.append((x, self.image.shape[1] - self.shape[2], self.shape[1], self.shape[2]))
                    else:
                        self.tiles.append((x, y, self.shape[1], self.shape[2]))
    
        print(f"Nr tiles: {len(self.tiles)}")

    def load_input(self, verbose=False):
        input_values = np.zeros([self.shape[0],self.shape[1],self.shape[2],self.shape[3]], dtype=np.float32)
        for i in range(self.shape[0]):
            if self.current_tile >= len(self.tiles) or self.current_tile == -1:
                self.current_tile = -1
                break
            
            if verbose:
                print(f"Processing tile: {self.current_tile+1}/{len(self.tiles)}")
            tile = self.tiles[self.current_tile]
            tile = self.image[tile[0]:tile[0]+tile[2], tile[1]:tile[1]+tile[3]].astype(np.float32)/255.
            input_values[i,...] = tile
            self.current_tile += 1
            

        self.interpreter.set_tensor(self.input_index, input_values)

    def detect(self):
        self.interpreter.invoke()
    
    def get_output(self):
        return self.interpreter.get_tensor(self.interpreter.get_output_details()[0]['index'])[0]


if __name__=="__main__":
    now = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default= '/home/sara/Documents/Master-thesis/TFLite/models/yolo_models/768_2-fp16.tflite', help='initial weights path')
    parser.add_argument('--image', type=str, default= '/home/sara/Documents/Master-thesis/TFLite/data/big.jpg', help='initial weights path')
    opt = parser.parse_args()

    detector = Detector(opt.weights)
    detector.load_image(opt.image)
    detector.tile_image()

    while detector.current_tile != -1:
        detector.load_input(verbose=True)
        detector.detect()
        detector.get_output()
    then = time.time()
    print(f"Time taken: {then-now}")

