import cv2
import numpy as np


def create_big_im():
    img = np.zeros([10000,10000,3], np.uint8)
    filename = "/home/sara/Documents/Master-thesis/TFLite/data/llbig.bmp"
    cv2.imwrite(filename, img)

def calculate_tiles(imgsize, inputsize):
    w = np.floor(imgsize[0]/inputsize[0])
    h = np.floor(imgsize[1]/inputsize[1])
    print(f"nr_complete_tiles_w: {w}")
    print(f"nr_complete_tiles_h: {h}")
    print(f"nr_complete_tiles_total: {w*h}")

    print(f"leftover pixels - right: {(imgsize[0] % inputsize[0])}")
    print(f"leftover pixels - bottom: {(imgsize[1] % inputsize[1])}")

    print(f"last tile start - right: {(imgsize[0] - inputsize[0])}")
    print(f"last tile start - bottom: {(imgsize[1] - inputsize[1])}")

    print(f"edge tiles: {w+h+1}")


create_big_im()

#calculate_tiles(imgsize = (10000, 10000), inputsize = (768,768))