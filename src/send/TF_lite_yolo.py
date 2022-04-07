from turtle import pos
import tensorflow as tf
import cv2
import numpy as np
import torch
import torchvision

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[0] = x[0] - x[2] / 2  # top left x
    y[1] = x[1] - x[3] / 2  # top left y
    y[2] = x[0] + x[2] / 2  # bottom right x
    y[3] = x[1] + x[3] / 2  # bottom right y
    return y

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="/home/sara/Documents/Master-thesis/TFLite/tf_lite/epoch80-fp16.tflite")
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
print("OUTPUT_DATA: ", pred)

boxes = []
confs = []
info = []

nr_bounding_boxes = 0
conf_threshold = 0.5
for det in pred:

    *xywh, conf, cls = det
    conf *= cls
    if conf >= conf_threshold:
        nr_bounding_boxes += 1
        cls = int(cls)

        xyxy = xywh2xyxy(np.asarray(xywh[:4]))

        p1, p2 = (int(xyxy[0]*img_size), int(xyxy[1]*img_size)), (int(xyxy[2]*img_size), int(xyxy[3]*img_size))

        boxes.append(xyxy.tolist())
        confs.append(conf)
        info.append([p1, p2, cls])
        
print(f"Nr bounding boxes over conf {conf_threshold} detected: {nr_bounding_boxes}")

b = torch.FloatTensor(boxes)
c = torch.FloatTensor(confs)
indexes = torchvision.ops.nms(b,c, iou_threshold=0.45)

for index in indexes:
    inf = info[index]

    print("")
    print(f"confidence: {c[index]}")
    print(f"class: {inf[2]}")
    print(f"position xyxy: {inf[0]}, {inf[1]}")
    
    cv2.rectangle(image, inf[0], inf[1], (128, 128, 128))    
    

cv2.imshow("Input", image)
cv2.waitKey(0)

