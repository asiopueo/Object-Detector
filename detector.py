#
# Object Detector
# Martin Lippl
# 
#
# Program which uses the stream of a webcam and applies a Mobile net to it.
#
#

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor
import time
from scipy.stats import norm


# Frozen inference graph files.
SSD_GRAPH_FILE = '../CarND-Object-Detection-Lab/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
#RFCN_GRAPH_FILE = 'rfcn_resnet101_coco_11_06_2017/frozen_inference_graph.pb'
#FASTER_RCNN_GRAPH_FILE = 'faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017/frozen_inference_graph.pb'

# Colors (one for each class)
cmap = ImageColor.colormap
print("Number of colors =", len(cmap))
COLOR_LIST = sorted([c for c in cmap.keys()])


# Return boxes with a confidence >= `min_score`
def filter_boxes(min_score, boxes, scores, classes):
    n = len(classes)
    idxs = []
    for i in range(n):
        if scores[i] >= min_score:
            idxs.append(i)
    
    filtered_boxes = boxes[idxs, ...]
    filtered_scores = scores[idxs, ...]
    filtered_classes = classes[idxs, ...]
    return filtered_boxes, filtered_scores, filtered_classes

# The original box coordinate output is normalized, i.e [0, 1].
# This converts it back to the original coordinate based on the image size.
def to_image_coords(boxes, height, width):
    box_coords = np.zeros_like(boxes)
    box_coords[:, 0] = boxes[:, 0] * height
    box_coords[:, 1] = boxes[:, 1] * width
    box_coords[:, 2] = boxes[:, 2] * height
    box_coords[:, 3] = boxes[:, 3] * width
    
    return box_coords


# Draw bounding boxes on the image
def draw_boxes(image, boxes, classes, thickness=4):
    #draw = ImageDraw.Draw(image)
    tmp = Image.fromarray(image)
    draw = ImageDraw.Draw(tmp)
    for i in range(len(boxes)):
        bot, left, top, right = boxes[i, ...]
        class_id = int(classes[i])
        color = COLOR_LIST[class_id]
        draw.line([(left, top), (left, bot), (right, bot), (right, top), (left, top)], width=thickness, fill=(0,0,255))

    return np.asarray(tmp)


# Loads a frozen inference graph        
def load_graph(graph_file):
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph




def pipeline(img, sess, detection_boxes, detection_scores, detection_classes, image_tensor):
    image_np = np.expand_dims(np.asarray(img, dtype=np.uint8), 0)

    # Actual detection.
    (boxes, scores, classes) = sess.run([detection_boxes, detection_scores, detection_classes], feed_dict={image_tensor: image_np})
    # Remove unnecessary dimensions
    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    classes = np.squeeze(classes)

    confidence_cutoff = 0.5
    # Filter boxes with a confidence score less than `confidence_cutoff`
    boxes, scores, classes = filter_boxes(confidence_cutoff, boxes, scores, classes)

    # The current box coordinates are normalized to a range between 0 and 1.
    # This converts the coordinates actual location on the image.
    height = img.shape[0]
    width = img.shape[1]
    box_coords = to_image_coords(boxes, height, width)
    #print(box_coords)

    # Each class with be represented by a differently colored box
    return draw_boxes(img, box_coords, classes)






if __name__=='__main__':
    detection_graph = load_graph(SSD_GRAPH_FILE)
    # detection_graph = load_graph(RFCN_GRAPH_FILE)
    # detection_graph = load_graph(FASTER_RCNN_GRAPH_FILE)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')


    #image = Image.open('./sample1.jpg')
    image = mpimg.imread('./sample1.jpg')

    with tf.Session(graph=detection_graph) as sess:
        image = pipeline(image, sess, detection_boxes, detection_scores, detection_classes, image_tensor)

    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.show()




