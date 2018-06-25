import cv2
import imutils

import tensorflow as tf
from moviepy.editor import VideoFileClip
from IPython.display import HTML

from detector import *

SSD_GRAPH_FILE = '../CarND-Object-Detection-Lab/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'

clip = VideoFileClip('../CarND-Object-Detection-Lab/driving.mp4')#.subclip(0,5)



camera = cv2.VideoCapture(0)

detection_graph = load_graph(SSD_GRAPH_FILE)
# detection_graph = load_graph(RFCN_GRAPH_FILE)
# detection_graph = load_graph(FASTER_RCNN_GRAPH_FILE)



with tf.Session(graph=detection_graph) as sess:
    image_tensor = sess.graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = sess.graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = sess.graph.get_tensor_by_name('detection_scores:0')
    detection_classes = sess.graph.get_tensor_by_name('detection_classes:0')
    

    while True:
        (grabber, frame) = camera.read()

        frame = pipeline(frame, sess, detection_boxes, detection_scores, detection_classes, image_tensor)

        cv2.imshow('Frame', frame)
        cv2.waitKey(1)




camera.release()
cv2.destroyAllWindows()


