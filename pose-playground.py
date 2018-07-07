import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import math

# Recommended size: 432x368
# Using mobilenet_thin model for accuracy

w = 432
h = 368
estimator = TfPoseEstimator(get_graph_path("mobilenet_thin"), target_size=(432, 368))

cam = cv2.VideoCapture(0)
ret_val, image = cam.read()

while True:
    ret_val, image = cam.read()

    humans = estimator.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)

    for human in humans:
        #let's locate the general head area

        headparts = [0, 1, 14, 15, 16, 17]

        left = image.shape[1]
        right = 0
        top = image.shape[0]
        bottom = 0

        nose = None
        neck = None

        for body_part in human.body_parts:
            print(human.body_parts[body_part])

            if human.body_parts[body_part].part_idx in headparts:
                # Detect the leftmost, rightmost, bottommost, and topmost positions
                x = math.floor(human.body_parts[body_part].x * image.shape[1])
                y = math.floor(human.body_parts[body_part].y * image.shape[0])

                if x < left:
                    left = x
                if x > right:
                    right = x
                if y < top:
                    top = y
                if y > bottom:
                    bottom = y

                if body_part == 0:
                    nose = human.body_parts[body_part]
                elif body_part == 1:
                    neck = human.body_parts[body_part]

        if neck is not None and nose is not None:
            nn_distance = math.floor((neck.y - nose.y) * image.shape[0])
            
            top = top - nn_distance
            if top < 0:
                top = 0

        cv2.rectangle(image, (left, top), (right, bottom), (0,0, 255), 2, 0)


    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

    cv2.imshow('test', image)

    if cv2.waitKey(1) == 27:
        break