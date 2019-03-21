# -*- coding: utf-8 -*-
# File: display_image.py

import cv2
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
"""
Example for display_image_from_dict():

import numpy as np
x={}
path = r'/home/koby_a/tensorflow/tf_so_distribution/tf_cpp_test/test/check_tank.jpg'
x['file_name'] = path
x['boxes'] = np.asarray([[73.2616,47.3668,578.927,272.847],[105.019,105.163,275.048,263.989],[385.819,131.032,568.592,249.198]],dtype=np.uint16)
import display_image
display_image.display_image_from_dict(x)
"""
def display_image_from_dict(img_dict):
    im = np.float32(Image.open(img_dict['file_name']))
    img = tf.image.rgb_to_yuv(im)
    with tf.Session() as sess:
        img = img.eval()
        #im_present = tf.concat([tf.expand_dims(t, 2) for t in [img[:, :, 0],img[:, :, 0],img[:, :, 0]]], 2)
        im_present = np.uint8(img[:, :, 0])
        boxes = img_dict['boxes']
        for c in boxes:
            cv2.rectangle(im_present, (c[0], c[1]), (c[2], c[3]), (0, 255, 0), 2)
        plt.imshow(im_present,cmap='gray')
        plt.show()

def rgb2yuv(path):
    im = np.float32(Image.open(path))
    img = tf.image.rgb_to_yuv(im)
    #img = cv2.cvtColor(im, cv2.COLOR_RGB2YUV)
    #img = tf.image.yuv_to_rgb(img)
    with tf.Session() as sess:
        arr = img.eval()
        #arr = img
        x = arr[:,:,0]
        np.save(r'/home/koby_a/tensorflow/tf_so_distribution/tf_cpp_test/test/Y.npy',x)
        print(type(x[1,0]))
        print(np.max(x))
        print(np.min(x))
        x = arr[:,:,1] + 128
        np.save(r'/home/koby_a/tensorflow/tf_so_distribution/tf_cpp_test/test/U.npy',x)
        print(np.max(x))
        print(np.min(x))
        x = arr[:,:,2] + 128
        np.save(r'/home/koby_a/tensorflow/tf_so_distribution/tf_cpp_test/test/V.npy',x)
        print(np.max(x))
        print(np.min(x))
        plt.imshow(x,cmap='gray')
        plt.show()