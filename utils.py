import cv2
import numpy as np

def resize_scale(img, scale = 0.3):
    resize = cv2.resize(img,(int(img.shape[1]*scale),int(img.shape[0]*scale)))
    return resize

def crop_circle(img, circle):
    '''

    :param img:
    :param circle: (x,y,r)
    :return:
    '''
    img = resize_scale(img)
    x,y,r = circle
    crop = img[y-r:y+r,x-r:x+r]
    return crop