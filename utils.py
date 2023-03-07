import cv2
import numpy as np
from lib.warp_and_reverse_warp import warp_polar, reverse_warp

def resize_scale(img, scale = 0.3):
    resize = cv2.resize(img,(int(img.shape[1]*scale),int(img.shape[0]*scale)))
    return resize

def crop_circle(img, circle, resize = True):
    '''

    :param img:
    :param circle: (x,y,r)
    :return:
    '''
    if resize:
        img = resize_scale(img)
    x,y,r = circle
    crop = img[y-r:y+r,x-r:x+r]
    return crop

def crop_circle_by_warp(img, circle):
    img, warp_img = warp_polar(img,circle)
    crop_circle_img = reverse_warp(img,warp_img, circle)
    return crop_circle_img


def rotate_PIL(img, center = "center"):
    pass

def rotate_image(image, angle, center = None):
    if center == None:
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
    else:
        image_center = tuple(center)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def fill_balck_circle(img, circle):
    x,y,r = circle
    img = cv2.circle(img, (x,y),r,(0,0,0),-1)
    return img
