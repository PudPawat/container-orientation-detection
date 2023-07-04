from numpy import *
import math
from statistics import mean
import numpy as np


def fit_circle_2d(x, y, w=[]):
    '''
    https://meshlogic.github.io/posts/jupyter/curve-fitting/fitting-a-circle-to-cluster-of-3d-points/
    fit circle by least square
    MRTHOD: using circle regression to fit the circle by those list of x and list of y
    :param x: all list of x points
    :param y: all list of y points
    :param w:
    :return:
    '''
    x = np.array(x)
    y = np.array(y)
    A = array([x, y, ones(len(x))]).T
    b = x * x.transpose() + y * y.transpose()

    # Modify A,b for weighted least squares
    if len(w) == len(x):
        W = diag(w)
        A = dot(W, A)
        b = dot(W, b)

    # Solve by method of least squares
    c = linalg.lstsq(A, b, rcond=None)
    # print("c",c)
    c = c[0]

    # Get circle parameters from solution c
    xc = c[0] / 2
    yc = c[1] / 2
    r = sqrt(c[2] + xc ** 2 + yc ** 2)
    loss = r - mean(sqrt((x - ones(len(x))*xc)**2 +(y - ones(len(y))*yc)**2))
    # print("loss",loss)
    return xc, yc, r, loss

def get_x_y_from_contour(contour):
    '''
    To collext data x,y coordinate from the points in contours
    get all x coords and all y corrds from contour
    :param contour:
    :return: list of all x , list of all y coords
    '''
    x = []
    y = []
    for coord in contour:
        x.append(coord[0][0])
        y.append(coord[0][1])
    return x, y