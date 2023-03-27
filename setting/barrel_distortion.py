import numpy as np
import cv2
crop_circle_fix_inner_r =  [537.1886663632225, 402.60514337733275, 130.1279930268229]
crop_circle_fix_inner_r =  [538.1886663632225, 402.60514337733275, 250.1279930268229]

def resize_scale(img, scale = 0.3):
    resize = cv2.resize(img,(int(img.shape[1]*scale),int(img.shape[0]*scale)))
    return resize


def crop_circle(img, circle, resize = True):
    '''
    :param img:
    :param circle: (x,y,r)
    :return:
    '''
    # if resize:
    #     img = resize_scale(img, 0.3)
    x,y,r = circle
    x,y,r = int(x/0.3), int(y/0.3), int(r/0.3)
    crop = img[y-r:y+r,x-r:x+r]
    return crop

src1 = cv2.imread("../dataset/20230318/S00011_1.png")
cv2.namedWindow('src1',cv2.WINDOW_NORMAL)
cv2.imshow('src1', src1)
src = crop_circle(src1, crop_circle_fix_inner_r)
cv2.namedWindow('src',cv2.WINDOW_NORMAL)
cv2.imshow('src', src)

width = src.shape[1]
height = src.shape[0]

distCoeff = np.zeros((4, 1), np.float64)

# TODO: add your coefficients here!
k1 = -7.0e-5;  # negative to remove barrel distortion
print(k1)
k2 = 0.0;
p1 = 0;
p2 = 0;

# p1 = -5.0e-5;
# p2 = -5.0e-5;

distCoeff[0, 0] = k1;
distCoeff[1, 0] = k2;
distCoeff[2, 0] = p1;
distCoeff[3, 0] = p2;

# assume unit matrix for camera
cam = np.eye(3, dtype=np.float32)

cam[0, 2] = width / 2.0  # define center x
cam[1, 2] = height / 2.0  # define center y
cam[0, 0] = 10.  # define focal length x
cam[1, 1] = 10.  # define focal length y

# here the undistortion will be computed
dst = cv2.undistort(src, cam, distCoeff)

# cv2.namedWindow('dst',cv2.WINDOW_NORMAL)
cv2.imshow('dst', dst)
cv2.imwrite("test.jpg", dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

distCoeff[0, 0] = -k1;
distCoeff[1, 0] = k2;
distCoeff[2, 0] = p1;
distCoeff[3, 0] = p2;

# assume unit matrix for camera
cam = np.eye(3, dtype=np.float32)

cam[0, 2] = width / 2.0  # define center x
cam[1, 2] = height / 2.0  # define center y
cam[0, 0] = 10.  # define focal length x
cam[1, 1] = 10.  # define focal length y

dst = cv2.undistort(src, cam, distCoeff)

# cv2.namedWindow('dst',cv2.WINDOW_NORMAL)
cv2.imshow('dst1', dst)
cv2.imwrite("test_3.jpg", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
