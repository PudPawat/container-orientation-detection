import numpy as np
import cv2
import math
import os
import json

from utils import save_json, open_json
# from bytecode import *
# try:
#     from utils import save_json, open_json
# except:
#     from ..utils import save_json, open_json
drawing = False  # true if mouse is pressed
ix, iy = -1, -1


# Create a function based on a CV2 Event (Left button click)
class Draw():
    def __init__(self):
        self.circle = None
        self.count = 0
        self.coords = []

    def draw_circle_one_click(self,event, x, y, flags, param):
        global ix, iy, drawing

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            # we take note of where that mouse was located
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            drawing == True


        elif event == cv2.EVENT_LBUTTONUP:
            radius = int(math.sqrt(((ix - x) ** 2) + ((iy - y) ** 2)))
            cv2.circle(img, (ix, iy), radius, (0, 0, 255), thickness=1)
            self.circle = ((ix, iy),radius)
            drawing = False
            return circle

    def get_coord(self,event,x,y, flags, param):
        # coords = []
        # count = 0
        if event == cv2.EVENT_LBUTTONDOWN:
            # we take note of where that mouse was located
            ix, iy = x, y
            self.coords.append([ix, iy])
            self.count +=1
            self.coord = (x,y)
            print(self.coord)
            cv2.putText(img,str(self.coord),self.coord, cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,200),1)
            if self.count == 3:
                print("Draw circle",self.count)
                self.circle = self.define_circle(self.coords[0],self.coords[1],self.coords[2]) # return ((cx, cy), radius)
                print(self.circle, self.circle[0])
                cv2.circle(img, (int(self.circle[0][0]),int(self.circle[0][1])), int(self.circle[1]), (0, 0, 255), thickness=1)
                cv2.circle(img, (int(self.circle[0][0]),int(self.circle[0][1])), int(0), (0, 0, 255), thickness=3)
                self.count = 0
                self.coords = []


    @staticmethod
    def define_circle(p1, p2, p3):
        """
        Returns the center and radius of the circle passing the given 3 points.
        In case the 3 points form a line, returns (None, infinity).
        """
        temp = p2[0] * p2[0] + p2[1] * p2[1]
        bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
        cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
        det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

        if abs(det) < 1.0e-6:
            return (None, np.inf)

        # Center of circle
        cx = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
        cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

        radius = np.sqrt((cx - p1[0]) ** 2 + (cy - p1[1]) ** 2)

        return ((cx, cy), radius)



if __name__ == '__main__':

    draw = Draw()
    default_config = {
      "params": {"HSV": [0, 0, 55, 180, 255, 255], "gaussianblur": [1, 1], "dilate": [5, 0], "erode": [15, 0]},
      "debug": "True",
      "show_result": "True",
      "flag_rotate": "None",
      "resize_ratio": 0.3,
      "save_img": "True",
      "inverse_threshold": "True",
      "n_symetric": 1,
      "crop_circle_fix_inner_r": [537, 400, 175],
      "crop_circle_fix_outer_r": [537, 402, 205],
      "crop_circle_platform": [537, 400, 391],
      "method_names": ["simple", "compare", "comparev2"],
      "method": ""
    }


    config_folder = "./config/"
    name_format = "notchv2_config_"

    circle = None
    circles = []
    # path = "F:\Ph.D\circle_classification\container-orientation-detection\dataset\\new_0219\\"
    path = "F:\Ph.D\contactlens\contact_lens_project\data\\new company1\\burr\FOV1"
    
    # path = "./dataset/distort"
    names = os.listdir(path)
    print(names)
    for name in names:
        config_class = name.split("_")[0]
        try:
            # with open('../config/notchv2_config_{}.json'.format(config_class), 'r') as openfile:
            config  = open_json(config_folder,name_format,config_class)
            #     config = json.load(openfile)
            print(config)
        except:
            config = default_config


        try:
            ratio = config["resize_ratio"]
        except:
            ratio = 0.3
        img = cv2.imread(os.path.join(path,name))
        img = cv2.resize(img,(int(img.shape[1]*ratio),int(img.shape[0]*ratio)))

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', draw.get_coord)
        # cv2.setMouseCallback('image', draw.draw_circle_one_click)

        while (1):

            cv2.imshow('image', img)
            if circle != draw.circle:
                print("draw.circle",draw.circle)
                circle = draw.circle
                print(circle)
                circle_resize = [(int(circle[0][0]/0.3),int(circle[0][1]/0.3)), int(circle[1]/0.3)]
                circles.append(circle_resize)

            # EXPLANATION FOR THIS LINE OF CODE:
            # https://stackoverflow.com/questions/35372700/whats-0xff-for-in-cv2-waitkey1/39201163
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord("r"):
                if circles != []:
                    del circles[-1]
                print("circles",circles)

            elif k == ord("i"): ## inner
                inner_circle = [circle[0][0], circle[0][1],circle[-1]]
                config["crop_circle_fix_inner_r"] = inner_circle
                print("inner_circle: ", inner_circle)
                save_json(config_folder, name_format, config_class, config)
                # with open('../config/notchv2_config_{}.json'.format(config_class), 'w') as openfile:
                #     json.dump(config, openfile)
            elif k == ord("o"): ## outer
                outer_circle = [circle[0][0], circle[0][1],circle[-1]]
                config["crop_circle_fix_outer_r"] = outer_circle
                print("outer_circle: ", outer_circle)
                save_json(config_folder, name_format, config_class, config)
                # with open('../config/notchv2_config_{}.json'.format(config_class), 'w') as openfile:
                #     json.dump(config, openfile)
            elif k == ord("p"): ## platform
                platfrom_circle = [circle[0][0], circle[0][1],circle[-1]]
                config["crop_circle_platform"] = platfrom_circle
                print("platfrom_circle: ", platfrom_circle)
                save_json(config_folder,name_format,config_class, config)


        try:
            crop_circle = draw.circle
            img_crop = img[int(crop_circle[1]):int(crop_circle[1] + crop_circle[2]),
                       int(crop_circle[0]):int(crop_circle[0] + crop_circle[2])]
            print("final circle", circles)
        except:
            pass
        cv2.destroyAllWindows()