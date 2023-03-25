import cv2 as cv
import numpy as np
import math
# from lib.trackbar import *


class Imageprocessing(object):

    def __init__(self):
        pass

    def read_params(self, params, frame, print = False):
        """
        Function Name: read_params

        Description: read all image processing from json file eg. threshold, HSV inrange
        and put those parameters to process in Imageorocessing()

        Argument:
            params [dict] -> [all parameters]
            frame [array] -> [image for processing]

        Parameters:

        Return:
            frame [array] -> [image after process]

        Edited by: [12-07-2020] [Pawat]
        """
        frame_proc = {}
        circle = {}
        line = {}
        for key in params.keys():
            if print:
                print(key)
            circle = []
            line = []
            # frame_result1 = frame.copy()
            # frame = cv2.resize(frame_result1, (int(frame_result1.shape[1] / self.opt.basic.resize_factor),
            #                                   int(frame_result1.shape[0] / self.opt.basic.resize_factor)))
            # frame_result = frame.copy()
            if key == "HSV":
                # frame_HSV, params['HSV'] = imgproc.HSV_range(frame, params[key])
                frame, params['HSV'] = self.imgproc.HSV_range(frame, params[key])
                frame_proc["HSV"] = frame

            elif key == "erode":
                # frame_erode, params['erode'] = imgproc.erode(frame, params[key])
                frame, params['erode'] = self.imgproc.erode(frame, params[key])
                frame_proc["erode"] = frame

            elif key == "dilate":
                # frame_dialte, params['dilate'] = imgproc.dilate(frame, params[key])
                frame, params['dilate'] = self.imgproc.dilate(frame, params[key])
                frame_proc["dilate"] = frame

            elif key == "thresh":
                # frame_binary, params['thresh'] = imgproc.threshold(frame, params[key])
                frame, params['thresh'] = self.imgproc.threshold(frame, params[key])
                frame_proc["thresh"] = frame

            elif key == "sharp":
                # frame_sharp, params['sharp'] = imgproc.sharpen(frame, params[key])
                frame, params['sharp'] = self.imgproc.sharpen(frame, params[key])
                frame_proc["sharp"] = frame

            elif key == "blur":
                # frame_blur, params['blur'] = imgproc.blur(frame, params[key])
                frame, params['blur'] = self.imgproc.blur(frame, params[key])
                frame_proc["blur"] = frame

            elif key == "gaussianblur":
                frame, params["gaussianblur"] = self.imgproc.gaussianblur(frame,params[key])
                frame_proc["gaussianblur"] = frame

            elif key == "line":
                # frame_line, lines, params['line'] = imgproc.line_detection(frame, frame0, params[key])
                if len(frame.shape) == 2:
                    frame0 = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
                frame, lines, params['line'] = self.imgproc.line_detection(frame, frame0, params[key])
                frame_proc["line"] = frame

            elif key == "canny":
                # frame_canny, params['canny'] = imgproc.canny(frame, params[key], show=True)
                frame, params['canny'] = self.imgproc.canny(frame, params[key], show=False)
                frame_proc["canny"] = frame

            elif key == "circle":
                # frame_circle, circle, params['circle'] = imgproc.circle_detection(frame, frame0, params[key], show=False)
                if len(frame.shape) == 2:
                    frame0 = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
                frame, circle, params['circle'] = self.imgproc.circle_detection(frame, frame0, params[key], show=False)
                frame_proc["circle"] = frame

            elif key == "sobel":
                frame, params["sobel"] = self.imgproc.sobel(frame,params[key],show=False)
                frame_proc["sobel"] = frame

        frame_proc["final"] = frame

        return frame_proc, circle, line

    def threshold(self, img, params, show = False):
        """
        Function Name: threshold
        
        Description: setting threshold value
        
        Argument:
            img [array] -> [image for thresholding]
            params [tuple] -> [all need params]
        
        Parameters:
        
        Return:
            img[array] -> [thresholded image]
            params[tuple] -> (th_val)
        
        Edited by: [12-4-2020] [Pawat]
        """        
        th_val = params
        if th_val == 0 :
            flag = cv.THRESH_BINARY+cv.THRESH_OTSU

        else:
            flag = cv.THRESH_BINARY

        if len(img.shape) == 3 :
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        _, th = cv.threshold(img,th_val, 255, flag)
        if show == True:
            cv.imshow("window_thresh", th)

        return th, (th_val)

    def canny(self, img, params, show = False):
        """
        Function Name: canny
        
        Description: Canny edge detection. there is two params X ,Y
        
        Argument:
            img [array] -> [image for thresholding]
            params [tuple] -> [all need params]
        
        Parameters:
        
        Return:
            img[array] -> [edge image]
            params[tuple] -> (Y_val, X_val)
        
        Edited by: [12-04-2020] [Pawat]
        """        
        Y_val, X_val = params

        if len(img.shape) == 3 :
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        canny = cv.Canny(img, Y_val, X_val)
        if show == True:
            cv.imshow("window_canny", canny)
        return canny, (Y_val, X_val)

    def canny_1(self,img,params, show = False):
        """
        Function Name: canny_1
        
        Description: another for parallel processing. there is two params X ,Y
        
        Argument:
            img [array] -> [image for thresholding]
            params [tuple] -> [all need params]
        
        Parameters:
        
        Return:
            img[array] -> [edge image]
            params[tuple] -> (Y_val, X_val)
        
        Edited by: [12-04-2020] [Pawat]
        """ 
        Y_val, X_val = params

        if len(img.shape) == 3 :
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        canny = cv.Canny(img, Y_val, X_val)
        if show == True:
            cv.imshow("var_canny_1", canny)
        return canny, (Y_val, X_val)

    def blur(self,img,params, show = False):
        """
        Function Name: blur
        
        Description: Bluring image by setting filter size
        
        Argument:
            img [array] -> [image for thresholding]
            params [tuple] -> [all need params]
        
        Parameters:
        
        Return:
            img[array] -> [blured image]
            params[tuple] -> (filter_size)
        
        Edited by: [12-04-2020] [Pawat]
        """        
        filter_size = params
        if filter_size < 1 :
            filter_size = 1

        blur = cv.blur(img, (int(filter_size), int(filter_size)))

        if show == True:
            cv.imshow("window_blur", blur)

        return blur,(filter_size)

    def gaussianblur(self,img,params, show = True):
        '''
        Buring
        :param img:
        :param show:
        :return: blur,(filter_size)
        '''

        x,y = params

        if not (x > 0 and x % 2 == 1):
            x = x+1
        if not (y > 0 and y % 2 == 1):
            y =y +1
        blur = cv.GaussianBlur(img, (int(x), int(y)),0)

        if show == True:
            cv.imshow("gaussianblur", blur)

        return blur,(x,y)

    def HSV_range(self,img,params,show = False, mode = "HSV"):
        """
        Function Name: HSV_range
        
        Description: HSV thresholding by setting lower bound and upper bound
        of Hue, Satuation, Value channel
        
        Argument:
            img [array] -> [image for thresholding]
            params [tuple] -> [all need params]
        
        Parameters:
        
        Return:
            img[array] -> [thresholded by HSV image]
            params[tuple] -> (low_H, low_S, low_V, high_H, high_S, high_V)
        
        Edited by: [12-04-2020] [Pawat]
        """ 
        low_H, low_S, low_V, high_H, high_S, high_V = params

        if len(img.shape) != 3:
            img = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
        if mode == "HSV":
            frame_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

            # frame_HSV = cv.cvtColor(frame_HSV, cv.COLOR_HSV2BGR)

            frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
        elif mode == "HLS":
            frame_HLS = cv.cvtColor(img, cv.COLOR_BGR2HLS)
            frame_threshold = cv.inRange(frame_HLS, (low_H, low_V, low_S), (high_H, high_V, high_S))

        if show == True:
            cv.imshow("window_HSV", frame_threshold)

        return  frame_threshold, [low_H, low_S, low_V, high_H, high_S, high_V]

    def HSV_range_1(self,img ,params ,show = False, mode = "HSV"):
        """
        Function Name: HSV_range
        
        Description: HSV thresholding by setting lower bound and upper bound
        of Hue, Satuation, Value channel
        
        Argument:
            img [array] -> [image for thresholding]
            params [tuple] -> [all need params]
        
        Parameters:
        
        Return:
            img[array] -> [thresholded by HSV image]
            params[tuple] -> (low_H, low_S, low_V, high_H, high_S, high_V)
        
        Edited by: [12-04-2020] [Pawat]
        """
        low_H, low_S, low_V, high_H, high_S, high_V = params

        if mode == "HSV":
            frame_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

            # frame_HSV = cv.cvtColor(frame_HSV, cv.COLOR_HSV2BGR)

            frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
        elif mode == "HLS":
            frame_HLS = cv.cvtColor(img, cv.COLOR_BGR2HLS)
            frame_threshold = cv.inRange(frame_HLS, (low_H, low_V, low_S), (high_H, high_V, high_S))

        if show == True:
            cv.imshow("window_HSV1", frame_threshold)

        return  frame_threshold, [low_H, low_S, low_V, high_H, high_S, high_V]


    def HSV_adjustment(self, img, factor_H, factor_S, factor_V):

        frame_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        # checkpoint to continue
        # purpose of this function is adjust like lighrroom

    def sharpen(self,img,params, show = False):
        """
        Function Name: sharpen
        
        Description: Shapen process by factor multiply by filter
        
        Argument:
            img [array] -> [image for thresholding]
            params [tuple] -> [all need params]
        
        Parameters:
        
        Return:
            img[array] -> [sharpen image]
            params[tuple] -> (factor)
        
        Edited by: [12-04-2020] [Pawat]
        """        
        factor = params
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        kernel = (factor/10) * kernel
        img = cv.filter2D(img, -1, kernel)
        # if show == True:
        #     cv.imshow(self.var_sharpen.window_sharp_name, img)
        # checkpoint to continue
        return img, (factor)


    def line_detection(self, img, draw_img,params, show = False):
        """
        Function Name: line_detection
        
        Description: Line detection
        with the following arguments:
        dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
        lines: A vector that will store the parameters (r,θ) of the detected lines
        rho : The resolution of the parameter r in pixels. We use 1 pixel. ( 1 to 10 )
        theta: The resolution of the parameter θ in radians. We use 1 degree (CV_PI/180) ( 30 - 180 )
        threshold: The minimum number of intersections to "*detect*" a line
        srn and stn: Default parameters to zero. Check OpenCV reference for more info.
        
        Argument:
            img [array] -> [image for detection]
            draw_img [array] -> [image for drawing]
            params [tuple] -> [parameters]
        
        Parameters:
        
        Return:
            image [array] -> [drown image]
            line [list] -> [list of detected line]
            params[tuple] -> (rho1, theta2, threshold3, none4, srn5, stn6)
        
        Edited by: [12-04-2020] [Pawat]
        """
        # copy_img = img.copy()
        if len(img.shape) == 3 :
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # copy_img = img.copy()
        rho1, theta2, threshold3, none4, srn5, stn6 = self.var_line_det.return_var()
        if rho1 == 0:
            rho1 = 1
        if theta2 == 0:
            theta2 = 1
        lines = cv.HoughLines(img, rho1, np.pi / theta2, threshold3, None, srn5, stn6)

        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                cv.line(draw_img, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

        if show == True:
            cv.imshow("window_line_detection", draw_img)

        return draw_img,lines , (rho1, theta2, threshold3, none4, srn5, stn6)

    def circle_detection(self, img,draw_img,params ,show= False):
        """
        Function Name: circle_detection
        
        Description: Circle detection by opencv
        
        Argument:
            img [array] -> [image for detection]
            draw_img [array] -> [image for drawing]
            params [tuple] -> [parameters]
        
        Parameters:
        
        Return:
            img[array] -> [drawn image]
            circle[list] -> list of circle
            params[tuple] -> (param1,param2, min, max)
        
        Edited by: [12-04-2020] [Pawat]
        """  

        if len(img.shape) == 3 :
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        rows =  img.shape[0]

        param1,param2, min,max = params

        if param1 == 0:
            param1 = 1
        if param2 == 0:
            param2 = 1
        circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, rows / 5,
                                   param1=param1, param2=param2,
                                   minRadius=min, maxRadius=max)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv.circle(draw_img, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv.circle(draw_img, center, radius, (255, 0, 255), 3)


        if show == True:
            cv.imshow("window_circle", draw_img)

        return img, circles, (param1,param2, min, max)

    def circle_detection_1(self, img,draw_img,params, show= False):
        """
        Function Name: circle_detection_1
        
        Description: Circle detection by opencv
        
        Argument:
            img [array] -> [image for detection]
            draw_img [array] -> [image for drawing]
            params [tuple] -> [parameters]
        
        Parameters:
        
        Return:
            img[array] -> [drawn image]
            circle[list] -> list of circle
            params[tuple] -> (param1,param2, min, max)
        
        Edited by: [12-04-2020] [Pawat]
        """

        if len(img.shape) == 3 :
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        rows =  img.shape[0]

        param1,param2, min, max = params

        if param1 == 0:
            param1 = 1
        if param2 == 0:
            param2 = 1
        circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, rows / 5,
                                   param1=param1, param2=param2,
                                   minRadius=min, maxRadius=max)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv.circle(draw_img, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv.circle(draw_img, center, radius, (255, 0, 255), 3)


        if show == True:
            cv.imshow("window_circle1", draw_img)

        return img, circles, (param1,param2, min, max)

    def dilate(self, img,params, show = False):
        """
        Function Name: dilate
        
        Description: dilation processing :making white parts bigger follow kernel shape and size
         type: 1 = RECTANGLE,2 = OPEN ,3 = Cross ,4 = DILATE ,5 = ERODE ,6 = ELLIPSE
        Argument:
            img [array] -> [image for thresholding]
            params [tuple] -> [all need params]
        
        Parameters:
        
        Return:
            img[array] -> [dilated image]
            params[tuple] -> (kernel_size, type_kernel)
        
        Edited by: [12-04-2020] [Pawat]
        """        
        # print("Note : \n type: 1 = RECTANGLE,2 = OPEN ,3 = Cross ,4 = DILATE ,5 = ERODE ,6 = ELLIPSE")
        kernel_size, type_kernel = params

        if type_kernel == 1:
            type_kernel = cv.MORPH_RECT #ok
        elif type_kernel == 2:
            type_kernel = cv.MORPH_OPEN #ok
        elif type_kernel == 3:
            type_kernel = cv.MORPH_CROSS #ok
        elif type_kernel == 4:
            type_kernel = cv.MORPH_DILATE #ok
        elif type_kernel == 5:
            type_kernel = cv.MORPH_ERODE #ok
        elif type_kernel == 6:
            type_kernel = cv.MORPH_ELLIPSE #ok
        else:
            type_kernel = cv.MORPH_ELLIPSE

        if kernel_size == 0:
            kernel_size = 1
        kernel = cv.getStructuringElement(type_kernel, (kernel_size, kernel_size))

        dialate = cv.dilate(img, kernel, iterations=1)

        if show == True:
            cv.imshow("window_dilate_det", dialate)

        return dialate, (kernel_size, type_kernel)


    def erode(self, img,params, show = False):
        """
        Function Name: erode
        
        Description: making white parts smaller follow kernel shape and size
        type: 1 = RECTANGLE,2 = OPEN ,3 = Cross ,4 = DILATE ,5 = ERODE ,6 = ELLIPSE
        
        Argument:
            img [array] -> [image for thresholding]
            params [tuple] -> [all need params]
        
        Parameters:
        
        Return:
            image[array] -> [eroded image]
            params[array] -> [kernel_size, type_kernel]
        
        Edited by: [12-04-2020] [Pawat]
        """        
        # print("Note : \n type: 1 = RECTANGLE,2 = OPEN ,3 = Cross ,4 = DILATE ,5 = ERODE ,6 = ELLIPSE")
        kernel_size, type_kernel = params
        # "ty:1REC,2GRA,3Cro,4DIA,5SQR,6STA,7ELIP"
        if type_kernel == 1:
            type_kernel = cv.MORPH_RECT  # ok
        elif type_kernel == 2:
            type_kernel = cv.MORPH_OPEN  # ok
        elif type_kernel == 3:
            type_kernel = cv.MORPH_CROSS  # ok
        elif type_kernel == 4:
            type_kernel = cv.MORPH_DILATE  # ok
        elif type_kernel == 5:
            type_kernel = cv.MORPH_ERODE  # ok
        elif type_kernel == 6:
            type_kernel = cv.MORPH_ELLIPSE  # ok
        else:
            type_kernel = cv.MORPH_ELLIPSE

        if kernel_size == 0 :
            kernel_size = 1
        kernel = cv.getStructuringElement(type_kernel, (kernel_size, kernel_size))
        # kernel = cv.getStructuringElement(type_kernel, (2 * kernel_size + 1, 2 * kernel_size + 1),
        #                                    (kernel_size, kernel_size))
        erode = cv.erode(img, kernel)
        if show == True:
            cv.imshow("window_erode", erode)

        return erode, (kernel_size, type_kernel)

    def sobel(self, img,params, show=True):
        kernel_size, delta_val, scale_val = params
        ddepth = cv.CV_16S

        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        grad_x = cv.Sobel(img, ddepth, 1, 0, ksize=kernel_size, scale=scale_val, delta=delta_val,
                          borderType=cv.BORDER_DEFAULT)
        grad_y = cv.Sobel(img, ddepth, 0, 1, ksize=kernel_size, scale=scale_val, delta=delta_val,
                          borderType=cv.BORDER_DEFAULT)
        abs_grad_x = cv.convertScaleAbs(grad_x)
        abs_grad_y = cv.convertScaleAbs(grad_y)
        grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)


        if show == True:
            cv.imshow("sobel", grad)

        return grad, (kernel_size, delta_val, scale_val)

    def barrel_distort(self, img,params, show = True):
        '''
        Threshold : setting threshold value
        :param img:
        :param show:
        :return:
        '''
        # offsetcx, offsetcy, ui_k1, ui_k2, ui_p1,ui_p2,focal_length_1,focal_length_2 = self.var_barrel_distort.return_var()
        offsetcx, offsetcy, ui_k1, ui_k2, ui_p1,ui_p2,focal_length_1,focal_length_2 = params
        width = img.shape[1]
        height = img.shape[0]
        print(width/2, height/2)

        distCoeff = np.zeros((4, 1), np.float64)

        # TODO: add your coefficients here!
        k1 = float(50 - ui_k1) * (1.0e-5)  # negative to remove barrel distortion
        k2 = float(50 - ui_k2) * (1.0e-5)
        p1 = float(50 - ui_p1) * (1.0e-5)
        p2 = float(50 - ui_p2) * (1.0e-5)


        distCoeff[0, 0] = k1;
        distCoeff[1, 0] = k2;
        distCoeff[2, 0] = p1;
        distCoeff[3, 0] = p2;

        # assume unit matrix for camera
        cam = np.eye(3, dtype=np.float32)

        cam[0, 2] = (width / 2.0)+(offsetcx-50)  # define center x
        cam[1, 2] = (height / 2.0)+(offsetcy-50)  # define center y
        cam[0, 0] = focal_length_1  # define focal length x
        cam[1, 1] = focal_length_2  # define focal length y

        # here the undistortion will be computed
        distort = cv.undistort(img, cam, distCoeff)
        if show == True:
            cv.imshow("distort", distort)

        # return distort, (ui_k1,ui_k2,ui_p1,ui_p2,cam[0, 2],cam[1, 2],cam[0, 0],cam[1, 1])
        return distort, (offsetcx, offsetcy, ui_k1, ui_k2, ui_p1,ui_p2,focal_length_1,focal_length_2)

    def crop(self, img,params,  show = True):
        '''
        Threshold : setting threshold value
        :param img:
        :param show:
        :return:
        '''
        # crop_x, crop_y = self.var_crop.return_var()
        crop_x, crop_y = params
        width = img.shape[1]
        height = img.shape[0]
        new_width_left = int((width/2)-((width/2)*(crop_x/100)))
        new_width_right = int((width/2)+((width/2)*(crop_x/100)))
        new_height_upper = int((height/2)+((height/2)*(crop_y/100)))
        new_height_lower = int((height/2)-((height/2)*(crop_y/100)))

        cropped_image = img[new_height_lower:new_height_upper , new_width_left:new_width_right]
        if show == True:
            cv.imshow("crop", cropped_image)


        # return distort, (ui_k1,ui_k2,ui_p1,ui_p2,cam[0, 2],cam[1, 2],cam[0, 0],cam[1, 1])
        return cropped_image, (crop_x, crop_y)

    def contour_area(self, img, params, show = True):
        '''
        Threshold : setting threshold value
        :param img:
        :param show:
        :return:
        '''
        if len(img.shape) == 3:  ## RGB 2 gray
            bi_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            bi_image = img

        try:
            _, contours, _ = cv.findContours(bi_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        except:
            _, contours = cv.findContours(bi_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # area_min, area_max, n, b2s = self.var_contour_area.return_var()
        area_min, area_max, n, b2s = params

        right_contours = []
        if contours is not None or contours != []:
            for contour in contours:
                try:
                    area = cv.contourArea(contour)
                    # print(area)
                except:
                    continue
                if area >= area_min and area <= area_max:
                    M = cv.moments(contour)
                    if M["m00"] == 0.0:
                        M["m00"] = 0.01
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    right_contours.append([int(area),[cX, cY], contour])


        ### sort and limit n
        draw_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        if right_contours != []:
            if b2s == 1:
                b2s_bool = True
            else:
                b2s_bool = False

            try:
                only_n_contour = sorted(right_contours, key=lambda x: x[0], reverse=b2s_bool)[0:n] # [::-1]

            except:
                only_n_contour = sorted(right_contours, key=lambda x: x[0], reverse=b2s_bool)[0:-1] # [::-1]
            print("only_n_contour",len(only_n_contour))
            for _,_, selected_contour in only_n_contour:
                cv.drawContours(draw_img, [selected_contour], -1, (255, 255, 255), -1)

        if show == True:
            cv.imshow("contours params", draw_img)
        return draw_img, (area_min, area_max, n, b2s)