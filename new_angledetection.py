import os
import numpy as np
import random


import json
from easydict import EasyDict
from pathlib import Path
from copy import deepcopy
from PIL import Image
from lib.contour_after_process import contour_area, contour_center_dis,contour_center_X_or_Y,contour_big2small_n_order
from lib.warp_and_reverse_warp import warp_polar, reverse_warp
from lib_save.read_params import *
from lib.compare_img_module import FeatureVisualizationModule
from utils import crop_circle, rotate_image, crop_circle_by_warp, fill_balck_circle

### simple_tiny
from lib.custom_circle_detection import fit_circle_2d, get_x_y_from_contour


def resize_scale(img, scale = 0.3):
    resize = cv2.resize(img,(int(img.shape[1]*scale),int(img.shape[0]*scale)))
    return resize


class OrientationDetectionv2():
    def __init__(self, path_ref = "", json_path = ""):
        try:
            with Path(json_path).open("r") as f:
                self.opt = json.load(f)
                self.opt = EasyDict(self.opt)
                print(self.opt)
        except:
            print(" No file {}".format(json_path))

        self.params = self.opt.params
        self.debug =  self.opt.debug.lower() in ("yes", "true", "t", "1")
        self.save_img = self.opt.save_img.lower() in ("yes", "true", "t", "1")
        self.process_order = 0
        self.show_result = self.opt.show_result.lower() in ("yes", "true", "t", "1")
        self.flag_rotate = self.opt.flag_rotate
        self.circle_inner_r = self.opt.crop_circle_fix_inner_r
        self.circle_outer_r = self.opt.crop_circle_fix_outer_r
        self.crop_circle_platform = self.opt.crop_circle_platform
        self.resize_ratio = self.opt.resize_ratio
        self.threshold = self.opt.inverse_threshold.lower() in ("yes", "true", "t", "1")
        self.n_symetric = self.opt.n_symetric
        self.read = read_save()
        self.path_ref = path_ref
        self.ref_names = os.listdir(self.path_ref)
        print(self.ref_names)

        self.FeatureCompare = FeatureVisualizationModule()

    def find_name_in_list(self, name):
        '''
        TO find ref name in the directory
        :param name:
        :return:
        '''
        result = None
        for ref_name in self.ref_names:
            if name == ref_name or (name in ref_name) or (ref_name in name):
                return ref_name

        return result


    def linear_process(self, img):
        '''
        to crop circle and warp polar for only the radius between inner_r and outter_r
        :param img:
        :return:
        '''
        result, _, _ = self.read.read_params(self.params,img)
        img_result = result["final"]
        cv2.imshow("result",img_result)
        img_result = cv2.resize(img_result, (int(img.shape[1] * self.resize_ratio), int(img.shape[0] * self.resize_ratio)))
        cv2.waitKey(0)
        img_linear_crop = self.crop_roi(img_result)
        return img_linear_crop

    def crop_roi(self, img):
        '''
        crop only area in between inner_r and outer_r
        :param img:
        :return:
        '''
        img, warp = warp_polar(img, self.circle_outer_r)
        img_linear_crop = warp[0: warp.shape[0], int(self.circle_inner_r[-1]):warp.shape[1]]
        if self.debug:
            cv2.imshow("warp_outer", warp)
            cv2.imshow("warp_outer_crop", img_linear_crop)
            if self.save_img:
                cv2.imwrite("debug_imgs/{}warp_outer.jpg".format(self.process_order), warp)
                cv2.imwrite("debug_imgs/{}warp_outer_crop.jpg".format(self.process_order), img_linear_crop)
                self.process_order += 1
        return img_linear_crop

    def compare_by_rotate_angle(self, refimg_crop_outer_r, img_crop_outer_r,times_rot = 1, max_rot_angle = 360):
        '''
        to compare ny iterating rotating the image and compare the score
        :param refimg_crop_outer_r:
        :param img_crop_outer_r:
        :param times_rot: time devided by max_rot_angle
        :param max_rot_angle: 360, 180, 120
        :return:
        '''
        fc_ref_img = self.FeatureCompare.get_fc_feature(refimg_crop_outer_r)
        # cv2.waitKey(0)
        scores = []
        n = times_rot  # try 360* n times if n = 2, try every 0.5 degree
        for i in range(int(times_rot * max_rot_angle)):
            test_rotate_img = rotate_image(img_crop_outer_r, i / (n))
            # cv2.imshow("test_rotate", test_rotate_img)
            fc_rotate = self.FeatureCompare.get_fc_feature(test_rotate_img)
            score = self.FeatureCompare.compare_cosine(fc_ref_img, fc_rotate)
            scores.append(score)
            # break
            # print(score)
            # cv2.waitKey(0)

        np_scores = np.asarray(scores)
        i_min = np.argmin(np_scores)

        print(i_min)

        angle = i_min/times_rot

        while (angle > 360 / self.n_symetric):
            angle = angle - (360 / self.n_symetric)

        return angle

    def compare_by_rotate_angle_minus(self, refimg_crop_outer_r, img_crop_outer_r,times_rot = 1, max_rot_angle = 360):
        '''
        to compare ny iterating rotating the image and compare the score
        :param refimg_crop_outer_r:
        :param img_crop_outer_r:
        :param times_rot: time devided by max_rot_angle
        :param max_rot_angle: 360, 180, 120
        :return:
        '''
        # fc_ref_img = self.FeatureCompare.get_fc_feature(refimg_crop_outer_r)
        # cv2.waitKey(0)
        scores = []
        n = times_rot  # try 360* n times if n = 2, try every 0.5 degree
        for i in range(int(times_rot * max_rot_angle)):
            test_rotate_img = rotate_image(img_crop_outer_r, i / (n))
            # cv2.imshow("test_rotate", test_rotate_img)
            # fc_rotate = self.FeatureCompare.get_fc_feature(test_rotate_img)
            # score = self.FeatureCompare.compare_cosine(fc_ref_img, fc_rotate)
            try:
                h, w, _ = test_rotate_img.shape
            except:
                h, w  = test_rotate_img.shape
            diff = cv2.subtract(refimg_crop_outer_r, test_rotate_img)
            err = np.sum(diff ** 2)
            score = err / (float(h * w))
            # print(score)
            # print(sum(score))
            scores.append(score)
            # break
            # print(score)
            # cv2.waitKey(0)

        np_scores = np.asarray(scores)
        i_min = np.argmin(np_scores)

        print(i_min)

        angle = i_min/times_rot

        # while (angle > 360 / self.n_symetric):
        #     angle = angle - (360 / self.n_symetric)

        return angle

    def process_param_img(self, img):
        '''
        Processs img by default config
        :param img:
        :return:
        '''
        result, _, _ = self.read.read_params(self.params, img)
        img_result = result["final"]
        return img_result

    def main_simple(self,img, class_name):
        '''

        :param img:
        :return:
        '''
        img_crop_platform = crop_circle(img, self.crop_circle_platform)
        result = deepcopy(img_crop_platform)
        linear_bi_img = self.linear_process(img)
        if len(linear_bi_img.shape) == 2:
            linear_bi_img_BGR = cv2.cvtColor(linear_bi_img,cv2.COLOR_GRAY2BGR)

        if self.threshold:
            _, linear_bi_img = cv2.threshold(linear_bi_img,0,255,cv2.THRESH_BINARY_INV)
        area_contours = contour_big2small_n_order(linear_bi_img,self.n_symetric)
        # area_contours = area_contours
        if len(area_contours) != 0:
            ### detect 1st biggest contour
            area, center, _ = area_contours[0]
            length360 = linear_bi_img.shape[0]
            angle = abs(center[1] )/length360 * 360
            print(angle)
            # if angle > 360/self.n_symetric:
            while (angle > 360/self.n_symetric):
                angle = angle - (360/self.n_symetric)
            cv2.putText(result, str(angle),(0, img_crop_platform.shape[0]),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,0),2)
            print(angle)
            cv2.imshow("result_1",result)
            if self.debug:
                for _,_, contour in area_contours:
                    cv2.drawContours(linear_bi_img_BGR,[contour],-1, (255,0,0),3)
                cv2.imshow("contour", linear_bi_img_BGR)
                cv2.waitKey(0)
                if self.save_img:
                    cv2.imwrite("debug_imgs/{}_{}contour.jpg".format(class_name,self.process_order), linear_bi_img_BGR)
                    cv2.imwrite("debug_imgs/{}_{}result_img_crop_platform.jpg".format(class_name,self.process_order), img_crop_platform)
                    cv2.imwrite("debug_imgs/{}_{}result.jpg".format(class_name,self.process_order), result)
                    self.process_order +=1
        cv2.waitKey(0)

    def main_simple_for_tiny(self,img, class_name):
        '''

        :param img:
        :return:
        '''

        img_crop_platform = crop_circle(img, self.crop_circle_platform)
        result = deepcopy(img_crop_platform)

        result_proc, _, _ = self.read.read_params(self.params, img)
        img_result = result_proc["final"]
        cv2.imshow("img_result", img_result)
        cv2.waitKey(0)

        a_contour = contour_big2small_n_order(img_result, 1)
        x,y = get_x_y_from_contour(a_contour[0][2])
        xc, yc, r, loss = fit_circle_2d(x,y)
        try:
            r = r - abs(self.opt["simple_tiny"]["outer_r_safety"])
        except:
            r = r - 5
        img, linear_bi_img = warp_polar(img_result, (xc, yc, r))
        if self.threshold:
            _, linear_bi_img = cv2.threshold(linear_bi_img, 0, 255, cv2.THRESH_BINARY_INV)

        cv2.imshow("test", linear_bi_img)
        cv2.waitKey(0)
        area_contours = contour_big2small_n_order(linear_bi_img, 3)

        if len(area_contours) != 0:
            ### detect 1st biggest contour
            area, center, _ = area_contours[0]
            length360 = linear_bi_img.shape[0]
            angle = abs(center[1] )/length360 * 360
            print(angle)
            # if angle > 360/self.n_symetric:
            while (angle > 360/self.n_symetric):
                angle = angle - (360/self.n_symetric)
            cv2.putText(result, str(angle),(0, img_crop_platform.shape[0]),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,0),2)
            print(angle)
            cv2.imshow("result_1",result)
            if self.debug:
                if len(linear_bi_img.shape) == 2:
                    linear_bi_img_BGR = cv2.cvtColor(linear_bi_img, cv2.COLOR_GRAY2BGR)
                for _,_, contour in area_contours:
                    cv2.drawContours(linear_bi_img_BGR,[contour],-1, (255,0,0),3)
                cv2.imshow("contour", linear_bi_img_BGR)
                cv2.waitKey(0)
                if self.save_img:
                    cv2.imwrite("debug_imgs/{}_{}contour.jpg".format(class_name,self.process_order), linear_bi_img_BGR)
                    cv2.imwrite("debug_imgs/{}_{}result_img_crop_platform.jpg".format(class_name,self.process_order), img_crop_platform)
                    cv2.imwrite("debug_imgs/{}_{}result.jpg".format(class_name,self.process_order), result)
                    self.process_order +=1



        # try:
        #     _, contours, _ = cv2.findContours(img_result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # except:
        #     _, contours = cv2.findContours(img_result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)




    def main_compare(self,img, class_name):
        '''
        under develop
        :param img:
        :param class_name:
        :return:
        '''
        img_crop_platform = crop_circle(img, self.crop_circle_platform)
        img = cv2.rotate(img, cv2.ROTATE_180)
        ## read ref
        print(class_name)
        file_ref_name = self.find_name_in_list(class_name)
        file_ref_name_platform = crop_circle(file_ref_name, self.crop_circle_platform)
        ref_img = cv2.imread(os.path.join(self.path_ref, file_ref_name))
        ref_linear_img = self.linear_process(ref_img)
        linear_img = self.linear_process(img)
        # cv2.imshow("ref", ref_img)
        if self.debug:
            cv2.imshow("ref_linear", ref_linear_img)
            cv2.imshow("linear", linear_img)
            cv2.waitKey(0)
            if self.save_img:
                cv2.imwrite("debug_imgs/{}ref_linear.jpg".format(self.process_order), ref_linear_img)
                cv2.imwrite("debug_imgs/{}linear.jpg".format(self.process_order), linear_img)
                self.process_order += 1

    def main_compare1(self, img, class_name):
        '''
        raw image and rotating to compare img and find lowest loss on the rotated img to ref_img under develop
        :param img:
        :param class_name:
        :return:
        '''

        # img = cv2.rotate(img, cv2.ROTATE_180)

        ## read ref
        print(class_name)
        file_ref_name = self.find_name_in_list(class_name)
        print(file_ref_name)
        ref_img  = cv2.imread(os.path.join(self.path_ref, file_ref_name))
        resize_refimg = resize_scale(ref_img)
        rotate_refimg_crop_c = crop_circle_by_warp(resize_refimg, self.circle_outer_r)
        rotate_refimg_crop_c = fill_balck_circle(rotate_refimg_crop_c, self.circle_inner_r)
        refimg_crop_outer_r = crop_circle(rotate_refimg_crop_c, self.circle_outer_r, False)
        # print(file_ref_name)
        cv2.imshow("ref", refimg_crop_outer_r)

        resize_img = resize_scale(img)
        ## test rotate
        rotate_img_crop_c = crop_circle_by_warp(resize_img, self.circle_outer_r)
        rotate_img_crop_c = fill_balck_circle(rotate_img_crop_c, self.circle_inner_r)
        img_crop_outer_r = crop_circle(rotate_img_crop_c, self.circle_outer_r, False)
        img_crop_outer_r = rotate_image(img_crop_outer_r, 90)
        cv2.imshow("rotate", img_crop_outer_r)
        if self.debug:
            cv2.imshow("refimg_crop_outer_r.jpg".format(self.process_order),refimg_crop_outer_r)
            cv2.imshow("img_crop_outer_r.jpg".format(self.process_order),img_crop_outer_r)
            if self.save_img:
                cv2.imwrite("debug_imgs/{}img_crop_outer_r.jpg".format(self.process_order),img_crop_outer_r)
                cv2.imwrite("debug_imgs/{}refimg_crop_outer_r.jpg".format(self.process_order),refimg_crop_outer_r)

        angle = self.compare_by_rotate_angle(refimg_crop_outer_r,img_crop_outer_r)
        print(angle)

    def main_compare_with_process(self, img, class_name):
        '''
        process image and rotating to compare img and find lowest loss on the rotated img to ref_img
        :param img:
        :param class_name:
        :return:
        '''

        # img = cv2.rotate(img, cv2.ROTATE_180)

        ## read ref
        print(class_name)
        file_ref_name = self.find_name_in_list(class_name)
        print(file_ref_name)
        ref_img = cv2.imread(os.path.join(self.path_ref, file_ref_name))
        ## cv process
        ref_img = self.process_param_img(ref_img)
        resize_refimg = resize_scale(ref_img)
        rotate_refimg_crop_c = crop_circle_by_warp(resize_refimg, self.circle_outer_r)
        rotate_refimg_crop_c = fill_balck_circle(rotate_refimg_crop_c, self.circle_inner_r)
        refimg_crop_outer_r = crop_circle(rotate_refimg_crop_c, self.circle_outer_r, False)
        # print(file_ref_name)
        cv2.imshow("ref", refimg_crop_outer_r)

        img = self.process_param_img(img)
        resize_img = resize_scale(img)

        ## test rotate
        rotate_img_crop_c = crop_circle_by_warp(resize_img, self.circle_outer_r)
        rotate_img_crop_c = fill_balck_circle(rotate_img_crop_c, self.circle_inner_r)
        img_crop_outer_r = crop_circle(rotate_img_crop_c, self.circle_outer_r, False)
        img_crop_outer_r = rotate_image(img_crop_outer_r, 90)
        cv2.imshow("rotate", img_crop_outer_r)
        cv2.waitKey(1)

        if self.debug:
            cv2.imshow("refimg_crop_outer_r.jpg".format(self.process_order),refimg_crop_outer_r)
            cv2.imshow("img_crop_outer_r.jpg".format(self.process_order),img_crop_outer_r)
            if self.save_img:
                cv2.imwrite("debug_imgs/{}img_crop_outer_r.jpg".format(self.process_order),img_crop_outer_r)
                cv2.imwrite("debug_imgs/{}refimg_crop_outer_r.jpg".format(self.process_order),refimg_crop_outer_r)

        ## process
        img_crop_outer_r_process = self.process_param_img(img_crop_outer_r)
        refimg_crop_outer_r_process = self.process_param_img(refimg_crop_outer_r)

        # if self.debug:
        #     cv2.imshow("refimg_crop_outer_r_process.jpg".format(self.process_order),refimg_crop_outer_r_process)
        #     cv2.imshow("img_crop_outer_r_process.jpg".format(self.process_order),refimg_crop_outer_r_process)
        #     if self.save_img:
        #         cv2.imwrite("debug_imgs/{}img_crop_outer_r_process.jpg".format(self.process_order),refimg_crop_outer_r_process)
        #         cv2.imwrite("debug_imgs/{}refimg_crop_outer_r_process.jpg".format(self.process_order),refimg_crop_outer_r_process)
        #
        # if len(img_crop_outer_r_process.shape) == 2:
        #     img_crop_outer_r_process = cv2.cvtColor(img_crop_outer_r_process, cv2.COLOR_GRAY2BGR)
        # if len(refimg_crop_outer_r_process.shape) == 2:
        #     refimg_crop_outer_r_process = cv2.cvtColor(refimg_crop_outer_r_process, cv2.COLOR_GRAY2BGR)
        cv2.waitKey(1)
        if len(refimg_crop_outer_r.shape) == 2:
            refimg_crop_outer_r = cv2.cvtColor(refimg_crop_outer_r, cv2.COLOR_GRAY2BGR)
        if len(img_crop_outer_r.shape) == 2:
            img_crop_outer_r = cv2.cvtColor(img_crop_outer_r, cv2.COLOR_GRAY2BGR)
        try:
            # refimg_crop_outer_r_process = rotate_image(refimg_crop_outer_r_process, 45)
            angle = self.compare_by_rotate_angle_minus(refimg_crop_outer_r_process, img_crop_outer_r_process,1,int(360/self.n_symetric))
            # angle = self.compare_by_rotate_angle(refimg_crop_outer_r_process, img_crop_outer_r_process,2,int(360/self.n_symetric))
        except:
            # refimg_crop_outer_r = rotate_image(refimg_crop_outer_r, 45)
            angle = self.compare_by_rotate_angle_minus(refimg_crop_outer_r, img_crop_outer_r,1,int(360/self.n_symetric))
            # angle = self.compare_by_rotate_angle(refimg_crop_outer_r, img_crop_outer_r,2,int(360/self.n_symetric))
        print(angle)
        cv2.waitKey(0)


    def preprocess(self, img, crop_circle):
        '''

        :param img:
        :param crop_circle:
        :return:
        '''

        img = cv2.resize(img, (int(img.shape[1] * self.resize_ratio), int(img.shape[0] * self.resize_ratio)))
        ### warp
        img, warp = warp_polar(img, crop_circle)
        reversed_warp = reverse_warp(img, warp, crop_circle)
        ### crop
        reversed_warp = reversed_warp[int(crop_circle[1] - crop_circle[2]):int(crop_circle[1] + crop_circle[2]),
                        int(crop_circle[0] - crop_circle[2]):int(crop_circle[0] + crop_circle[2])]
        return reversed_warp


if __name__ == '__main__':
    path_imgs = "dataset/20230318"
    names = os.listdir(path_imgs)

    for name in names:
        img_path = os.path.join(path_imgs,name)
        print("detect img name", name)
        img = cv2.imread(img_path)

        class_name = name.split("_")[0]
        detect = OrientationDetectionv2("dataset/20230318",json_path = "config/notchv2_config_{}.json".format(class_name))
        # detect.main_compare(img, class_name)
        # detect.main_simple(img, class_name)
        detect.main_simple_for_tiny(img, class_name)
        # detect.main_compare1(img, class_nam         detect.main_compare_with_process(img, class_name)
        # cv2.waitKey(0)