import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import matplotlib.pyplot as plt
import os
import pandas as pd

from scipy.spatial import distance
import time
from detect_notch import OrientationDetection
try:
    from lib.warp_and_reverse_warp import warp_polar, reverse_warp
except:
    from .lib.warp_and_reverse_warp import warp_polar, reverse_warp

import json
from easydict import EasyDict
from pathlib import Path

def preprocess( img, crop_circle, resize_ratio = 0.3):
    '''

    :param img:
    :param crop_circle:
    :return:
    '''

    img = cv2.resize(img, (int(img.shape[1] * resize_ratio), int(img.shape[0] * resize_ratio)))
    ### warp
    img, warp = warp_polar(img, crop_circle)
    reversed_warp = reverse_warp(img, warp, crop_circle)
    ### crop
    reversed_warp = reversed_warp[int(crop_circle[1] - crop_circle[2]):int(crop_circle[1] + crop_circle[2]),
                    int(crop_circle[0] - crop_circle[2]):int(crop_circle[0] + crop_circle[2])]
    return reversed_warp

class FeatureVisualization():
    def __init__(self, index=0, selected_layer=0, model = "alexnet", json_path = "config/classification.json"):


        try:
            with Path(json_path).open("r") as f:
                self.opt = json.load(f)
                self.opt = EasyDict(self.opt)
                print(self.opt)
        except:
            print(" No file {}".format(json_path))

        self.folder_ref = self.opt.folder_ref
        try:
            self.names_ref = os.listdir(folder_ref)
        except:
            print("The directory {} is not exist ".format(folder_ref))
            raise
        self.debug = self.opt.debug.lower() in ("yes", "true", "t", "1")
        self.circle_platfrom = self.opt.crop_circle_platform

        self.index = index
        # self.img_path = img_path
        self.selected_layer = selected_layer

        if model == "vgg":
            # Load pretrained model
            self.pretrained_model = models.vgg16(pretrained=True)
            # print(self.pretrained_model)
            self.pretrained_model2 = models.vgg16(pretrained=True)
        elif model == "mobilenet_v2":
            self.pretrained_model = models.mobilenet_v2(pretrained=True)
            self.pretrained_model2 = models.mobilenet_v2(pretrained=True)

        elif model == "densenet121":
            self.pretrained_model = models.densenet121(pretrained=True)
            self.pretrained_model2 = models.densenet121(pretrained=True)

        elif model == "densenet201":
            self.pretrained_model = models.densenet201(pretrained=True)
            self.pretrained_model2 = models.densenet201(pretrained=True)

        elif model == "resnext50_32x4d":
            self.pretrained_model = models.resnext50_32x4d(pretrained=True)
            self.pretrained_model2 = models.resnext50_32x4d(pretrained=True)

        elif model == "vgg19":
            self.pretrained_model = models.vgg19(pretrained=True)
            self.pretrained_model2 = models.vgg19(pretrained=True)

        elif model == "alexnet":
            self.pretrained_model = models.alexnet(pretrained=True)
            self.pretrained_model2 = models.alexnet(pretrained=True)

        elif model == "squeezenet1_1":
            self.pretrained_model = models.squeezenet1_1(pretrained=True)
            self.pretrained_model2 = models.squeezenet1_1(pretrained=True)

        elif model == "mnasnet1_0":
            self.pretrained_model = models.wide_resnet101_2(pretrained=True)
            self.pretrained_model2 = models.wide_resnet101_2(pretrained=True)


        else:
            # Load pretrained model
            self.pretrained_model = models.vgg16(pretrained=True)
            # print(self.pretrained_model)
            self.pretrained_model2 = models.vgg16(pretrained=True)
            print("vgg")

        self.cuda_is_avalible = torch.cuda.is_available()
        print(self.cuda_is_avalible)
        if self.cuda_is_avalible:
            self.pretrained_model.to(torch.device("cuda:0"))
            self.pretrained_model2.to(torch.device("cuda:0"))

    # @staticmethod
    def preprocess_image(self, cv2im, resize_im=True):

        # Resize image
        if resize_im:
            cv2im = cv2.resize(cv2im, (224, 224))
        im_as_arr = np.float32(cv2im)
        im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
        im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
        # Normalize the channels
        for channel, _ in enumerate(im_as_arr):
            im_as_arr[channel] /= 255
        # Convert to float tensor
        im_as_ten = torch.from_numpy(im_as_arr).float()
        # Add one more channel to the beginning. Tensor shape = 1,3,224,224
        im_as_ten.unsqueeze_(0)
        # Convert to Pytorch variable
        im_as_var = Variable(im_as_ten, requires_grad=False)
        if self.cuda_is_avalible:
            im_as_var = im_as_var.to(torch.device("cuda:0"))
        return im_as_var
    def set_index(self, index):
        self.index = index

    def process_image(self, img):
        # print('input image:')
        img = self.preprocess_image(img)
        return img

    def get_feature(self,img):
        # Image  preprocessing
        input = self.process_image(img)
        # print("input.shape:{}".format(input.shape))
        x = input
        self.pretrained_model.eval()
        with torch.no_grad():
            for index, layer in enumerate(self.pretrained_model):
                x = layer(x)
                #             print("{}:{}".format(index,x.shape))
                if (index == self.selected_layer):
                    return x

    def get_conv_feature(self,img):
        # Get the feature map
        features = self.get_feature(img)
        result_path = './feat_' + str(self.selected_layer)

        if not os.path.exists(result_path):
            os.makedirs(result_path)

    def plot_probablity(self, outputs):
        outputs = outputs.cpu()
        outputs = outputs.data.numpy()
        # print(outputs.shape)
        outputs = np.ndarray.tolist(outputs)
        # print(type(outputs),outputs)
        # print(len(outputs[0]))
        # x = range(0, 4096)

        # plt.bar(x, outputs[0])
        # plt.xlabel("Dimension")
        # plt.ylabel("Value")
        # plt.title("FC feature {}".format(str(self.index)))
        # plt.show()

    def get_fc_feature(self,img):
        input = self.process_image(img)
        self.pretrained_model2.eval()
        # self.pretrained_model2.classifier = nn.Sequential(*list(self.pretrained_model2.classifier.children())[0:4])
        with torch.no_grad():
            outputs = self.pretrained_model2(input)
        # self.plot_probablity(outputs)
        return outputs

    def compare_cosine(self, out1, out2):
        metric = 'cosine'
        out1 = out1.cpu()
        out2 = out2.cpu()
        cosineDistance = distance.cdist(out1, out2, metric)[0]
        return cosineDistance


    def get_similar_img(self, img1):
        if self.debug:
            cv2.putText(img1, "INPUT", (0, img1.shape[0] - 10), cv2.FONT_HERSHEY_COMPLEX, 3, (200, 200, 0), 3)
            cv2.imshow("1", img1)
        # imgasvar = featureVis.preprocess_image(img1)
        outputs1 = featureVis.get_fc_feature(img1)
        featureVis.plot_probablity(outputs1)

        for j in range(len(self.names_ref)):
            img2 = cv2.imread(os.path.join(self.opt.folder_ref, self.names_ref[j]))

            # imgasvar = featureVis.preprocess_image(img2)
            featureVis.set_index(j)
            outputs2 = featureVis.get_fc_feature(img2)
            featureVis.plot_probablity(outputs2)

            dis = featureVis.compare_cosine(outputs1, outputs2)
            cv2.waitKey(1)

            result.append(dis[0])

        result_array = np.asarray(result)
        ind = np.argmin(result_array)
        class_obj = names_ref[ind].split("_")[0]
        print("The class is {}".format(class_obj))
        # print(os.path.join(self.folder_ref, self.names_ref[ind]))
        answer = cv2.imread(os.path.join(self.folder_ref, self.names_ref[ind]))
        if self.debug:
            cv2.putText(answer, "ANSWER {}".format(str(class_obj)), (0, answer.shape[0] - 10), cv2.FONT_HERSHEY_COMPLEX, 3.5, (50, 0, 200), 3)
            cv2.imshow("answer_class", answer)
            cv2.waitKeyEx(0)

        return class_obj



if __name__ == '__main__':
    folder = "F:\Pawat\Projects\Imageprocessing_Vistools\data\container\image\Darker - Exposure time 120000us close some ambient light"
    folder_ref = "F:\Pawat\Projects\Imageprocessing_Vistools\data\container\\light2_class"
    folder_ref = "F:\Pawat\Projects\container-orientation-detection\dataset\class_registeration"

    names = os.listdir(folder)
    names_ref = os.listdir(folder_ref)
    i = 0
    j = 0


    ### fill the other object
    orientation_detection_A = OrientationDetection( path=os.path.join(folder, names[i]), json_path="config/notch_config_A.json")
    ## EX
    orientation_detection_B = OrientationDetection( path=os.path.join(folder, names[i]), json_path="config/notch_config_B.json")



    def resize_scale(img, scale=0.3):
        resize = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
        return resize


    featureVis = FeatureVisualization()
    all_result = []
    cv2.namedWindow("1",cv2.WINDOW_NORMAL)
    cv2.namedWindow("answer_class",cv2.WINDOW_NORMAL)
    for i in range(len(names)):
        result = []
        img1 = cv2.imread(os.path.join(folder, names[i]))
        # img1 = preprocess(img1, orientation_detection_A.crop_circle_platform)
        # img1 = cv2.rotate(img1,cv2.ROTATE_90_COUNTERCLOCKWISE)

        name_class = featureVis.get_similar_img(img1)

        if name_class == "A":
            print(img1.shape)
            angle = orientation_detection_A.detect(img1)
            if angle is not None:
                result = cv2.putText(img1, "ANGLE: {}".format(str("%.2f" % round(angle, 2))),
                                     (0, img1.shape[0] - 10), cv2.FONT_HERSHEY_COMPLEX, 8, (0, 50, 255), 8)

                cv2.namedWindow("RESULT", cv2.WINDOW_NORMAL)
                cv2.imshow("RESULT", result)

        ### Example
        elif name_class == "B":
            angle = orientation_detection_B.detect(img1)
            if angle is not None:
                result = cv2.putText(img1, "ANGLE: {}".format(str("%.2f" % round(angle, 2))),
                                     (0, img1.shape[0] - 10), cv2.FONT_HERSHEY_COMPLEX, 8, (0, 50, 255), 8)

                cv2.namedWindow("RESULT", cv2.WINDOW_NORMAL)
                cv2.imshow("RESULT", result)

        all_result.append(result)


    # print(all_result)
    # df = pd.DataFrame(all_result, columns= names_ref)
    # print(df)
    # df.to_csv("result.csv")
