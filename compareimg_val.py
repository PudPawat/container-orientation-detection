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
    def __init__(self, index=0, selected_layer=0, model = "vgg", json_path = "config/classification.json", features_path = ""):


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
        self.crop_circle = self.opt["crop_circle"].lower() in ["t", "true", "True"]

        self.index = index
        # self.img_path = img_path
        self.selected_layer = selected_layer
        self.modelnames = ["vgg16","mobilenet_v2","densenet121","densenet121","densenet201","resnext50_32x4d","vgg19","alexnet","squeezenet1_1","mnasnet1_0"]
        if model == "vgg16":
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


        ## initialize the data
        try:
            # Read the JSON file
            with open(features_path, "r") as file:
                json_data = file.read()
            # Parse the JSON data into a dictionary
            self.names_result = json.loads(json_data)
            self.names_ref = list(self.names_result.keys())
            print("READ feature file successfully ")


        except:

            self.get_imgs_in_ref_score()
            features_result_path = f"./config/features_result_{model}.json"
            print(self.names_result_save)
            with open(features_result_path, 'w') as file:
                json.dump(self.names_result_save, file)

            print("Dictionary saved as JSON successfully.")


    def get_imgs_in_ref_score(self):
        names_result = {}
        names_result_save = {}
        print("self.names_ref",self.names_ref)

        key = 0
        for j in range(len(self.names_ref)):
            img2 = cv2.imread(os.path.join(self.opt.folder_ref, self.names_ref[j]))

            if self.crop_circle is True:

                img2 = preprocess(img2, self.circle_platfrom, resize_ratio= 0.3)

            # cv2.imshow("see_crop", img2)
            # key = cv2.waitKey(int(key))
            # print("0")
            # imgasvar = featureVis.preprocess_image(img2)
            self.set_index(j)
            outputs2 = self.get_fc_feature(img2)
            # self.plot_probablity(outputs2)

            # names_result[self.names_ref[j]] = outputs2
            names_result_save[self.names_ref[j]] = outputs2.cpu().tolist()

        self.names_result_save = names_result_save
        self.names_result = names_result_save

        return names_result

    # @staticmethod
    def preprocess_image(self, cv2im, resize_im=True):
        # print(len(cv2im))
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

    def compare_cosine(self, out1, out2, metric = None):

        '''
        metric = 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
        'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon',
        'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
        'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
        'sqeuclidean', 'wminkowski', 'yule'.
        :param out1:
        :param out2:
        :return:
        '''
        metric_list = ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
        'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon',
        'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
        'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
        'sqeuclidean', 'wminkowski', 'yule']
        if metric not in metric_list:
            metric = 'cosine'

        try:
            out1 = out1.cpu()
        except:
            out1 = out1

        try:
            out2 = out2.cpu()
        except:
            out2 = out2
        cosineDistance = distance.cdist(out1, out2, metric)[0]
        return cosineDistance

    def compare_cosine_all(self, out1, out2):
        metric = 'cosine'
        out1 = out1.cpu()
        out2 = out2.cpu()
        cosineDistance = distance.cdist(out1, out2, metric)
        return cosineDistance


    def get_similar_img(self, img1, metric = ""):
        '''
        to get similar image
        :param img1:
        :return:
        '''
        if self.debug:
            cv2.putText(img1, "INPUT", (0, img1.shape[0] - 10), cv2.FONT_HERSHEY_COMPLEX, 3, (200, 200, 0), 3)
            cv2.imshow("1", img1)
        # imgasvar = self.preprocess_image(img1)
        outputs1 = self.get_fc_feature(img1)
        # print("outputs1", outputs1)
        self.plot_probablity(outputs1)
        # print("outputs1", outputs1)
        result = []
        for j, name in enumerate(self.names_result.keys()):
            dis = self.compare_cosine(outputs1, self.names_result[name],metric = metric)
            result.append(dis[0])
            dis_all = self.compare_cosine(outputs1, self.names_result[name],metric = metric )
            print(name, "dis_all: ", dis_all)

        result_array = np.asarray(result)
        ind = np.argmin(result_array)
        print(self.names_ref)
        print(ind)
        try:
            class_obj = self.names_ref[ind].split("_")[0]
        except:
            class_obj = self.names_ref[ind]
        print("The class is {}".format(class_obj))
        # print(os.path.join(self.folder_ref, self.names_ref[ind]))
        answer = cv2.imread(os.path.join(self.folder_ref, self.names_ref[ind]))
        if self.debug:
            cv2.putText(answer, "ANSWER {}".format(str(class_obj)), (0, answer.shape[0] - 10), cv2.FONT_HERSHEY_COMPLEX, 3.5, (50, 0, 200), 3)
            cv2.imshow("answer_class", answer)
            cv2.waitKey(0)

        return class_obj


    def get_similar_n_img(self, img1, metric = "", get_n_imge = 5):
        '''
        to get similar image
        :param img1:
        :return:
        '''
        if self.debug:
            cv2.putText(img1, "INPUT", (0, img1.shape[0] - 10), cv2.FONT_HERSHEY_COMPLEX, 3, (200, 200, 0), 3)
            cv2.imshow("1", img1)
        # imgasvar = self.preprocess_image(img1)
        outputs1 = self.get_fc_feature(img1)
        # print("outputs1", outputs1)
        self.plot_probablity(outputs1)
        # print("outputs1", outputs1)
        result = []
        for j, name in enumerate(self.names_result.keys()):
            dis = self.compare_cosine(outputs1, self.names_result[name],metric = metric)
            result.append(dis[0])
            dis_all = self.compare_cosine(outputs1, self.names_result[name],metric = metric )
            print(name, "dis_all: ", dis_all)

        result_array = np.asarray(result)

        n_result = []
        n_ind_result = []

        for n_result_index in range(get_n_imge):

            ind = np.argmin(result_array)
            n_ind_result.append(ind)

            try:
                n_result.append(self.names_ref[ind].split("_")[0])
            except:
                n_result.append(self.names_ref[ind])

            result_array[ind] = 999

        print("The class is {}".format(n_result[-1]))
        # print(os.path.join(self.folder_ref, self.names_ref[ind]))
        answer = cv2.imread(os.path.join(self.folder_ref, self.names_ref[n_ind_result[0]]))
        if self.debug:
            cv2.putText(answer, "ANSWER {}".format(str(n_result[-1])), (0, answer.shape[0] - 10), cv2.FONT_HERSHEY_COMPLEX, 3.5, (50, 0, 200), 3)
            cv2.imshow("answer_class", answer)
            cv2.waitKey(0)

        return n_result



if __name__ == '__main__':
    folder = "F:\Pawat\Projects\Imageprocessing_Vistools\data\container\image\Darker - Exposure time 120000us close some ambient light"
    folder = "F:\Ph.D\circle_classification\Images_all_class\\0_all_class"
    folder = "dataset\class_registeration"
    folder = "dataset\\20230311"
    folder_ref = "F:\Pawat\Projects\Imageprocessing_Vistools\data\container\\light2_class"
    folder_ref = "F:\Ph.D\circle_classification\Images_all_class\\0_all_class_aug"
    folder_ref = "dataset\class_registeration"
    folder_ref = "dataset\\20230311"

    names = os.listdir(folder)
    names_ref = os.listdir(folder_ref)
    i = 0
    j = 0


    ### fill the other object
    # orientation_detection_A = OrientationDetection( path=os.path.join(folder, names[i]), json_path="config/notch_config_A.json")
    # ## EX
    # orientation_detection_B = OrientationDetection( path=os.path.join(folder, names[i]), json_path="config/notch_config_B.json")



    def resize_scale(img, scale=0.3):
        resize = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
        return resize

    def test_model_and_acc(model = "" ,metric = "", crop_circle = []):
        fea_path = f"config/features_result_{model}.json"
        featureVis = FeatureVisualization(model = model, features_path= fea_path)
        all_result = []
        cv2.namedWindow("1",cv2.WINDOW_NORMAL)
        cv2.namedWindow("answer_class",cv2.WINDOW_NORMAL)
        val_result = []
        for i in range(len(names)):
            result = []
            img1 = cv2.imread(os.path.join(folder, names[i]))
            img1 = preprocess(img1, featureVis.circle_platfrom)
            # img1 = preprocess(img1, orientation_detection_A.crop_circle_platform)
            # img1 = cv2.rotate(img1,cv2.ROTATE_90_COUNTERCLOCKWISE)
            name_class = featureVis.get_similar_img(img1, metric= metric)
            name_classes = featureVis.get_similar_n_img(img1, metric =  metric , get_n_imge=5)
            print(name_classes)
            if name_class in names[i]:
                val_result.append(True)
            else:
                val_result.append(False)
            all_result.append(result)
        count_true = val_result.count(True)
        acc = (count_true/ len(val_result)) *100
        print(str(acc) + "% accuracy")
        return  acc

    def test_an_image(img, model = None):

        fea_path = f"config/features_result_{model}.json"
        featureVis = FeatureVisualization(model = model, features_path= fea_path)

        if type(img) == "String":
            img1 = cv2.imread(img)
        else:
            img1 = img
        img1 = preprocess(img1, featureVis.circle_platfrom)
        name_classes = featureVis.get_similar_n_img(img1, metric=metric, get_n_imge= 5)

        print(name_classes)

    modelnames = ["mobilenet_v2", "densenet121", "densenet121", "densenet201", "resnext50_32x4d",
                       "vgg19", "alexnet", "squeezenet1_1", "mnasnet1_0"]
    ''' metric = 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
        'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon',
        'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
        'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
        'sqeuclidean', 'wminkowski', 'yule'.'''
    metric = "cosine"
    metrics = ["cosine", "euclidean",'sqeuclidean','seuclidean','matching']

    model_result = {}
    featureVis = FeatureVisualization(features_path="config/features_result.json")
    model_result["dataset"] = featureVis.folder_ref


    with open("config/classification.json", "r") as file:
        json_data_classification = file.read()
    model_result["config"] = json.loads(json_data_classification)


    print(model_result)
    for metric in metrics:
        for modelname in modelnames:
            print("model: ", modelname)
            acc = test_model_and_acc(model = modelname, metric = metric)
            model_result[modelname] = acc

            print(model_result)



        with open(f"model_result_{metric}.json", "w") as file:
            json.dump(model_result, file)
        print("READ feature file successfully ")
