import os
import shutil

path = "F:\Ph.D\circle_classification\Images_all_class\\0_1_all_class"
names = os.listdir(path)

path_for_dataset = "F:\Ph.D\circle_classification\Images_all_class\\train_data"
if not os.path.exists(path_for_dataset):
    os.mkdir(path_for_dataset)

classes = []


condition = ""
for name in names:
    classname = name.split("_")[0]
    classes.append(classname)
    path_the_class = os.path.join(path_for_dataset,classname)

    if condition in name:

        if not os.path.exists(path_the_class):
            os.mkdir(path_the_class)
            shutil.copy(os.path.join(path,name), os.path.join(path_the_class, name))
        else:
            shutil.copy(os.path.join(path,name), os.path.join(path_the_class, name))





