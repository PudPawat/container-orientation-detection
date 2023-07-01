import cv2
import os
import shutil

if __name__ == '__main__':
    path = "F:\Ph.D\circle_classification\Images_all_class\Images"
    suffix = "_2"
    save_path = "F:\Ph.D\circle_classification\Images_all_class\\0_all_class"

    if not os.path.exists(save_path):
        print("mkdir")
        os.mkdir(save_path)

    names = os.listdir(path)

    for name in names:
        file_name = os.path.basename(os.path.join(path, name))
        file_name_without_extension = os.path.splitext(file_name)[0]

        if suffix in file_name_without_extension:
            print(file_name_without_extension)
            shutil.copy(os.path.join(path,name), os.path.join(save_path, name))

