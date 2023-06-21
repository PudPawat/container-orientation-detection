import cv2
import numpy as np

def adjust_brightness(image, value):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v = np.clip(v, 0, 255)
    hsv = cv2.merge((h, s, v))
    adjusted_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return adjusted_image

def adjust_contrast(image, value):
    alpha = float(value) / 127.0
    adjusted_image = cv2.addWeighted(image, alpha, image, 0, 0)
    return adjusted_image

def adjust_shadows(image, value):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.subtract(l, value)
    l = np.clip(l, 0, 255)
    lab = cv2.merge((l, a, b))
    adjusted_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return adjusted_image

# Load the image
# image = cv2.imread('path_to_your_image.jpg')
#
# # Adjust brightness
# brightness_value = 50  # Increase or decrease brightness (-255 to 255)
# brightness_adjusted = adjust_brightness(image, brightness_value)
#
# # Adjust contrast
# contrast_value = 1.5  # Increase or decrease contrast (0.0 to 3.0)
# contrast_adjusted = adjust_contrast(image, contrast_value)
#
# # Adjust shadows
# shadow_value = 50  # Increase or decrease shadows (-255 to 255)
# shadow_adjusted = adjust_shadows(image, shadow_value)
#
# # Display the original and adjusted images
# cv2.imshow('Original Image', image)
# cv2.imshow('Brightness Adjusted Image', brightness_adjusted)
# cv2.imshow('Contrast Adjusted Image', contrast_adjusted)
# cv2.imshow('Shadow Adjusted Image', shadow_adjusted)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


if __name__ == '__main__':

    import os
    path = "F:\Ph.D\circle_classification\Images_all_class\\0_all_class"




    save_path = "F:\Ph.D\circle_classification\Images_all_class\\0_all_class_aug"
    if not os.path.exists(save_path):
        print("mkdir")
        os.mkdir(save_path)


    names = os.listdir(path)

    methods = {
        "brightness_values" : [-10,40, 80],
        # "contrast_values" : [1.0],
        # "shadow_adjusteds": [30,60]
    }


    for name in names:

        file_name = os.path.basename(os.path.join(path, name))
        file_name_without_extension = os.path.splitext(file_name)[0]
        image = cv2.imread(os.path.join(path, name))
        # cv2.imshow("test", image)
        for key, values  in methods.items():
            if key == "brightness_values":
                for value in values:
                    brightness_adjusted = adjust_brightness(image, value)
                    suffix = "bright_" + str(value) + ".png"
                    cv2.imwrite(os.path.join(save_path,file_name_without_extension + suffix),brightness_adjusted)


            # cv2.imshow("after_bn", image)
            # cv2.waitKey(0)
            if key == "contrast_values":
                for value in values:
                    contrast_adjusted = adjust_contrast(image, value)
                    suffix = "contrast_" + str(value) + ".png"
                    cv2.imwrite(os.path.join(save_path, file_name_without_extension + suffix), contrast_adjusted)



            if key == "shadow_adjusteds":
                for value in values:
                    shadow_adjusted = adjust_shadows(image, value)
                    suffix = "shallow_" + str(value) + ".png"
                    cv2.imwrite(os.path.join(save_path, file_name_without_extension + suffix),shadow_adjusted)


