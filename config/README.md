# Setting config
## classification.json


the classification.json will be load to conpareimg.py 

compareimg.py is the main.py to run the classification and orientation

```sh
{
  "folder_ref": "dataset\\class_registeration", ## path to dataset for classification
  "debug": "True", # to see debug process
  "crop_circle_platform": [546.1550518881522, 421.04824877486305, 375], ## circle of the platform(optional)
  "resize_ratio": 0.3 ## 0.3 is fix (optional)
}
```
### Set class in the dataset/class_registeration 

set the name of an image 
- name_somthing.jpg --->
things before undersocre is name of the class

![alt text](https://github.com/PudPawat/container-orientation-detection/blob/main/info_image/classification_data_set.PNG?raw=true)

## notch_config.json
- Type some Markdown on the left
- See HTML in the right
- ✨Magic ✨

```sh{
  "params": {"HSV": [0, 0, 70, 180, 255, 255], "erode": [4, 0], "dilate": [4, 0]}, ## HSV and other img_processign parameter (change only HSV is enough)
  ### ""HSV: [low_h,low_s, low_v,high_h,high_s, high_v]
  "debug": "False",
  "show_result": "False",
  "flag_rotate": "None",
  "resize_ratio": 0.3,
  "crop_circle_fix_r": [546.1550518881522, 421.04824877486305, 255], ## circle on the object
  "crop_circle_platform": [546.1550518881522, 421.04824877486305, 375], ## circle on the the platform the same in classification
  "threshold_score": 100, 
  "text_color": [255,255,0],
  "notch": {
    "notch_dis_criterion": 15, ## now scale image 0.3, So use 15. if bigger use bigger number
    "notch_dis_from_right_edge": 8, ## if notch on the edge just set 5 - 10 
    # else please set the destance to notch from edge
    "notch_area_min": 500, # min area to detect  notch to filter contours
    "notch_area_max": 5000 # max area to detect notch to filter contours
  },
  "object": {
    "object_contour_area_min": 2000, ## min area of the object
    "object_contour_area_max": 200000, ## max area of the object
    "object_contour_distance_criterion": 50, ## dis from center of the platform 
    "circle_radius_minus_safety": 10 ## to remove noise on edge
  }
}
```

### set HSV 
# GOOD
![alt text](https://github.com/PudPawat/container-orientation-detection/blob/main/info_image/Good_HSV_setting.PNG?raw=true)

# BAD
![alt text](https://github.com/PudPawat/container-orientation-detection/blob/main/info_image/BAD_HSV_SETTING_1.PNG?raw=true)

![alt text](https://github.com/PudPawat/container-orientation-detection/blob/main/info_image/BAD_HSV_SETTING_2.PNG?raw=true)
