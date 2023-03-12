# container-orientation-detection
container orientation detection

## config setup 
in setup folder 

### consider 
image name in a directory for setting up should be 
S00000_0.jpg 

and the config will save the config file name following S00000 so please make sure the name of the objec in the image is right
```
{
  "params": {"HSV": [0, 0, 55, 180, 255, 255], "gaussianblur": [1, 1], "dilate": [5, 0], "erode": [15, 0]},
  "debug": "True",
  "show_result": "True",
  "flag_rotate": "None",
  "resize_ratio": 0.3,
  "save_img": "True",
  "inverse_threshold": "True",
  "n_symetric": 1,
  "crop_circle_fix_inner_r": [537, 400, 180],
  "crop_circle_fix_outer_r": [537, 400, 200],
  "crop_circle_platform": [537, 400, 391],
  "method_names": ["simple", "compare", "comparev2"],
  "method": ""
}
``` 

![alt text](https://github.com/PudPawat/container-orientation-detection/blob/main/info_image/config_new_setting.png?raw=true)

#### 2 setting algorithm 
#### 1: set_circle.py
To set inner_circle, outer_circle, and platform circle

so set the path directory 
'''
path = "../dataset/20230311"
'''

and the method to draw the circle is pick 3 points in the image 
after you pick it perfectly 

PRESS
i = save inner_circle 
o = save outer_circle
p = save platform_circle 

#### 2: setting params (HSV, dilate, erode)

about  setting_param.py 

1.) see the config of the setting in main.json in /config

``` 
{
  "basic": {
    "source": "F:\\Ph.D\\circle_classification\\container-orientation-detection\\dataset\\20230311",
    "camera_config": "config/camera_config/a2A2590-60ucPRO_40065215.pfs",
    "process_name": ["erode","sharp","blur","thresh","line","HSV","dilate","canny","circle","sobel"],
    "process": ["HSV","dilate","erode"],
    "config_path": "../config/",
    "config_name_format": "notchv2_config_"
  }
}
``` 
in general set only source 
source is the path to images directory for setting up

2.) run setting_param.py 
setting all params in UI

PRESS s to save to the config 