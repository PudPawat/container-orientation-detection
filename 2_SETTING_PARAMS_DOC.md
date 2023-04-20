# setting_params.py usage 
this is an algorithm to setup all process computer vision within one config file. 

## content 

- setup config of the setting e.g. path to image directory [config/main.json]
- UI usage
- save and process 


## setup config

this topic will talk about [config/main.json] 
let's look into the format first
#### focus only the fields we use

- "source" is images directory to setup parameters is could be "relative path" or "absolute path"
- "process" is the algorithm to process the image for example\
["HSV", "erode", "dilate"] so it will process HSV algorithm > erode > dilate respectively\
these's only used for our algorithm
```
{
  "basic": {
    "source": "dataset\\20230318", # images directory to setup parameters
    "resize": "False", 
    "camera_config": "config/camera_config/a2A2590-60ucPRO_40065215.pfs",
    "process_name": ["erode","sharp","blur","gaussianblur","thresh","line","HSV","dilate","canny","circle","sobel","barrel_distort","crop","contour_area"],
    "process": ["crop","barrel_distort","HSV","erode","dilate","contour_area","thresh"], # the process that you need to arrange
    "config_path": "./config/",
    "config_name_format": "notchv2_config_",
    "params_temp": "config/params.json"
  }
}
```


## UI usage
then go to setting up the parameters 
I suggest to visit this/

[![Watch the video](https://drive.google.com/file/d/14nNEcNqJ4s8NHQkEZMCcJOjjBIxKbBbl/view?usp=sharing)

#### command in UI \
Press A to update to parameters\ 
Press S to save\
Press N to Next the image \
Press R to previous image \
Press Q to quit

#### getting the config file
if the format of image is fit to (e.g. S0000_0.jpg) it will automatically save in this file "notchv2_config_S0000.json"
```
{
  "params": {
    a dict from setting_params.py
  },
  "debug": "True", # to show debug message
  "show_result": "True", # to show image
  "flag_rotate": "None", 
  "resize_ratio": 0.3, # resize image ratio 
  "save_img": "True", # save debug image
  "inverse_threshold": "True", # inverse threshold binary 
  "n_symetric": 1, # number of symetric of the unique object to detect 
  "crop_circle_fix_inner_r": [
    538.7253254825479,
    402.83041037971105,
    254.7494678162871
  ],  # inner outer of the unique object 
  "crop_circle_fix_outer_r": [
    538.6831364300194,
    403.3175883340236,
    266.6971472939909
  ],  # inner outer of the unique object 
  "crop_circle_platform": [
    537,
    400,
    391
  ],
  "method_names": [
    "tiny",
    "simple",
    "compare",
    "comparev2"
  ],
  "method": "",
  "simple_tiny": {
    "outer_r_safety": -10
  }
}
```

else it will save into params.json like this format
```
{"crop": [73, 76], "barrel_distort": [405, 391, 57, 50, 0, 73, 10, 10], "HSV": [0, 0, 61, 180, 255, 255], "erode": [2, 0], "dilate": [6, 2], "contour_area": [822157, 1486880, 3, 1], "thresh": [10, 1]}
```
in the else case you need to copy "notchv2_config_default.json" and replace "default" to "S0000" (the format following object)\
and replace the content from "params.json" into "params" field in "notchv2_config_S0000.json". 


## save and process 
make sure the file you save put it in /config directory