# container-orientation-detection
container orientation detection

## INSTALLATION 
```
pip install -r requirement.txt
```
or

```
pip3 install -r requiremtnt.txt
```

## Name format 
### IMAGE
image S0000_0.jgp 

### Algorithm
config for algorithm: notchv2_config_S0000.json 

### Setting 
main.json for setting up conputer vision params

## SETUP 
algorithm config setup 

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
- run seeting_params.py to set design parameters [HSV, dilate, erode, etc...]
see video https://drive.google.com/file/d/14nNEcNqJ4s8NHQkEZMCcJOjjBIxKbBbl/view?usp=sharing 
- using set_circle.py to see circle parameters in "crop_circle_fix_inner_r", "crop_circle_fix_outer_r", "crop_circle_platform"

![alt text](https://github.com/PudPawat/container-orientation-detection/blob/main/info_image/config_new_setting.png?raw=true)




## APPLICATION AND EXPLAINATION 
this code is to classification compareimg.py by easily using by put a reference images into a folder

![alt text](https://github.com/PudPawat/container-orientation-detection/blob/main/info_image/EX1.PNG?raw=true)


and the class has notch can the detect angle of the object

![alt text](https://github.com/PudPawat/container-orientation-detection/blob/main/info_image/EX2.PNG?raw=true)
![alt text](https://github.com/PudPawat/container-orientation-detection/blob/main/info_image/EX3.PNG?raw=true)


## Add detect notch from a CLASS

- add notch_config_{class}.json # with all setting
- go to comaprimg.py 
- find the code below
orientation_detection_A repersent a setting notch of class A
orientation_detection_B repersent a setting notch of class B

```
### fill the other object
    orientation_detection_A = OrientationDetection( path=os.path.join(folder, names[i]), json_path="config/notch_config_A.json")
    ## EX
    orientation_detection_B = OrientationDetection( path=os.path.join(folder, names[i]), json_path="config/notch_config_B.json")
```
#### Attribute explaintion 
path = "path to image background"
json_path = "config/notch_config_A.json" # setting path in config see setting in config

- and add if condition like the code below
the code be low show adding a class B 
```
        ### Example
        elif name_class == "B":
            angle = orientation_detection_B.detect(img1)
            if angle is not None:
                result = cv2.putText(img1, "ANGLE: {}".format(str("%.2f" % round(angle, 2))),
                                     (0, img1.shape[0] - 10), cv2.FONT_HERSHEY_COMPLEX, 8, (0, 50, 255), 8)

                cv2.namedWindow("RESULT", cv2.WINDOW_NORMAL)
                cv2.imshow("RESULT", result)
