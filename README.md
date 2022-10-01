# container-orientation-detection
container orientation detection

## INSTALLATION 
```
pip install -r requirement.txt

or

pip3 install -r requiremtnt.txt

```


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
