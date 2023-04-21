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
for each object name S0000 should have its own config file -> notchv2_config_S0000.json (S0000 is an example)
So, the purpose of the setup is to set this config file completely. 


#### let's look into the file


algorithm config setup : below is a sample of JSON file which needed to set the step is following this.

- 1.) put the images in any folder you want with the correct format S0000_0.jpg 
- 2.) check [SETTING_PARAMS_DOC](SETTING_PARAMS_DOC.md) to run setting_params.py or setting_params copy.py which you need to check main.json details [SETTING_PARAMS_DOC](SETTING_PARAMS_DOC.md)"SETTING_PARAMS_DOC.md"
- 3.) to set up check [set_circle](SET_CIRCL.md) circle_fix_outer_r", "crop_circle_fix_inner_r", "crop_circle_platform"

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


to run the file

## RUN new_algorithm.py 

you can set image directory to test on the line which has path_imgs = ".." 
and replace the path in on .. 

you can test the algorithm on the path "dataset/20230311" if all the config havn't been deleted on the local machine\
it should be able to run without error on every image in the directory. 