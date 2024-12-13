# MOTE-NAS: Multi-Objective Training-based Estimate for Efficient Neural Architecture Search

![](https://img.shields.io/badge/Python-3-blue)
![](https://img.shields.io/badge/TensorFlow-2-orange)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

This project provides a few-cost estimate for Neural Architecture Search. 

1. This code covers the core function of the proposed MOTE.

2. File "example.py" and "gen_mote.py" is the main program, which records the code for MOTE generation.

3. Files "meta_models.py", "reduced_data.py" respectively correspond to the core methods mentioned in the paper - Reduced Architecture and Reduced Data.


The paper has been accepted by **NeurIPS 2024.**

Paper Link: https://reurl.cc/KdMalq

<div align=center>
<img src=https://github.com/D0352276/MOTE-NAS/blob/main/overview/overview.png width=100% />
</div>

## Requirements

- [Python 3](https://www.python.org/)
- [TensorFlow 2](https://www.tensorflow.org/)
- [OpenCV](https://docs.opencv.org/4.5.2/d6/d00/tutorial_py_root.html)
- [Numpy](http://www.numpy.org/)


## How to Get Started?
```bash
#Predict
python3 main.py -p cfg/predict_coco.cfg

#Train
python3 main.py -t cfg/train_coco.cfg

#Eval
python3 main.py -ce cfg/eval_coco.cfg
```


## More Info

### Change Model Scale
The model's default scale is 224x224, if you want to change the scale to 320~512, 

please go to cfg/XXXX.cfg and change the following two parts:
```bash
# input_shape=[512,512,3]
# out_hw_list=[[64,64],[48,48],[32,32],[24,24],[16,16]]
# input_shape=[416,416,3]
# out_hw_list=[[52,52],[39,39],[26,26],[20,20],[13,13]]
# input_shape=[320,320,3]
# out_hw_list=[[40,40],[30,30],[20,20],[15,15],[10,10]]
input_shape=[224,224,3]
out_hw_list=[[28,28],[21,21],[14,14],[10,10],[7,7]]

weight_path=weights/224_nolog.hdf5

                         |
                         | 224 to 320
                         V
                         
# input_shape=[512,512,3]
# out_hw_list=[[64,64],[48,48],[32,32],[24,24],[16,16]]
# input_shape=[416,416,3]
# out_hw_list=[[52,52],[39,39],[26,26],[20,20],[13,13]]
input_shape=[320,320,3]
out_hw_list=[[40,40],[30,30],[20,20],[15,15],[10,10]]
# input_shape=[224,224,3]
# out_hw_list=[[28,28],[21,21],[14,14],[10,10],[7,7]]

weight_path=weights/320_nolog.hdf5
```


### Fully Dataset
The entire MS-COCO data set is too large, here only a few pictures are stored for DEMO, 

if you need complete data, please download on this [page.](https://cocodataset.org/#download)


### Our Data Format
We did not use the official format of MS-COCO, we expressed a bounding box as following:
```bash
[ left_top_x<float>, left_top_y<float>, w<float>, h<float>, confidence<float>, class<str> ]
```
The bounding boxes contained in a picture are represented by single json file.

For detailed format, please refer to the json file in "data/coco/train/json".


### AP Performance on MS-COCO

For detailed COCO report, please refer to "mscoco_result".

<img src=https://github.com/D0352276/CSL-YOLO/blob/main/demo/result_table.png width=100% />


## TODOs

- Improve the calculator script of FLOPs.
- Using Focal Loss will cause overfitting, we need to explore the reasons.




