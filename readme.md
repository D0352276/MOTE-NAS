# MOTE-NAS: Multi-Objective Training-based Estimate for Efficient Neural Architecture Search

![](https://img.shields.io/badge/Python-3-blue)
![](https://img.shields.io/badge/TensorFlow-2-orange)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

This project provides a few-cost estimate for Neural Architecture Search. 

1. This code covers the core function of the proposed MOTE.
2. File **"example.py"** and **"gen_mote.py"** is the main program, which records the code for MOTE generation.
3. Files **"meta_models.py"**, **"reduced_data.py"** respectively correspond to the core methods mentioned in the paper -
   Reduced Architecture(RA) and Reduced Data(RD).


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


## How to Get MOTE for an Architecture(cell_pth) ?
```python
from nas_prcss import CellPth2Cell
from gen_mote import GetProxyC100TrainData, CellPth2MOTE
train_x,train_y=GetProxyC100TrainData(labels_len=10)
CellPth2MOTE(train_x,train_y,cell_pth,cell_type="nas201",proxy_labels_len=10)
cell_dict=CellPth2Cell(cell_pth)
print(cell_dict["mote"])
```
## How to Get Correlation of MOTE Ranking and Actual Ranking ?
```python
from nas_prcss import SamplingCellPths,FilteringByDirtyBit
from eval_cell import CellPths2Psp
cells_dir="data/nasbench201_img16-10"
gt_key="test_accuracy_200"
cell_pths=SamplingCellPths(cells_dir,shuffle=True)
psp=CellPths2Psp(cell_pths,gt_key,"mote")
print(psp)
```

### Fully Dataset
The NASBench-101 data set is too large, here only the NASBench-201, 
we will provide NB101 ASAP.


### Our Data Format
coming soon.


### The Comparison of Search Efficiency on NASBench-201

For detailed report, please refer to our paper.

<img src=https://github.com/D0352276/MOTE-NAS/blob/main/overview/cmptable.png width=100% />


## TODOs

- Provide NB101.
- Provide more complete code.

