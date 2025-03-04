# CEmb-SAM
This is the official repository for CEmb-SAM: Segment Anything Model with Condition Embedding for Joint Learning from Heterogeneous Datasets.


## Getting Started
We provide GUI to test on sample images

1. GUI

Install `PyQt5`

```
python gui.py --sam_ckpt <path/to/sam_vit_b/checkpoint> --ckpt <path/to/CEmbSam/checkpoint> --emb_class <the_number_of_embedding_classes>
```

Load the image to the GUI and specify segmentation targets by drawing bounding boxes.

<!-- ![Demo](./assets/gui.py@server7%202023-11-08.gif) -->
<img src="./assets/gui.py@server7%202023-11-08.gif" width="400" height="300" />

## Training Model (CEmb-sam)

#### Datasets Preparation
We use two datasets, the public benchmark BUSI dataset and Nerve dataset.

1. BUSI

#### Training
The model was trained on RTX3090.


## Sample result images
Segmentation results on BUSI (1st and 2nd rows) and peripheral nerve
dataset (3rd and 4th rows).
<p float="left">
  <img src="assets/results (4).jpg?raw=true" width="80.00%" /> 
</p>

## Reference
```
@inproceedings{shin2023cemb,
  title={CEmb-SAM: Segment Anything Model with Condition Embedding for Joint Learning from Heterogeneous Datasets},
  author={Shin, Dongik and Kim, MD, Beomsuk and Baek, Seungjun},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={275--284},
  year={2023},
  organization={Springer}
}
```