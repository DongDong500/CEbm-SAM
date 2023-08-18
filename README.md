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
@article{shin2023cemb,
  title={CEmb-SAM: Segment Anything Model with Condition Embedding for Joint Learning from Heterogeneous Datasets},
  author={Shin, Dongik and Kim, Beomsuk and Baek, Seungjun},
  journal={arXiv preprint arXiv:2308.06957},
  year={2023}
}
```