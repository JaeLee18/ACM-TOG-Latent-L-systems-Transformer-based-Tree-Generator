# [Official] Latent-L-systems:Transformer based Tree Generator🌲

# Environment
```
conda env create -f environment.yml
```

# Data
* .lstring files are uploaded to Google Drive. Please see the urls under `data/lstrings.txt`

# Visualization
* Please refer to https://github.com/edisonlee0212/PlantArchitect

# Steps
1) First, put a folder with *.lstring files on the same directory.
```
├── Acacia
│   ├── *.lstring
├── preprocessing.py
```    
2) Run the Preprocessing file. It will do all the jobs. It will take about 2 hours using multi-processing with all CPUs.
```
python preprocessing.py
```
3) Training
* Set the correct file paths from Step2 in train.py Line#32, Line#36, Line#40, Line#41
```
python train.py
```
4) Inference
```
WIP
```

If our paper has been helpful, we kindly ask that you cite it in your work. https://dl.acm.org/doi/10.1145/3627101

```
@article{10.1145/3627101,
author = {Lee, Jae Joong and Li, Bosheng and Benes, Bedrich},
title = {Latent L-Systems: Transformer-Based Tree Generator},
year = {2023},
issue_date = {February 2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {43},
number = {1},
issn = {0730-0301},
url = {https://doi.org/10.1145/3627101},
doi = {10.1145/3627101},
journal = {ACM Trans. Graph.},
month = {nov},
articleno = {7},
numpages = {16},
keywords = {geometric modeling, L-systems, neural networks}
}
```

# Video
[![Intro Video](https://img.youtube.com/vi/1SPSQ-IwcvQ/0.jpg)](https://www.youtube.com/watch?v=1SPSQ-IwcvQ)



