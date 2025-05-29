<div align="center">

# LuSeg: Efficient Negative and Positive Obstacles Segmentation via Contrast-Driven Multi-Modal Feature Fusion on the Lunar

</div>
<p align="center">
  <a href="https://github.com/Shuaifeng-Jiao">Shuaifeng Jiao</a>,
  <a href="https://ieeexplore.ieee.org/author/37085664722">Zhiwen Zeng</a>,
  <a href="https://github.com/WangYushan9264">Zhuoqun Su</a>,
  <a href="https://github.com/Chen-Xieyuanli">Xieyuanli Chen</a>
   Zongtan Zhou<sup>*</sup>,
  <a href="https://scholar.google.com.hk/citations?hl=en&user=cp-6u7wAAAAJ">Huimin Lu</a>
 
  
</p>

<p align="center">
  <span style="display: inline-block;">
    <a href="https://arxiv.org/pdf/2503.11409" target='_blank'>
      <img src="https://img.shields.io/badge/Paper-%F0%9F%93%83-slategray">
    </a>
  </span>
  
  <span style="display: inline-block;">
    <a href="https://drive.google.com/drive/folders/17qlztnZj_POkljKoBVu3Qccu09rBvYwT?usp=sharing" target='_blank'>
      <img src="https://img.shields.io/badge/LESS-%F0%9F%9B%B0%EF%B8%8F-lightyellow">
    </a>
  </span>

  <span style="display: inline-block;">
    <a href="https://drive.google.com/drive/folders/1u5eHiEPSuPZm1nmjdD4iixaATTorBbI8?usp=sharing" target='_blank'>
      <img src="https://img.shields.io/badge/Dataset-%F0%9F%93%82-lightblue">
    </a>
  </span>

  <span style="display: inline-block;">
    <a href="https://1drv.ms/v/c/9fa243fc9d6318bd/ERalIxw2JaRPgX6dHvGNDggB-5JHOughIMZkYQ6CgkrQpQ?e=y4zu7J" target='_blank'>
      <img src="https://img.shields.io/badge/Video-%F0%9F%8E%AC-pink">
    </a>
  </span>
</p>




## 🔥News:

- [2025-03] Code released.
- [2025-02] Submitted to IROS 2025.

# 1. 🛰️Introduction
This repository introduces the Lunar Exploration Simulator System (LESS), a lunar surface simulation system, alongside the LunarSeg dataset, which supplies RGB-D data for the segmentation of lunar obstacles, 
including both positive and negative instances. Furthermore, it presents a novel two-stage segmentation network, termed LuSeg.


Our accompanying video is available at **[Demo](https://1drv.ms/v/c/9fa243fc9d6318bd/ERalIxw2JaRPgX6dHvGNDggB-5JHOughIMZkYQ6CgkrQpQ?e=y4zu7J)**
<p align="center">
  <img src="./assets/cover.jpg" alt="Alt text" width="600" height="350">
</p>

## Citation

Please cite the corresponding paper:
```bibtex
@article{jiao2024luseg,
  title={LuSeg: Efficient Negative and Positive Obstacles Segmentation via Contrast-Driven Multi-Modal Feature Fusion on the Lunar},
  author={Shuaifeng Jiao, Zhiwen Zeng, Zhuoqun Su, Xieyuanli Chen, Zongtan Zhou, Huimin Lu},
  journal={arXiv preprint arXiv:2503.11409},
  year={2025}
}
```


# 2. 🇨🇳Lunar Exploration Simulation System (LESS)
## 2.1 The Overall of Lunar Exploration Simulator System 
The LESS system integrates a high-fidelity lunar terrain model, a customizable rover platform, and a multi-modal sensor suite, while also supporting the Robot Operating System (ROS) to enable realistic data generation and the validation of autonomous perception algorithms for the rover.LESS provides a scalable platform for developing and validating perception algorithms in extraterrestrial environments. This open-source framework is designed for high extensibility, allowing researchers to integrate additional sensors or customize terrain models according to the specific requirements of their applications.
<p align="center">
  <img src="./assets/LESS.jpg" alt="Alt text" width="600" height="350">
</p>


## 2.2 Application Examples of LESS
You can collect multimodal data based on your needs in the LESS system.
<p align="center">
  <img src="./assets/Lunar_dataset.gif" alt="Alt text" width="600" height="380">
</p>

## 2.3 Installation

To install the LESS on your workstation and learn more about the system, please refer to the [LESS_install](LESS_Install.md).

# 3.💡LuSeg
## 3.1 LuSeg Overview
LuSeg is a novel two-stage training segmentation
method that effectively maintains the semantic consistency of multimodal features via our proposed Contrast-Driven Fusion module. Stage I involves single-modal training using only RGB images
as input, while Stage II focuses on multi-modal training with both RGB and depth images as input. In Stage II, the output of the depth
encoder is aligned with the output of the RGB encoder from Stage I, whose parameters are frozen during this stage. This serves as input
to our proposed Contrast-Driven Fusion Module (CDFM). The final output of Stage II is the result of our LuSeg.
<p align="center">
  <img src="./assets/framework11.png" alt="Alt text" width="800" height="350">
</p>

## 3.2 Dataset
The **LunarSeg** dataset is a dataset of lunar obstacles, including both positive and negative instances. 
The dataset is collected using the LESS system and is available
for download at [Google Drive](https://drive.google.com/drive/folders/1u5eHiEPSuPZm1nmjdD4iixaATTorBbI8?usp=sharing).

## 3.3 Pre-trained weights
The pre-trained weights of LuSeg can be downloaded from [here](https://drive.google.com/drive/folders/1vi8G7TvnZ6snw-Pb7n-0XKohm6wvow9_?usp=sharing).

## 3.4 Training and Evaluation
We train and evaluate our code in Python 3.7, CUDA 12.1, Pytorch 2.3.1

### Train
You can download our pretrained weights to reproduce our results,
and you also can train the LuSeg model on the LunarSeg dataset by running the following command:
```bash
#Stage I
python train_RGB.py --data_dir /your/path/to/LunarSeg/ --batch_size 4 --gpu_ids 0

#Stage II
python train_TS.py --data_dir /your/path/to/LunarSeg/ --batch_size 4 --gpu_ids 0 --rgb_dir /your/path/to/LunarSeg/StageI/trained_rgb/weight/
```
### Evaluation
You can evaluate the LuSeg model on the LunarSeg dataset by running the following command:
```bash
python run_demo_lunar.py --data_dir /your/path/to/LunarSeg/test/ --batch_size 2 --gpu_ids 0 --model_dir /your/path/to/LunarSeg/StageII/trained_ts/weight/
```

# Acknowledgement
We would like to express our sincere gratitude for the following open-source work that has been immensely helpful in the development of LuSeg.
- [InconSeg](https://github.com/lab-sun/InconSeg) InconSeg: Residual-Guided Fusion With Inconsistent Multi-Modal Data for Negative and Positive Road Obstacles Segmentation.

# License
This project is free software made available under the MIT License. For details see the LICENSE file.
