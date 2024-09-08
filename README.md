# OOD-Libraries

## OOD Detection

| OOD Method | Description | Implement | Publication |
|:-------:|-------|-------|-------|
| CORES | - | - | ![Static Badge](https://img.shields.io/badge/2024-CVPR-brightgreen?style=for-the-badge) |
| [NAP](#nap) | - | - | ![Static Badge](https://img.shields.io/badge/2023-CVPR-brightgreen?style=for-the-badge) |
| [KNN](#knn) | - | - | ![Static Badge](https://img.shields.io/badge/2022-ICML-orange?style=for-the-badge) |
| [ASH](#ash) | - | [ASH.py](ood_methods/ASH.py) | ![Static Badge](https://img.shields.io/badge/2022-ICLR-4bb2ff?style=for-the-badge) |
| [DICE](#dice) | - | [DICE.py](ood_methods/DICE.py) | ![Static Badge](https://img.shields.io/badge/2022-ECCV-pink?style=for-the-badge) |
| [ReAct](#react) | - | [ReAct.py](ood_methods/ReAct.py) | ![Static Badge](https://img.shields.io/badge/2021-NeurIPS-8A2BE2?style=for-the-badge) |
| [GradNorm](#gradnorm) | - | [GradNorm.py](ood_methods/GradNorm.py) | ![Static Badge](https://img.shields.io/badge/2021-NeurIPS-8A2BE2?style=for-the-badge) |
| [Energy](#energy) | - | [Energy.py](ood_methods/Energy.py) | ![Static Badge](https://img.shields.io/badge/2020-NeurIPS-8A2BE2?style=for-the-badge) |
| [Mahalanobis](#maha) | - | [Mahalanobis.py](ood_methods/Mahalanobis.py) | ![Static Badge](https://img.shields.io/badge/2018-NeurIPS-8A2BE2?style=for-the-badge) |
| [ODIN](#odin) | - | [ODIN.py](ood_methods/ODIN.py) | ![Static Badge](https://img.shields.io/badge/2018-ICLR-4bb2ff?style=for-the-badge) |
| [MSP](#msp) | - | [MSP.py](ood_methods/MSP.py) | ![Static Badge](https://img.shields.io/badge/2017-ICLR-4bb2ff?style=for-the-badge) |


## Datasets

###  

### large-scale datasets
ImageNet-1000
iNaturalist
SUN
Places
Texture

## References

<div id="msp"></div> 


- [A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks.](https://arxiv.org/pdf/1610.02136) [[Code]](https://github.com/hendrycks/error-detection)


<div id="odin"></div> 

- [Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks.](https://arxiv.org/pdf/1706.02690) [[Code]](https://github.com/facebookresearch/odin)

<div id="maha"></div> 

- [A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks.](https://arxiv.org/pdf/1807.03888) [[Code]](https://github.com/pokaxpoka/deep_Mahalanobis_detector)

<div id="energy"></div>

- [Energy-based Out-of-distribution Detection.](https://arxiv.org/pdf/2010.03759) [[Code]](https://github.com/wetliu/energy_ood)

<div id="gradnorm"></div>

- [On the Importance of Gradients for Detecting Distributional Shifts in the Wild.](https://arxiv.org/pdf/2110.00218) [[Code]](https://github.com/deeplearning-wisc/gradnorm_ood)

<div id="react"></div> 

- [ReAct: Out-of-distribution Detection With Rectified Activations.](https://arxiv.org/pdf/2111.12797) [[Code]](https://github.com/deeplearning-wisc/react)

<div id="dice"></div> 

- [DICE: Leveraging Sparsification for Out-of-Distribution Detection.](https://arxiv.org/pdf/2111.09805) [[Code]](https://github.com/deeplearning-wisc/dice)

<div id="ash"></div> 

- [Extremely Simple Activation Shaping for Out-of-Distribution Detection.](https://arxiv.org/pdf/2209.09858) [[Code]](https://github.com/andrijazz/ash)


<div id="knn"></div> 

- [Out-of-Distribution Detection with Deep Nearest Neighbors.](https://arxiv.org/pdf/2204.06507) [[Code]](https://github.com/deeplearning-wisc/knn-ood)

<div id="nap"></div> 

- [Detection of Out-of-Distribution Samples Using Binary Neuron Activation Patterns.](https://openaccess.thecvf.com/content/CVPR2023/papers/Olber_Detection_of_Out-of-Distribution_Samples_Using_Binary_Neuron_Activation_Patterns_CVPR_2023_paper.pdf) [[Code]](https://github.com/safednn-group/nap-ood)
