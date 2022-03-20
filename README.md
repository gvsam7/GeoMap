# GeoMap

## Description

This project's aim is to apply deep learning techniques, to map cement plants in China and help monitor the pollution. Classify cement factories using satellite images. 

![China](https://github.com/gvsam7/GeoMap/blob/main/Images/China_cement.PNG)

*Data:* LandSat band 10 (B10) Thermal infrared (TIRS) 1 (10.6-11.19 micrometers wavelength) and band 11 Thermal infrared (TIRS) 2 (11.50-12.51 micrometers wavelength) images
 were extracted from the Satellites and used to train various Deep Learning architectures to classify the cement plants and the surrounding land cover.

![Band7](https://github.com/gvsam7/GeoMap/blob/main/Images/B7_ThermalInfraRed.PNG)

| Bands                                 | Wavelength (micrometers) | Resolution (meters) | Useful for mapping                                                         |
| :---                                  | :---                    | :---                | :----               
| Band 7 - Short-wave Infrared (SWIR) 2 | 2.11-2.29               | 30                  | Improved moisture content of soil and vegetation; penetrates thin clouds   |
| Band 10 - Thermal Infrared (TIRS) 1   | 10.6-11.19              | 100                 | 100 meter resolution, thermal mapping and estimated soil moisture          |
| Band 11 - Thermal Infrared (TIRS) 2   | 11.50-12.51             | 100                 | 100 meter resolution, improved thermal mapping and estimated soil moisture |

<p align="center">
<img src="https://github.com/gvsam7/GeoMap/blob/main/Images/B7B10_ThermalInfraRed.PNG">
</p>

*Architectures:* 5 CNN, ResNet18, ResNet50, VGG13, DenseNet161, EfficientNet.

*Images:* 256x256 pixel images.

*Test Procedure:* 5 runs for each architecture for each of the compressed data. Then plot the Interquartile range.

*Plots:* Average GPU usage per architecture, Interquartile, F1 Score heatmap for each class, Confusion Matrix, PCA and t-SNE plots, and most confident incorrect predictions.

*Data augmentations:* Geometric Transformations, Cutout, Mixup, and CutMix, Pooling (Global Pool, Mix Pool, Gated Mix Pool).

## Papers
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)