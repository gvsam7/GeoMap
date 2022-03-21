# GeoMap

## Description

The project's aim is to apply deep learning techniques, to map cement plants in China and help monitor the pollution.  

![China](https://github.com/gvsam7/GeoMap/blob/main/Images/China_cement.PNG)

*Data:* Data of cement plants and the surounded landcovers where extracted from Landsat8 earth observation satellite. Landsat8 employ's two main sensors:
1. Operational Land Imager (OLI): generates 9 spectral bands (Band 1 to 9). OLI images can discriminate vegetation types, cultural features, biomass, vigor, etc.
2. Thermal Infrared Sensor (TIRS): consists of 2 thermal bands (Band 10 and 11) with a spatial resolution of 100 meters. TIRS measures Earth’s thermal energy, useful when tracking how land and water is used.
LandSat band 10 (B10) Thermal infrared (TIRS) 1, band 11 Thermal infrared (TIRS) 2, and band 7 Short-Wave Infrared (SWIR) 2 images were extracted from the Satellites and used to train various Deep Learning architectures to classify the cement plants and the surrounding land cover.

![Band7](https://github.com/gvsam7/GeoMap/blob/main/Images/B7_ThermalInfraRed.PNG)

| Bands                                 | Wavelength (micrometers) | Resolution (meters) | Useful for mapping                                                         |
| :---                                  | :---                    | :---                 | :----                                                                      |
| Band 7 - Short-wave Infrared (SWIR) 2 | 2.11-2.29               | 30                   | Improved moisture content of soil and vegetation; penetrates thin clouds   |
| Band 10 - Thermal Infrared (TIRS) 1   | 10.6-11.19              | 100                  | 100 meter resolution, thermal mapping and estimated soil moisture          |
| Band 11 - Thermal Infrared (TIRS) 2   | 11.50-12.51             | 100                  | 100 meter resolution, improved thermal mapping and estimated soil moisture |

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