# GeoMap

## Description

This project's aim is to apply deep learning techniques, to map cement plants in China and help monitor the pollution. Classify
                cement factories using satellite images. 

![alt text](https://github.com/gvsam7/GeoMap/tree/main/Images/b10_ThermalInfraRed.PNG)

*Data:* LandSat band 10 (B10) Thermal infrared (TIRS) 1 (10.6-11.19 micrometers wavelength) and band 11 Thermal infrared (TIRS) 2 (11.50-12.51 micrometers wavelength) images
 were extracted from the Satellites and used to train various Deep Learning architectures to classify the cement plants and the surrounding land cover.

*Architectures:* 5 CNN, ResNet18, ResNet50.

*Images:* 256x256 pixel images.

*Test Procedure:* 5 runs for each architecture for each of the compressed data. Then plot the Interquartile range.

*Plots:* Average GPU usage per architecture, Interquartile, F1 Score heatmap for each class, Confusion Matrix, PCA and t-SNE plots, and most confident incorrect predictions.

*Data augmentations:* Geometric Transformations, Cutout, Mixup, and CutMix, Pooling (Global Pool, Mix Pool, Gated Mix Pool).
