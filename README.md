# Simple web API to extract features from images

## Face Detection

Inspired by the [cvlib](https://github.com/arunponnusamy/cvlib), the endpoint
`/faces` returns True if the model detects a face on the picture.

The current model for face detection is an AlexNet based model
presented in the paper
published by **Levi** and **Hassner**: ["Age and Gender Classification using Convolutional Neural Networks"](https://talhassner.github.io/home/projects/cnn_agegender/CVPR2015_CNN_AgeGenderEstimation.pdf)


#### Requirements
 - [aiohttp](https://github.com/aio-libs/aiohttp)
 - [opencv-python](https://github.com/skvark/opencv-python)