# ZPO projekt - Backend
Code based on my previous framework BNN with some improvements/customizations
## Features
Provides CLI interface for upscaling images using various algorithms.
### Upscaling anime art
![image](https://github.com/Panjaksli/BNN/assets/82727531/718568a6-111a-4436-870b-c206874185eb)
### How does it work ?
The model is trained on the error of reference image and low res image upscaled with bicubic interpolation:\
d(x) = f(x) - g(x),\
where: d(x) is error function, f(x) is full resolution image and g(x) is an approximation of f(x).\
This error is then added to the upscaled image during inference:\
f(x) = g(x) + d(x) = g(x) - g(x) + f(x) = f(x).\
This results in reconstruction of original image f(x).

