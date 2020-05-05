# Stochastic-TravelGAN
Final project for NMEP.

ImageNet Classes/Labels: [Link](https://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57)
ImageNet Utils: [Link](https://github.com/tzutalin/ImageNet_Utils)

model.py: 
- Contains definitions for each of the networks and losses in the TravelGAN.

data.py:
- Contains functions that help get the data. The CelebA dataset is 
  [here](https://www.kaggle.com/jessicali9530/celeba-dataset). To use it, create
  a directory named celebA in the data/ folder and unzip the zip file in
  data/celebA.

train.py:
- Contains the code that initializes the TravelGAN, optimizers, etc., and then
  trains it. To run everything, just do python3 train.py.
