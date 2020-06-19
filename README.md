# FACEIt-SPADE #

Generating faces by manipulating segmentation maps of facial attributes using AI.

## Project Overview: ##

Developed an interactive web application to create realistic faces from semantic segmentation maps of the facial attributes using a SPADE based GAN and deployed with Streamlit.
This a type of conditional Generative Adversarial Network(GAN) that takes in segmentation map of the facial attributes and generates a real looking face conditioned according to the input map. 
The process is termed as Semantic Image Synthesis.

## Presentation Slides : ##
Details about the project can viewed from Presentation [Slides](https://docs.google.com/presentation/d/1g1K8sfpHj0pCookdpxHvF1n1CFKMPeFp2Sqz1Ngfvf4/edit#slide=id.g8a61dbc963_0_51)

## Results: ##

![alt text](https://github.com/SAPreetha/FaceIt-SPADE/blob/master/Results/Ex1.jpg)


## Streamlit App: ##
Here's a small example of the streamlit app. This interactive web application lets user manipulate and move the facial attribute masks to generate segmented face maps and converts them to picture of real looking faces.

![](https://github.com/SAPreetha/FaceIt-SPADE/blob/master/Images/1.mov)

## Codes: ##


## Model: ##

The model called SPADE is developed by [NVlabs](https://github.com/NVlabs/SPADE) and is based on the [paper](https://arxiv.org/pdf/1903.07291.pdf).
My project uses SPADE model codes from the NVIDIA repository and the training procedure from their paper.

![alt text](https://github.com/SAPreetha/FaceIt-SPADE/blob/master/Images/spade_architecture.jpg)
![alt text](https://github.com/SAPreetha/FaceIt-SPADE/blob/master/Images/gen_dis_architecture.jpg)

### Loss graphs ###

Loss Functions:
* Hinge loss: A generated Image is rescaled to multiples scales and for each of them discriminator computes  the realness score and back propagates the cumulative loss
* Feature matching loss: L1 distance between the discriminator feature maps of the real images and the discriminator feature maps of the fake images is penalized.
* VGG loss:  VGG-19 pre-trained on Imagenet is used to compute the feature maps for real and fake images. Then L1 distance between the feature maps of real and fake images is penalized.

![alt text](https://github.com/SAPreetha/FaceIt-SPADE/blob/master/Images/loss.jpg)





Data:
Dataset used : [CelebA](https://github.com/switchablenorms/CelebAMask-HQ) 
Details: 30000 images with facial attribute masks
Sample images in Folder: Data
Codes for generating segmentation maps from the masks: Data_gen Folder

Training details:





## Acknowledgements: ##
##### Articles: #####
* SPADE: State of the art in Image-to-Image Translation by Nvidia [Article](https://medium.com/@kushajreal/spade-state-of-the-art-in-image-to-image-translation-by-nvidia-bb49f2db2ce3)
* Understanding GauGAN [Article](https://blog.paperspace.com/nvidia-gaugan-introduction/)
