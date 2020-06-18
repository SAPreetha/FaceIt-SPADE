#FACEIt_SPADE:
Generating faces by manipulating segmentation maps of facial attributes using AI.

##Project Overview:
Developed an interactive web application to create realistic faces from semantic segmentation maps of the facial attributes using a SPADE based GAN and deployed with Streamlit.
This a type of conditional Generative Adversarial Network(GAN) that takes in segmentation map of the facial attributes and generates a real looking face conditioned according to the input map. 
The process is termed as Semantic Image Synthesis.
##Results:

![alt text](https://github.com/SAPreetha/FaceIt-SPADE/blob/master/Results/Ex1.jpg)


The model called SPADE is developed by [NVlabs](https://github.com/NVlabs/SPADE) and is based on the [paper](https://arxiv.org/pdf/1903.07291.pdf).
My project uses SPADE model codes from the NVIDIA repository and the training procedure from their paper.




Data:
Dataset used : [CelebA](https://github.com/switchablenorms/CelebAMask-HQ) 
Details: 30000 images with facial attribute masks
Sample images in Folder: Data
Codes for generating segmentation maps from the masks: Data_gen Folder

Training details:


Codes: 


Acknowledgements:
Articles:
*SPADE: State of the art in Image-to-Image Translation by Nvidia.https://medium.com/@kushajreal/spade-state-of-the-art-in-image-to-image-translation-by-nvidia-bb49f2db2ce3
*Understanding GauGAN:[Article](https://blog.paperspace.com/nvidia-gaugan-introduction/)
