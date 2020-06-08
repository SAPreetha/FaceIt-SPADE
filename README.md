FACEGAN_SPADE:
Generating and manipulating real facial structure by parts using AI.

Project Overview:
This a type of conditional Generative Adversarial Network(GAN) that takes in semantic segmentation map of the facial attributes and generates a real looking face conditioned according to the input map. The process is termed as Semantic Image Synthesis.

The model is called SPADE was developed by [NVlabs](https://github.com/NVlabs/SPADE) and is based on the [paper](https://arxiv.org/pdf/1903.07291.pdf).
The project uses SPADE model codes from the NVIDIA repository and the training procedure from their paper.

Results:


Data:
Dataset used : [CelebA](https://github.com/switchablenorms/CelebAMask-HQ) 
Details: 30000 images with facial attribute masks
Sample images in Folder: Data
Codes for generating segmentation maps from the masks: Data_gen Folder

Training details:


Codes: 


Acknowledgements:
Articles:
SPADE: State of the art in Image-to-Image Translation by Nvidia.https://medium.com/@kushajreal/spade-state-of-the-art-in-image-to-image-translation-by-nvidia-bb49f2db2ce3
Understanding GauGAN:[Article](https://blog.paperspace.com/nvidia-gaugan-introduction/)
