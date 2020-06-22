# FACEIt-SPADE #

Generating faces by manipulating segmentation maps of facial attributes using AI.

## Project Overview: ##

Developed an interactive web application to create realistic faces from semantic segmentation maps of the facial attributes using a SPADE based GAN and deployed with Streamlit.
This a type of conditional Generative Adversarial Network(GAN) that takes in segmentation map of the facial attributes and generates a real looking face conditioned according to the input map. 
The process is termed as Semantic Image Synthesis.

## Presentation Slides : ##
Details about the project can viewed from Slide deck here [Slides](https://docs.google.com/presentation/d/1g1K8sfpHj0pCookdpxHvF1n1CFKMPeFp2Sqz1Ngfvf4/edit#slide=id.g8a61dbc963_0_51)


## Streamlit App: Create your own Avatars for video games!: ##
Run the App [FACEIt](http://34.221.66.235:8501)
Here's a small example of the streamlit app. This interactive web application lets the user select from a wide variety of facial attribute masks and manipulate and move the masks to generate segmented face maps and converts them to pictures of real looking faces.

![alt text](https://github.com/SAPreetha/FaceIt-SPADE/blob/master/Images/1.gif)

## Usage ##
### Installation: ###
```
git clone https://github.com/SAPreetha/FaceIt-SPADE
cd SPADE/
pip install requirements.txt
streamlit run faceit_app.py
```

### DATA and Training details ###
Data:
Dataset used : [CelebA](https://github.com/switchablenorms/CelebAMask-HQ) 
Details: 30000 images with facial attribute masks
Sample images in Folder: datasets/celeb_data/train_img and datasets/celeb_data/train_label/
Codes for generating segmentation maps from the masks: Data_gen Folder

Training details:
*  Training set 20K images(128*128 pixels)
*  Batch size 32 images, Trained using 8 distributed K80 GPUs


### Pretrained Model files : [checkpoints]()  ###
* latest_net_D.pth
* latest_net_G.pth


### Train ###
```
$cd SPADE/
$python train.py --name celeb --gpu_ids 0,1,2,3,4,5,6,7 --batchSize 32 --dataset_mode custom --no_instance --label_nc 19 --label_dir datasets/celeb_data/train_label/ --image_dir datasets/celeb_data/train_image/ --preprocess_mode 'resize_and_crop' --load_size 128 --aspect_ratio 1 --crop_size 128 --niter 100 --tf_log 

```


### Test ###

```
$cd SPADE/
$python test.py --name celeb_new --label_nc 19 --dataset_mode custom --no_instance --preprocess_mode 'resize_and_crop' --label_dir datasets/celeb_data/test2_label --image_dir datasets/celeb_data/test2_image/ --load_size 128 --aspect_ratio 1 --crop_size 128
```

## Results: ##

![alt text](https://github.com/SAPreetha/FaceIt-SPADE/blob/master/Results/Ex1.jpg)


![alt text](https://github.com/SAPreetha/FaceIt-SPADE/blob/master/Results/Ex2.jpg)



## Model: ##

The model called SPADE is developed by [NVlabs](https://github.com/NVlabs/SPADE) and is based on the [paper](https://arxiv.org/pdf/1903.07291.pdf).
My project uses SPADE model codes from the NVIDIA repository and the training procedure from their paper.

![alt text](https://github.com/SAPreetha/FaceIt-SPADE/blob/master/Images/spade_architecture.jpg)
![alt text](https://github.com/SAPreetha/FaceIt-SPADE/blob/master/Images/gen_dis_architecture.jpg)


![alt text](https://github.com/SAPreetha/FaceIt-SPADE/blob/master/Images/loss.jpg)

Loss Functions:
* Hinge loss: A generated Image is rescaled to multiples scales and for each of them discriminator computes  the realness score and back propagates the cumulative loss
* Feature matching loss: L1 distance between the discriminator feature maps of the real images and the discriminator feature maps of the fake images is penalized.
* VGG loss:  VGG-19 pre-trained on Imagenet is used to compute the feature maps for real and fake images. Then L1 distance between the feature maps of real and fake images is penalized.








## Acknowledgements: ##
##### Articles: #####
* Understanding GauGAN [Article](https://blog.paperspace.com/nvidia-gaugan-introduction/)

* SPADE: State of the art in Image-to-Image Translation by Nvidia [Article](https://medium.com/@kushajreal/spade-state-of-the-art-in-image-to-image-translation-by-nvidia-bb49f2db2ce3)
