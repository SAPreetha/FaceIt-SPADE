import streamlit as st
from PIL import Image
import cv2
import os
import shutil
import numpy as np

color_list = [[0, 0, 0], [204 / 255, 0, 0], [76 / 255, 153 / 255, 0], [204 / 255, 204 / 255, 0],
              [51 / 255, 51 / 255, 255 / 255], [204 / 255, 0, 204 / 255], [0, 255 / 255, 255 / 255],
              [255 / 255, 204 / 255, 204 / 255], [102 / 255, 51 / 255, 0], [255 / 255, 0, 0], [102 / 255, 204 / 255, 0],
              [255 / 255, 255 / 255, 0], [0, 0, 153 / 255], [0, 0, 204 / 255], [255 / 255, 51 / 255, 153 / 255],
              [0, 204 / 255, 204 / 255], [0, 51 / 255, 0], [255 / 255, 153 / 255, 51 / 255], [0, 204 / 255, 0]]
label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip',
              'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

N_opt = 6 #No of options for each different facial attribute
#merge the masks in the sequence given in label list
def func_merge(im1, im2, im3, im4, im5, im6, im7, im8, im9, im10, im11):
    ret_im1 = im1.copy()
    ret_im1[im1 != 0] = 1
    ret_im1[im2 != 0] = 2
    ret_im1[im3 != 0] = 4
    ret_im1[im4 != 0] = 5
    ret_im1[im5 != 0] = 6
    ret_im1[im6 != 0] = 7
    ret_im1[im7 != 0] = 10
    ret_im1[im8 != 0] = 11
    ret_im1[im9 != 0] = 12
    ret_im1[im10 != 0] = 13
    ret_im1[im11 != 0] = 17

    return (ret_im1)


file_source = '..Path to masks./mask/'
#the app has the following features now
label_list = ['skin', 'nose', 'eye', 'brow', 'mouth', 'lip', 'hair', 'neck']
# size of the segmented map image
L = 512


select_list = [i for i in range(N_opt)]

#selecting pre-processed face skin and neck segmaps that match each other.
option_skin = st.sidebar.selectbox(' Select Skin ', select_list)
filename_skin = file_source + 'skin/' + 'skin_' + str(option_skin) + '.png'
filename_neck = file_source + 'neck/' + 'neck_' + str(option_skin) + '.png'


im_skin = cv2.imread(filename_skin, 0)
im_skin_new = im_skin.copy()
im_neck = cv2.imread(filename_neck, 0)
im_neck_new = im_neck.copy()

option_nose = st.sidebar.selectbox(' Select Nose ', select_list)
filename_nose = file_source + 'nose/' + 'nose_' + str(option_nose) + '.png'

im_nose = cv2.imread(filename_nose, 0)
im_nose_new = im_nose.copy()

#selecting pre-processed left eye and right eye segmaps that match each other.
option_eye = st.sidebar.selectbox(' Select Eye ', select_list)
filename_leye = file_source + 'l_eye/' + 'l_eye_' + str(option_eye) + '.png'
filename_reye = file_source + 'r_eye/' + 'r_eye_' + str(option_eye) + '.png'


im_leye = cv2.imread(filename_leye, 0)
im_leye_new = im_leye.copy()
im_reye = cv2.imread(filename_reye, 0)
im_reye_new = im_reye.copy()


#selecting pre-processed left eyebrow and right eyebrow segmaps that match each other.
option_brow = st.sidebar.selectbox(' Select EyeBrow ', select_list)
filename_lbrow = file_source + 'l_brow/' + 'l_brow_' + str(option_brow) + '.png'
filename_rbrow = file_source + 'r_brow/' + 'r_brow_' + str(option_brow) + '.png'

im_lbrow = cv2.imread(filename_lbrow, 0)
im_lbrow_new = im_lbrow.copy()
im_rbrow = cv2.imread(filename_rbrow, 0)
im_rbrow_new = im_rbrow.copy()

option_mouth = st.sidebar.selectbox(' Select Mouth ', select_list)
filename_mouth = file_source + 'mouth/' + 'mouth_' + str(option_mouth) + '.png'

im_mouth = cv2.imread(filename_mouth, 0)
im_mouth_new = im_mouth.copy()

#selecting pre-processed upper lip and lower lip segmaps that match each other.
option_lip = st.sidebar.selectbox(' Select Lip ', select_list)
filename_llip = file_source + 'l_lip/' + 'l_lip_' + str(option_lip) + '.png'
filename_ulip = file_source + 'u_lip/' + 'u_lip_' + str(option_lip) + '.png'

im_llip = cv2.imread(filename_llip, 0)
im_llip_new = im_llip.copy()
im_ulip = cv2.imread(filename_ulip, 0)
im_ulip_new = im_ulip.copy()

option_hair = st.sidebar.selectbox(' Select HairStyle ', select_list)
filename_hair = file_source + 'hair/' + 'hair_' + str(option_hair) + '.png'

im_hair = cv2.imread(filename_hair, 0)
im_hair_new = im_hair.copy()

#radio option to move a particular facial attribute
rad_option = st.radio("select feature to move", label_list, 0)

#moves skin and face simultaneously for user ease
if (rad_option == 'skin'):
    x = st.slider('x_step', -512, 512, 0)

    y = st.slider('y_step', -512, 512, 0)

    T = np.float32([[1, 0, x], [0, 1, y]])
    im_skin_new = cv2.warpAffine(im_skin, T, (L, L))
    im_neck_new = cv2.warpAffine(im_neck, T, (L, L))

if (rad_option == 'nose'):
    x = st.slider('x_step', -512, 512, 0)

    y = st.slider('y_step', -512, 512, 0)

    T = np.float32([[1, 0, x], [0, 1, y]])
    im_nose_new = cv2.warpAffine(im_nose, T, (L, L))

#moves left eye and right eye simultaneously for user ease
if (rad_option == 'eye'):
    x = st.slider('x_step', -512, 512, 0)

    y = st.slider('y_step', -512, 512, 0)

    T = np.float32([[1, 0, x], [0, 1, y]])
    im_leye_new = cv2.warpAffine(im_leye, T, (L, L))
    im_reye_new = cv2.warpAffine(im_reye, T, (L, L))

#moves left eyebrow and right eyebrow simultaneously for user ease
if (rad_option == 'brow'):
    x = st.slider('x_step', -512, 512, 0)

    y = st.slider('y_step', -512, 512, 0)

    T = np.float32([[1, 0, x], [0, 1, y]])
    im_lbrow_new = cv2.warpAffine(im_lbrow, T, (L, L))
    im_rbrow_new = cv2.warpAffine(im_rbrow, T, (L, L))

if (rad_option == 'mouth'):
    x = st.slider('x_step', -512, 512, 0)

    y = st.slider('y_step', -512, 512, 0)

    T = np.float32([[1, 0, x], [0, 1, y]])
    im_mouth_new = cv2.warpAffine(im_mouth, T, (L, L))


#moves upperlip and lower lip simultaneously for user ease
if (rad_option == 'lip'):
    x = st.slider('x_step', -512, 512, 0)

    y = st.slider('y_step', -512, 512, 0)

    T = np.float32([[1, 0, x], [0, 1, y]])
    im_ulip_new = cv2.warpAffine(im_ulip, T, (L, L))
    im_llip_new = cv2.warpAffine(im_llip, T, (L, L))

if (rad_option == 'hair'):
    x = st.slider('x_step', -512, 512, 0)

    y = st.slider('y_step', -512, 512, 0)

    T = np.float32([[1, 0, x], [0, 1, y]])
    im_hair_new = cv2.warpAffine(im_hair, T, (L, L))


# generates segmap in greyscale with one hot encoded facial attribute masks
#input for the GAN generator
seg_map1 = func_merge(im_skin_new, im_nose_new, im_leye_new, im_reye_new, im_lbrow_new, im_rbrow_new, im_mouth_new,
                      im_ulip_new, im_llip_new, im_hair_new, im_neck_new)

im_base = np.zeros((512, 512, 3))

#colorises the segmap for display
for idx, color in enumerate(color_list):
    im_base[seg_map1 == idx] = color


st.image(im_base, caption='Segmented map', width=256, clamp=True)

#colorises the segmap for display
if st.button('segmap_generate'):
    filename_seg_path = '..path to save segmap./seg.jpg'
    seg_map = func_merge(im_skin_new, im_nose_new, im_leye_new, im_reye_new, im_lbrow_new, im_rbrow_new, im_mouth_new,
                         im_ulip_new, im_llip_new, im_hair_new, im_neck_new)
    cv2.imwrite(filename_seg_path, seg_map)
    st.write("Seg_map ready")

    img_file_source = '..path to test_image file../'
    img_filename = img_file_source + 'filename'

    img_file_target = '..copy test_image file to new location../'
    img_filename_target = img_file_target + 'filename'
    shutil.copy(img_filename, img_file_target)
    os.rename(img_filename_target, img_file_target + 'seg.jpg') #reame test image to seg.jpg to maintain convention of same names from image and segmented maps
    img_filename_target = '..path to final test_image file../test_image2/seg.jpg'

filename_seg_path = 'path to segmap../seg.jpg'
img_filename_target = '..path to test_image file../'

if st.button('Convert'):
    os.system(
        "python test.py --gpu_ids 0 --name celeb_1 --dataset_mode custom --no_instance --preprocess_mode 'resize_and_crop' --label_dir datasets/celeb_data/test_label2 --image_dir datasets/celeb_data/test_image2/ --load_size 128 --aspect_ratio 1 --crop_size 128 --label_nc 19")

    filename_result = '/home/ubuntu/AWS_spade/SPADE/results/celeb_1/test_latest/images/synthesized_image/seg.png'
    image2 = Image.open(filename_result)

    st.image(image2, caption='Result', width=256)

filename_result = '/home/ubuntu/AWS_spade/SPADE/results/celeb_1/test_latest/images/synthesized_image/seg.png'

filename_result_label = '/home/ubuntu/AWS_spade/SPADE/results/celeb_1/test_latest/images/input_label/seg.png'

if st.button('Clear Memory'):
    os.remove(filename_result)
    os.remove(filename_result_label)
    os.remove(filename_seg_path)
    os.remove(img_filename_target)
