import os
import cv2
import glob
import numpy as np

label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

folder_base = '/Users/preethasaha/PycharmProjects/Insight_pr/AWS_spade/SPADE/datasets/CelebAMask-HQ/CelebAMask-HQ-mask-anno'
folder_save = '/Users/preethasaha/PycharmProjects/Insight_pr/AWS_spade/SPADE/datasets/annotations/one_hot'


img_num = 24000



for k in range(img_num):
  print(k)

  
  folder_num = k // 2000
  im_base = np.zeros((512, 512))
  for idx, label in enumerate(label_list):
    filename = os.path.join(folder_base,str(folder_num),str(k).rjust(5, '0') + '_' + label + '.png')
    #print (filename)
    if (os.path.exists(filename)):



      #print (label, idx+1)
      #print (filename)
      im = cv2.imread(filename)
      im = im[:, :, 0]
      im_base[im != 0] = (idx + 1)
  filename_save = os.path.join(folder_save,folder_save, str(k) + '.jpg')
  cv2.imwrite(filename_save, im_base)

