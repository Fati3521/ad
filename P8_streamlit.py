
from flask import Flask, render_template, request, send_from_directory, send_file
import requests
import json

from PIL import Image
import numpy as np 
import pandas as pd
import os
from azureml.core import Workspace, Datastore, Dataset
from tensorflow import keras
import streamlit as st
import pickle
from joblib import dump, load
from PIL import Image
from skimage.transform import resize
import cv2
import time



def LayersToRGBImage(img):
    colors = [(0, 0, 0 ), (128, 64, 128), (150, 100, 100),
             (220, 220, 0), (107, 142, 35), (70, 130, 180),
             (220, 20, 60), (119, 11, 32)]

    nimg = np.zeros((img.shape[0], img.shape[1], 3))
    for i in range(img.shape[2]):
        c = img[:,:,i]
        col = colors[i]
        
        for j in range(3):
            nimg[:,:,j]+=col[j]*c
    nimg = nimg/255.0
    return nimg



def label2cat(v):
    try:
        my_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:1, 8:1, 9:1, 10:1, 11:2,
          12:2, 13:2, 14:2, 15:2, 16:2, 17:3, 18:3, 19:3, 20:3, 21:4, 22:4,
          23:5, 24:6, 25:6, 26:7, 27:7, 28:7, 29:7, 30:7, 31:7, 32:7, 33:7}
   
        fct = np.vectorize(my_dict.get)
        b = fct(v)
        img_pil = Image.fromarray(b.astype(np.uint8))       
        return img_pil
    except IndexError:
        return 0 
    
    
st.title("Car Segmentation App")
st.write('\n') 
# For newline
st.write('\n')
image = Image.open('image.png')
st.write('\n')

width = 512
height = 256
dim = (width, height)
show = st.image(image, use_column_width=True)

st.sidebar.title("Upload Image")
#Disabling warning
st.set_option('deprecation.showfileUploaderEncoding', False)

#Choose your own image
rep = requests.get("http://127.0.0.1:5000/api/get_img_list/")
data = rep.json()
noms_img = data['data']
image = st.sidebar.selectbox('Select an Option', noms_img, index=0)  


if image is not None: 
    st.text(image)
    rep = requests.post("http://127.0.0.1:5000/api/download_image/", json={'image': image})

    imgs = rep.json()['data']
    
    img_raw = imgs['raw']
    img_mask = imgs['mask']
    raw_img = Image.open(img_raw)
    mask_img = Image.open(img_mask)
             
    # On affiche l'image RGB choisie
    show.image(raw_img, 'Uploaded Image', use_column_width=True)
    
    ##################
    ##### LABEL TRUE #
    # changement des catégories
    image_label = label2cat(mask_img)
    # on resize 
    temp_image_label = image_label.resize((256, 128), Image.NEAREST)
    temp_image_label_array = np.asarray(temp_image_label)
    print(temp_image_label_array.shape)
    
    temp_image_label_array = keras.utils.to_categorical(temp_image_label_array, num_classes=8)
    print(temp_image_label_array.shape)
    
    img_color = LayersToRGBImage(temp_image_label_array)



predict_button = st.button('predict')
if predict_button:
    rep = requests.post("http://127.0.0.1:5000/api/predict/", json={'image': image})
    #st.text(rep)
    img_pred = rep.json()['data']
    pred_mask = Image.open(img_pred)
    

    images_list = [img_color, pred_mask]
    caption = ['True Label', 'Predict Label']
    st.image(images_list, width=512, caption=caption)
    
    legend = Image.open('Legend.jpg')
    show = st.image(legend, width=None)

        
