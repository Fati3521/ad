from __future__ import print_function
from flask import Flask, render_template, request, send_from_directory, send_file
import requests
from flask import jsonify
from PIL import Image
import numpy as np 
import pandas as pd
import os
from azureml.core import Workspace, Datastore, Dataset
import json

import sys
import streamlit as st
import pickle
from joblib import dump, load
from PIL import Image
from skimage.transform import resize
import cv2
import time
import re
import glob
from scipy import misc
from PIL import Image


# telechargement du modele entrainé sur azure
import segmentation_models as sm
sm.set_framework('tf.keras')

import tensorflow as tf
from tensorflow import keras


CLASS_WEIGHTS = None
metrics = [sm.metrics.IOUScore(class_weights=CLASS_WEIGHTS), sm.metrics.FScore(class_weights=CLASS_WEIGHTS)]
loss = sm.losses.DiceLoss(class_weights=CLASS_WEIGHTS)
opt = keras.optimizers.Adam(learning_rate=0.001)


# Loading the saved model
model_train = tf.keras.models.load_model('../model', custom_objects = {"dice_loss" : loss,
                                                                      "val_iou_score": sm.metrics.IOUScore(class_weights=CLASS_WEIGHTS),
                                                                      "iou_score" : sm.metrics.IOUScore(class_weights=CLASS_WEIGHTS), 
                                                                      "f1-score": sm.metrics.FScore(class_weights=CLASS_WEIGHTS),
                                                                      "val_f1-score": sm.metrics.FScore(class_weights=CLASS_WEIGHTS)   })

    

    
app = Flask(__name__)


########################
# FONCTIONS IMAGES
########################
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


# si on n'utilise pas Streamlit
########################
# DASHBOARD 
########################
@app.route('/')
def dashboard():
    print('>>>>>>>>>>>>>>> /')
    salutation = 'Thank for using flask-fundamentum!'
    return render_template('home.html', msg=salutation)

                                      
########################
# RETOURNE LISTE DES IMAGES  # retourne la liste des noms des images disponibles pour le test en JSON
########################
@app.route("/api/get_img_list/",  methods= ['GET'])
def listimgs():
    print( '>>>>>>>>>>>>>>> /api/get_img_list')
    print( '\t >>>>>>>>>>>>>>> 1')
    
    val_files = np.load('./static/val_files_test.npy')

    print(' \t >>>>>>>>>>>>>>> 2')
    
    response =  {'status': 'ok', 'data': val_files.tolist()}
    print('app.py - get_img_list', response)

    return jsonify(response)    



################################
# TELECHARGER IMAGE RAW EN LOCAL
################################
@app.route("/api/download_image/", methods=['POST'])
def download_image():
    print('>>>>>>>>>>> /api.download_image')
    print('\t >>>>>>>>>>>>>>> 1 download image')
    
    data = request.get_json()
    image = data["image"]
    print(image)
      
    print('\t >>>>>>>>>>>>>>> 2 download image')
    
    image_path = glob.glob('../../data/data2/data2/leftImg8bit/test/' + image + '_img.png')
    dset_raw = Image.open(image_path[0])
    
    image_path_gt = glob.glob('../../data/data2/data2/gtFine/test/' + image + '_mask.png')
    dset_gt = Image.open(image_path_gt[0])
        
          
    print('\t >>>>>>>>>>>>>>> 3')
    data = {
        'raw':"../../data/data2/data2/leftImg8bit/test/" + image + '_img.png',
        'mask':"../../data/data2/data2/gtFine/test/" + image + '_mask.png'
        }
                                      
    response = {'status': 'ok', 'data':data}
    return jsonify(response)


##################################
# TELECHARGER IMAGE RAW SOUS AZURE
##################################
@app.route("/api/download_image/", methods=['POST'])
def download_image2():
    print('>>>>>>>>>>> /api.download_image')
    
    print('\t >>>>>>>>>>>>>>> 1')
    data = request.get_json()
    image = data["image"]
      
    print('\t >>>>>>>>>>>>>>> 2')
    
    subscription_id = 'e7c8495b-647b-464c-8fc6-74fad7e47bb1'
    resource_group = 'gpe-ressource-lab1'
    workspace_name = 'test-workspace3' 

    ws = Workspace(subscription_id, resource_group, workspace_name)
    dstore = Datastore.get(ws, 'workspaceblobstore')  
    # on va chercher l'image brute dans le dossier Azure
    dset_raw = Dataset.File.from_files(path=[(dstore, ('data_folder2/leftImg8bit/val/frankfurt/' + image + '_leftImg8bit.png'))])
    dset_gt = Dataset.File.from_files(path=[(dstore, ('data_folder2/gtFine/val/' + image + '_gtFine_labelIds.png' ))])     
    print(dset_raw)
    print(dset_gt)
           
                     
    print('\t >>>>>>>>>>>>>>> 3')
    temp_dir_raw = dset_raw.download(os.getcwd()+ '/static/images/raw/', overwrite = True)
    temp_dir_gt = dset_gt.download(os.getcwd()+ '/static/images/mask/', overwrite = True)                                  
    
    print('\t >>>>>>>>>>>>>>> 4')
    data = {
        'raw':"./static/images/raw/" + image + '_leftImg8bit.png',
        'mask':"./static/images/mask/" + image + '_gtFine_labelIds.png'
        }
                                      
    response = {'status': 'ok', 'data':data}
    return jsonify(response)



###########
# PREDICT   # La requête predict renvoie l’adresse où l’image prédite est stoquée
###########
                                      
@app.route("/api/predict/", methods=['POST'])
def predict(): 
                                      
    print('>>>>>>>>>>> /api/predict')
    print('\t >>>>>>>>>>>>>>> 1 predict')
    data = request.get_json()
    image = data["image"]
    print(image)
    
    
    print('\t >>>>>>>>>>>>>>> 2 predict')
    

    ###############
    ## SOUS AZURE
    ##############
    
    #headers = {'Content-Type': 'application/json', 'Cache-Control': 'no-cache'}
    #print(headers)
    # adresse du endpoint:
    #resp = requests.post('http://3b4d0601-2904-4b29-a504-b173f8a49ad4.westeurope.azurecontainer.io/score', input_data, headers=headers)   
   # print('resp: ', resp)
     
    ###############
    ## EN LOCAL
    ##############
    image_path = "../../data/data2/data2/leftImg8bit/test/" + image + '_img.png'
    input_data = json.dumps({"data" : image })   
    
    
    print('\t >>>>>>>>>>>>>>> 3 predict') 
    ###############
    ## SI MODELE EN LOCAL (model_train)
    ##############
    # Initialization
    X = Image.open(image_path)
    X = X.resize((128, 64), Image.BILINEAR)
    X = np.array(X)
    print(X.shape)
    X = np.expand_dims(X, axis=0)
    print(X.shape) 
    
    pred_mask = model_train.predict(X)
    prediction_array = LayersToRGBImage(np.squeeze(pred_mask, axis=0))

    new_image = tf.keras.preprocessing.image.array_to_img(prediction_array)
    new_image.save('./static/images/predict/'+ image + '_predict.png', 'PNG') 
    
      
    ##################
    ### MODEL VIA ENDPOINT
    ##################
    #print(resp.json())
    #new_image = tf.keras.preprocessing.image.array_to_img(np.array(json.loads(resp.json())))
    #new_image = Image.fromarray(np.array(json.loads(resp.json()), dtype='uint8'))
    #prediction = json.loads(resp.json())['imagepred']
    #print(prediction)
    #new_image.save('./static/images/predict/'+ image + '_predict.png', 'PNG') 
 

    # Repertoire static: stoque de manière temporaire le fichier segmenté.                                       
    print('\t >>>>>>>>>>>>>>> 5') 
    response = {'status': 'ok', 'data': "./static/images/predict/" + image + '_predict.png'}
    return jsonify(response)


###########
# CLEAN # La fonction clean supprime tous les répertoires
###########                                      
#@app.route('/api/clean/')
#def clean():                                      
                                              
    
if __name__ == "__main__":
    app.run()
    
 
    


    
