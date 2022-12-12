# -*- coding: utf-8 -*-
# Import required libraries

import cv2
import numpy as np
import streamlit as st
import requests
import os
import functools
from matplotlib import gridspec
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
import sys
import imageio

# Home UI 

def main():

    st.set_page_config(layout="wide")

    font_css = """
        <style>
        button[data-baseweb="tab"] {
        font-size: 26px;
        }
        </style>
        """

    st.write(font_css, unsafe_allow_html=True)
    stleTransfer()


def uploadImage(key, new_height=480):

    uploaded_file = st.file_uploader("Choose a Image file",key=key)
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        # Pre-processing image: resize image
        return preProcessImg(img, new_height)
    
    return cv2.cvtColor(preProcessImg(cv2.imread('sample.jpg'),new_height),cv2.COLOR_BGR2RGB)

 # UI Options  
    if tabs == 'Annotate Image':
        cartoonization()
    if tabs == 'Resize Image':
        resizeImg()

def preProcessImg(img, new_height=480):
    # Pre-processing image: resize image
    img = cv2.resize(img,(256,256))
    return img



    
def cartoonization():
    st.header("Annotate Image")

    img = uploadImage("annotation_img")
    #Bilateral Blurring
    img1b=cv2.bilateralFilter(img1g,3,75,75)
    plt.imshow(img1b,cmap='gray')
    plt.axis("off")
    plt.title("AFTER BILATERAL BLURRING")
    plt.show()

#Creating edge mask
    edges=cv2.adaptiveThreshold(img1b,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,3,3)
    plt.imshow(edges,cmap='gray')
    plt.axis("off")
    plt.title("Edge Mask")
    plt.show()

#Eroding and Dilating
    kernel=np.ones((3,3),np.uint8)
    img1e=cv2.erode(img1b,kernel,iterations=5)
    img1d=cv2.dilate(img1e,kernel,iterations=5)
    plt.imshow(img1d,cmap='gray')
    plt.axis("off")
    plt.title("AFTER ERODING AND DILATING")
    plt.show()

#Clustering - (K-MEANS)
    imgf=np.float32(img1).reshape(-1,3)
    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,20,1.0)
    compactness,label,center=cv2.kmeans(imgf,5,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center=np.uint8(center)
    final_img=center[label.flatten()]
    final_img=final_img.reshape(img1.shape)

    final=cv2.bitwise_and(final_img,final_img,mask=edges)
    plt.imshow(final,cmap='gray')
    plt.axis("off")
    plt.savefig('output1', bbox_inches='tight')

    plt.show()
    
def resizeImg():
    st.header("Resize Image")

    img = uploadImage("resize_img")

    scaleFactor = st.slider("Times Image",1,10,1,1,key='resize')/5
    scaledImg = cv2.resize(img, None, fx=scaleFactor, fy = scaleFactor, interpolation = cv2.INTER_LINEAR)

    st.image(scaledImg)
    pass



if __name__ == "__main__":
    main()
