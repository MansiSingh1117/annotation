# -*- coding: utf-8 -*-
# Import required libraries

import cv2
import numpy as np
import streamlit as st
import requests
import os
import functools

import matplotlib.pylab as plt

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
    tabs = st.sidebar.selectbox(
        'Choose one of the following',
        ('Cartoonized Image','Resize Image'),
        key="main_menu"
    )
 
# UI Options  
    
    cartoonization()

   

def uploadImage(key, new_height=480):

    uploaded_file = st.file_uploader("Choose a Image file",key=key)
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        # Pre-processing image: resize image
        return preProcessImg(img, new_height)
    
    return cv2.cvtColor(preProcessImg(cv2.imread('natalie.jpg'),new_height),cv2.COLOR_BGR2RGB)

 

def preProcessImg(img, new_height=480):
    # Pre-processing image: resize image
    #img = cv2.resize(img,(300,300))
    return img



    
def cartoonization():
    st.header("Cartoonized Image")

    img = uploadImage("annotation_img")
    #Bilateral Blurring
    img1b=cv2.bilateralFilter(img,3,75,75)
    plt.imshow(img1b,cmap='gray')
    plt.axis("off")
    plt.title("AFTER BILATERAL BLURRING")
    plt.show()

    img_grey=cv2.cvtColor(img1b,cv2.COLOR_BGR2GRAY)

#Creating edge mask
    #edges=cv2.adaptiveThreshold(img_grey,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,3,3)
    #plt.imshow(edges,cmap='gray')
    #plt.axis("off")
    #plt.title("Edge Mask")
    #plt.show()



#Eroding and Dilating
    kernel=np.ones((3,3),np.uint8)
    img1e=cv2.erode(img1b,kernel,iterations=5)
    img1d=cv2.dilate(img1e,kernel,iterations=5)
    plt.imshow(img1d,cmap='gray')
    plt.axis("off")
    plt.title("AFTER ERODING AND DILATING")
    plt.show()
    
 

#Clustering - (K-MEANS)
    imgf=np.float32(img).reshape(-1,3)
    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,20,1.0)
    compactness,label,center=cv2.kmeans(imgf,5,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center=np.uint8(center)
    final_img=center[label.flatten()]
    final_img=final_img.reshape(img.shape)
    plt.imshow(final_img,cmap='gray')

    #final=cv2.bitwise_and(final_img,final_img,mask=edges)
    #plt.imshow(final,cmap='gray')
    #plt.axis("off")
    plt.savefig('output1', bbox_inches='tight')
    st.image(final_img)

    plt.show()
    




if __name__ == "__main__":
    main()
