#!/usr/bin/env python
# coding: utf-8

# # Object Detection using OpenCV (cv2)

# # Packages to be installed

# In[1]:


#Install the necessary package before proceeding the next cells for object detection
#pip install cvlib
#pip install cv2
#pip install opencv-python
#pip install matplotlib


# # Import the libraries

# In[2]:


import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox


# # Reading the Image 

# In[3]:


#Read the image 

image = cv2.imread('Downloads/testimage.jpeg')
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111)
ax.imshow(image)


# # Implementing the Object Detection using cvlib

# In[4]:


#using cvlib we can process the object detection at ease. we are using matplotlib to display the output within the notebook.

bbox, label, conf = cv.detect_common_objects(image)
output_image = draw_bbox(image, bbox, label, conf)
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111)
ax.imshow(image)


# In[ ]:




