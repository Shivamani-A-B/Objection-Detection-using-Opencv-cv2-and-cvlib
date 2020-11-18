#!/usr/bin/env python
# coding: utf-8

# # Object Detection using OpenCV (cv2)

# In[6]:


#Install the necessary package before proceeding the next cells for object detection
#pip install cvlib
#pip install cv2
#pip install opencv-python
#pip install matplotlib


# # Import the libraries

# In[8]:


import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox


# # Reading the Image and Executing the Object Detection

# In[12]:


#Read the image 

image = cv2.imread('Downloads/testimage.jpeg')

#using cvlib we can process the object detection at ease. we are using matplotlib to display the output within the notebook.

bbox, label, conf = cv.detect_common_objects(im)
output_image = draw_bbox(image, bbox, label, conf)
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111)
ax.imshow(image,cmap='gray')


# In[ ]:




