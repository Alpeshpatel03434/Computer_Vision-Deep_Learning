#!/usr/bin/env python
# coding: utf-8

# # Social Distancing detector

# In[ ]:





# ## Importing the libraries

# In[1]:


import math
import numpy as np
import cv2 as cv

import imutils
from itertools import combinations


# ## Centroid and Distance

# In[2]:


def Distance(p1, p2):
    
    dist = math.sqrt(p1**2 + p2**2)
    
    return dist


# In[3]:


def Centroid(x, y, w, h): 
    
    cx = int(x+w/2.0)
    cy = int(y+h/2.0)
    
    return cx, cy


# ## Model MobileNetSSD  (Single Shot MultiBox Detector)
# 
# ### Pre-trained model For Object Detection

# **MobileNetSSD Model Used for Persone OR Human Detection**

# **Link to Download :** https://github.com/chuanqi305/MobileNet-SSD

# In[4]:


protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"

detector = cv.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)


# **We need only one Object**

# In[5]:


Classes = "person"


# ## Social Distancing Detection Function 

# In[6]:


def detect(input_filename):

    cap = cv.VideoCapture(input_filename)

    while True:
        ret, frame = cap.read()
        
        if (ret!=True):
            break
            
        frame = imutils.resize(frame, width=950, height=1350)
        
        H = frame.shape[0]
        W = frame.shape[1]
        
        blob = cv.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        detector.setInput(blob)
        person_detections = detector.forward()
        
        centroid_dict = dict()
        objectId = 0
        
        rect_box = []
        red_zone_list = []
            
        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(person_detections[0, 0, i, 1])

                if Classes != "person":
                    continue

                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")

                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                startX, startY, endX, endY = person_box.astype("int")
                
                
                cx, cy = Centroid(int(startX), int(startY), int(endX), int(endY))  
                
                centroid_dict[objectId] = (int(cx), int(cy), startX, startY, endX, endY)
                objectId = objectId + 1      
                 
                text = "Persone"
                cv.putText(frame, text, (startX, startY-5), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
                
                for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2):
                    dx, dy = p1[0] - p2[0], p1[1] - p2[1]
                    
                    distance = Distance(dx, dy)
                    if distance < 75.0:
                        if id1 not in red_zone_list:
                            red_zone_list.append(id1)
                        if id2 not in red_zone_list:
                            red_zone_list.append(id2)
                            
                for id, box in centroid_dict.items():
                    
                    # Color code BGR Formate
                    if id in red_zone_list:
                        # Red                   
                        cv.rectangle(frame, (box[2], box[3]), (box[4], box[5]), (0, 0, 255), 2)
                    
                    else :
                        # Green
                        cv.rectangle(frame, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2)
                    
                        

        
        cv.imshow("Social Distancing detector", frame)
        key = cv.waitKey(1)
        if key == ord('q') or key == 0:
            break
     
    cv.destroyAllWindows()


# In[ ]:





# ## Current Direcotry

# In[7]:


import os


# In[ ]:


print(os.getcwd())


# ## Input filename 

# In[8]:


#input_filename = 'test_video.mp4'

input_filename = 'test/Video3.mp4'


# ## Check file avalible or not

# In[9]:


if not os.path.isfile(input_filename):
    raise FileNotFoundError


# ## Output

# In[10]:


detect(input_filename)


# ### Green rectangle determine scocial distancing maintain
# 
# ### Red rectangle determine social distancing Viollance

# In[ ]:




