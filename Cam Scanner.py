#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


im_path="./bill2.jpg"
#read image from path
img=cv2.imread(im_path)
print(img.shape)
#image resize
img=cv2.resize(img,(1500,800))
print(img.shape)
plt.imshow(img)
plt.show()


# # preprocessing Image
# *Remove the noise
# *edge detection
# *contour selection
# *best contour selection
# *project to screen 

# In[3]:


#remove the noise
##Image blurring

orig=img.copy()
gray=cv2.cvtColor(orig,cv2.COLOR_BGR2GRAY)
plt.imshow(gray,cmap="binary")
plt.show()

blurred=cv2.GaussianBlur(gray,(5,5),0)
plt.imshow(blurred,cmap="binary")
plt.show()


# In[4]:


regen=cv2.cvtColor(blurred,cv2.COLOR_GRAY2BGR)
plt.imshow(orig,cmap="binary")
plt.show()

plt.imshow(regen,cmap="binary")
plt.show()


# In[5]:


regen.shape


# In[6]:


##Edge Detection

edge=cv2.Canny(blurred,0,50)
orig_edge=edge.copy()
 
plt.imshow(orig_edge)
plt.title("Edge Detection")
plt.show()


# In[7]:


##Countours Extraction

contours,_=cv2.findContours(edge,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
print(len(contours))

contours=sorted(contours,reverse=True,key=cv2.contourArea)


# In[8]:


#select the best contour Region

for c in contours:
    p=cv2.arcLength(c,True)
    
    approx=cv2.approxPolyDP(c,0.01*p,True)
    
    if len(approx) == 4:
        target=approx
        break
print(target.shape)
        


# In[9]:


#Reorder target Contor
def reorder(h):
    
    h=h.reshape((4,2))
    print(h)
    
    hnew=np.zeros((4,2),dtype=np.float32)
    
    add=h.sum(axis=1)
    hnew[3]=h[np.argmax(add)]
    hnew[1]=h[np.argmax(add)]

    diff=h.sum(axis=1)
    hnew[0]=h[np.argmax(diff)]
    hnew[2]=h[np.argmax(diff)]
    return hnew


# In[10]:


reorder=reorder(target)
print("*********")
print(reorder)


# In[11]:


##project to a fixed screen

input_representation=reorder
output_map=np.float32([[0,0],[800,0],[800,800],[0,800]])


# In[12]:


M= cv2.getPerspectiveTransform(input_representation,output_map)

ans=cv2.warpPerspective(orig,M,(800,800))


# In[13]:


plt.imshow(ans)
plt.title("Edge Detection")
plt.show()


# In[14]:


res=cv2.cvtColor(ans,cv2.COLOR_BGR2GRAY)

b_res=cv2.GaussianBlur(res,(3,3),0)
plt.imshow(res,cmap="binary")
plt.show()

plt.imshow(b_res,cmap="binary")
plt.show()


# In[ ]:




