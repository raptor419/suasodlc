import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL as pil
import argparse
import sys
import glob
import imutils
import detect
import random

def detect(c):
    shape = "unidentified"
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    if len(approx) == 3:
        shape = "triangle"
    elif len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
    elif len(approx) == 5:
        shape = "pentagon"
    elif len(approx) == 6:
        shape = "hexagon"
    elif len(approx) == 7:
        shape = "heptagon"
    elif len(approx) == 8:
        shape = "octagon"   
    else:
        shape = str(approx)
    return shape

def rdetect(c):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    if len(approx) <= 8:
        return True
    else:
        return False

def sdetect():
    for impath in glob.glob("output/json/*.jpg"):
        img=cv2.imread(impath)
        im=cv2.Canny(img,200,200)
        onimg=cv2.imread(impath)
        oimg=cv2.imread(impath)
        hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue,sens=60,40
        lowergreen=np.array([hue-sens,50,50])
        uppergreen=np.array([hue+sens,255,255])
        mask=cv2.inRange(hsv, lowergreen, uppergreen)
        img=cv2.bitwise_and(img,img,mask=cv2.bitwise_not(mask))
        img=cv2.bitwise_and(img,img,mask=cv2.bitwise_not(im))
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret,threshed = cv2.threshold(frame,127,255,cv2.THRESH_TOZERO)
        img=cv2.bitwise_and(img,img,mask=cv2.bitwise_not(threshed))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        ret,thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_TOZERO)
        _, cnts, heier = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        n = 0
        cnt=list()
        for c in cnts:
            area = cv2.contourArea(c)
            if area>100:
                cnt.append(c)
                shape = detect(c)
                n+=1
##        haar = cv2.CascadeClassifier('E://stage2.xml')
##        if haar.detectMultiScale(gray, 1.3, 5):
##            shape="person"
    return shape

def badsuredetect(cimg):
    val=list()
    shape=list()
    cimg = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
    for impath in glob.glob("data/temp/output/*.png"):
        timg = cv2.imread(impath)
        timg = cv2.cvtColor(timg, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(cimg,timg,cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        val.append(max_val)
        shape.append(impath[impath.rfind('/'):impath.rfind('.')])
    if max(val)>0.8:
        nshape=shape[val.index(max(val))]
    else:nshape=None
    return nshape

def suredetect(cimg):
    minmatch = 10
    img1 = cimg
    val=list()
    shape=list()
    for impath in glob.glob("data/temp/*.png"):
        img2 = cv2.imread(impath, cv2.IMREAD_COLOR)
        #img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1,None)
        kp2, des2 = orb.detectAndCompute(img2,None)
##        print(len(kp2),len(des2))
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
##        img3 = cv2.drawKeypoints(img2,kp2, None,color=(0,255,0), flags=0)
##        plt.imshow(img3),plt.show()
        des2=random.sample(list(des2),80)
        try:des1=list(des1)
        except:des1=des2[0:3]
        matches = bf.match(np.array(des1),np.array(des2))
        val.append(len(list(matches)))
        shape.append(impath)
    if max(val)>minmatch:
        nshape=shape[val.index(max(val))]
    else:nshape=None
    return nshape
    
    
        
