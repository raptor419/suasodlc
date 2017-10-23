import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL as pil
import argparse
import sys
import glob
import imutils
import detect

##ap = argparse.ArgumentParser()
##ap.add_argument("-i","--images", required=True, help="path to input dataset of images")
##args=vars(ap.parse_args())

processed_pics=list()

while True:
    i = 0
    for impath in glob.glob("rsync/*.jpg"):
        i += 1
        img=cv2.imread(impath)
        im=cv2.Canny(img,200,200)
        onimg=img.copy()
        oimg=img.copy()
        outimg=img.copy()
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
        #blurred = cv2.Canny(blurred,100,100)
        ret,thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_TOZERO)
        #thresh = cv2.fastNlMeansDenoising(thresh,None)
        _, cnts, heier = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        n = 0
        cnt=list()
        for c in cnts:
            area = cv2.contourArea(c)
            if area>1000:
                cnt.append(c)
                if detect.rdetect(c):
                    n+=1
                    cv2.drawContours(oimg,c, -1, (0, 255, 0), 10)
                    x,y,w,h = cv2.boundingRect(c)
                    cimg = onimg[y-min(y,h//4,50):y+h+min(y,h//4,50), x-min(x,w//4,50):x+w+min(x,w//4,50)]
                    cv2.imwrite('output/json/'+str(i)+' - '+str(n)+'.jpg',cimg)
                    M = cv2.moments(c)
                    cX = int((M["m10"] / M["m00"]))
                    cY = int((M["m01"] / M["m00"]))
                    shape = detect.detect(c)
                    sshape = detect.suredetect(cimg)
                    if sshape!=None:
                        cv2.drawContours(outimg,c, -1, (0, 255, 0), 10)
                        cv2.putText(outimg, sshape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)
                        cv2.imwrite('output/off/json/'+str(i)+' - '+str(n)+'.jpg',cimg)
                    cv2.putText(oimg, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)
                    # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
##        cv2.namedWindow("Image",cv2.WINDOW_NORMAL)
##        cv2.resizeWindow("Image", 600,600)
##        cv2.imshow("Image", oimg)
##        cv2.waitKey(0)
        cv2.imwrite('output/'+str(i)+'.jpg',oimg)
        cv2.imwrite('output/off/'+str(i)+'.jpg',outimg)
    break
detect.sdetect()

        
        
        
