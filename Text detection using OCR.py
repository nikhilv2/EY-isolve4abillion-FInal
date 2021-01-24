import cv2
import numpy as np
import pytesseract
import os
import re
import pandas as pd
# C:\Program Files\Tesseract-OCR
per = 25

pixelThreshold=250
roi = [[(203, 106), (433, 141), 'text', 'Aadhar Number'], [(156, 168), (434, 204), 'text', 'Name'],
       [(156, 237), (431, 270), 'text', 'District'], [(76, 339), (98, 353), 'box', 'Male'],
       [(203, 337), (225, 352), 'box', 'Female'], [(329, 338), (351, 353), 'box', 'Others'],
       [(76, 401), (97, 419), 'box', '0-18'], [(204, 402), (226, 417), 'box', '19-35'],
       [(328, 403), (351, 416), 'box', '36-50'], [(440, 404), (463, 416), 'box', '51-60'],
       [(537, 403), (558, 417), 'box', '60 above'], [(75, 466), (97, 475), 'box', 'Obese yes'],
       [(227, 466), (246, 479), 'box', 'Obese No'], [(76, 528), (100, 543), 'box', 'Heart Disease yes'],
       [(228, 527), (249, 540), 'box', 'Heart Disease No'], [(77, 591), (101, 604), 'box', 'Lung problems Yes'],
       [(230, 592), (253, 607), 'box', 'Lung Problems No'], [(77, 655), (102, 669), 'box', 'Muscle Pain'],
       [(244, 655), (271, 669), 'box', 'Tiredeness'], [(371, 656), (395, 670), 'box', 'Headache'],
       [(488, 655), (511, 669), 'box', 'Mild Allergies'], [(76, 685), (99, 701), 'box', 'Fever'],
       [(178, 686), (201, 700), 'box', 'None'], [(75, 751), (99, 764), 'box', 'Covid Positive'],
       [(225, 750), (249, 767), 'box', 'Covid Negative']]

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
imgQ = cv2.imread('Query Final.png')
h,w,c = imgQ.shape
#imgQ = cv2.resize(imgQ,(w//1,h//1))
orb = cv2.ORB_create(1000)
kp1, des1 = orb.detectAndCompute(imgQ,None)
#impKp1 = cv2.drawKeypoints(imgQ,kp1,None)
path = 'C:\\Users\\nikhi\\PycharmProjects\\EYFINAL\\UserForms'
myPicList = os.listdir(path)
print(myPicList)
for j,y in enumerate(myPicList):
    img = cv2.imread(path +"/"+y)
    img = cv2.resize(img, (w // 1, h // 1))
    #cv2.imshow(y, img)
    kp2, des2 = orb.detectAndCompute(img,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2,des1)
    matches.sort(key= lambda x: x.distance)
    good = matches[:int(len(matches)*(per/100))]
    imgMatch = cv2.drawMatches(img,kp2,imgQ,kp1,good[:20],None,flags=2)
    #cv2.imshow(y, img)
    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(srcPoints,dstPoints,cv2.RANSAC,2.0)
    imgScan = cv2.warpPerspective(img,M,(w,h))
    cv2.imshow(y, imgScan)
    imgShow = imgScan.copy()
    imgMask = np.zeros_like(imgShow)

    myData = []

    print(f'################## Extracting Data from Form {j}  ##################')

    for x,r in enumerate(roi):
        cv2.rectangle(imgMask, (r[0][0],r[0][1]),(r[1][0],r[1][1]),(0,255,0),cv2.FILLED)
        imgShow = cv2.addWeighted(imgShow,0.99,imgMask,0.1,0)

        imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        #cv2.imshow(str(x), imgCrop)

        if r[2] == 'text':
            print('{} :{}'.format(r[3],pytesseract.image_to_string(imgCrop)))
            myData.append(pytesseract.image_to_string(imgCrop))
        if r[2] =='box':
            imgGray = cv2.cvtColor(imgCrop,cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imgGray,17,255, cv2.THRESH_BINARY_INV)[1]
            totalPixels = cv2.countNonZero(imgThresh)
            if totalPixels>pixelThreshold: totalPixels =1;
            else: totalPixels=0
            print(f'{r[3]} :{totalPixels}')
            myData.append(totalPixels)
        cv2.putText(imgShow,str(myData[x]),(r[0][0],r[0][1]),
                    cv2.FONT_HERSHEY_PLAIN,2.5,(0,0,255),4)
    with open('DataOutput1.csv','a+') as f:
        for data in myData:
            f.write((str(data)+','))
        f.write('\n')


    #imgShow = cv2.resize(imgShow, (w // 3, h // 3))
    print(myData)




    cv2.imshow(y,imgShow)



#cv2.imshow("KeyPoints", impKp1)
cv2.imshow("Output",imgQ)
cv2.waitKey(0)
