import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import ndimage

import os,sys
dir = os.path.dirname(__file__)
filename = os.path.join(dir, 'images')
print("--------------=-=-=-=-=-=-  " + str(filename))

                            #C:/Users/Smekac/Desktop/pr/ImagesOfCars        #Paziti apsolutna putanja
path, dirs, files = os.walk(filename).__next__()
file_count = len(files)
print("Broj slika je : " + str(file_count) )

# index = 1
# while index <= file_count:    //za rad sa svim slikama


def ZaDetekciju(Okvir):
    img_GRAY_gs = cv2.cvtColor(Okvir, cv2.COLOR_RGB2GRAY)  # konvert u grayscale
    plt.imshow(img_GRAY_gs, 'gray')
    image_barcode_bin = cv2.adaptiveThreshold(img_GRAY_gs, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 61,15)
    ####
    retSlike, image_bin = cv2.threshold(img_GRAY_gs, 0, 255, cv2.THRESH_OTSU)
    print("pRAG JE: " + str(retSlike))
    img, contours, hierarchy = cv2.findContours(image_barcode_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img = Okvir.copy()
    cv2.drawContours(img, contours, -1, (255, 0, 0), 1)

    contours_Tablica = []  # ovde ce biti samo konture koje pripadaju bar-kodu
    for contour in contours:  # za svaku konturu
        center, size, angle = cv2.minAreaRect(
            contour)  # pronadji pravougaonik minimalne povrsine koji ce obuhvatiti celu konturu
        width, height = size
        xx, yy, w, h = cv2.boundingRect(contour)
        x, y = center
        if w > 3 and w < 60 and h > 6 and h < 90:  # uslov da kontura pripada (trebalo bi preko Krugova)
                contours_Tablica.append(contour)  # ova kontura pripada bar-kodu

    img = Okvir.copy()
    cv2.drawContours(img, contours_Tablica, -1, (0, 0, 255), 4)
    plt.imshow(img)

    return img, contours_Tablica, retSlike

prvaSlika = cv2.imread('images\img-3.png')
SlikaKrugova = cv2.cvtColor(prvaSlika, cv2.COLOR_BGR2RGB) #HLS I HSV # razlike otkritii !!!
plt.imshow(SlikaKrugova)
plt.show()

iscrtana, konturee, VrednostPraga = ZaDetekciju(SlikaKrugova)

upper_red = np.array([255,120,120])
lower_red = np.array([50,0,0])

mask = cv2.inRange(SlikaKrugova, lower_red, upper_red)
output = cv2.bitwise_and(SlikaKrugova, SlikaKrugova, mask=mask)
#output2 = 255 - output  # Ako su kola u mraku NE treba invertovati (TREBA DODATNI USLOV)
plt.imshow(output)
#plt.show()
plt.figure()
plt.imshow(SlikaKrugova)
plt.show()

ret, thresh = cv2.threshold(output, 70, 255, 0) # Ponekad se izgubi neki !!!
plt.imshow(thresh)
plt.show()

iscrtana, contours, retSlike = ZaDetekciju(thresh)
plt.imshow(iscrtana)
plt.show()

print('Broj kontura odnosno krugova: ' + str(len(contours)/2) )