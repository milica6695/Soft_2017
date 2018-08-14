import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.datasets import mnist


def ZaDetekciju(Okvir):
    img_GRAY_gs = cv2.cvtColor(Okvir, cv2.COLOR_RGB2GRAY)  # konvert u grayscale
    # image_barcode_bin = cv2.threshold(img_GRAY_gs, 100, 200, cv2.THRESH_BINARY)   #adaptiveThreshold(img_GRAY_gs, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 61,15)
    retSlike, image_bin = cv2.threshold(img_GRAY_gs, 100, 255, cv2.THRESH_OTSU)
    print("pRAG JE: " + str(retSlike))
    img, contours, hierarchy = cv2.findContours(image_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img = Okvir.copy()
    cv2.drawContours(img, contours, -1, (0, 255, 255), 4)  # Ovde sve konture naznaci
    # plt.imshow(img)
    # plt.show()

    contours_Tablica = []  # ovde ce biti samo konture koje pripadaju bar-kodu
    for contour in contours:  # za svaku konturu
        center, size, angle = cv2.minAreaRect(
            contour)  # proe
        xx, yy, w, h = cv2.boundingRect(contour)
        x, y = center
        if w > 3 and h > 5 and h < 400:  # uslov da kontura pripada (trebalo bi preko Krugova)
            contours_Tablica.append(contour)  # ova kontura pripada bar-kodu
            #print('Detektovano kontura(linija):  ' + str(len(contours_Tablica)))
            #print('Kordinate duzi su: ' + str(xx) + ',' + str(yy+h) + '  A druge tacke: ' + str(xx+w) +',' + str(yy)+ '  //Sirina je: ' + str(w) + ' __VISINA JE:  ' + str(h) )
            return xx, yy + h, xx + w, yy;
    # img = Okvir.copy()
    # cv2.drawContours(img, contours_Tablica, -1, (0, 255, 0), 1)
    # plt.imshow(img)
    # plt.show()
    #
    # return img, contours_Tablica, retSlike


def kontureBrojeva(slika):
    upper_red = np.array([255, 255, 255])
    lower_red = np.array([140, 140, 140])  # Posto su brojevi bjeli

    mask = cv2.inRange(slika, lower_red, upper_red)
    output = cv2.bitwise_and(slika, slika, mask=mask)
    plt.imshow(output)
    plt.show()
    return output

def iseci_broj(broj):
    _, siva = cv2.threshold(broj, 128, 255, cv2.THRESH_BINARY)
    _, konture, _ = cv2.findContours(siva, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #povrsine kontura nadjenih na slici broja
    povrsine = []
    for kontura in konture:
        povrsine.append(cv2.contourArea(kontura))
    if len(povrsine) == 0:
        return broj
    najveca = 0
    #trazi najvecu
    for i,povrs in enumerate(povrsine):
        if povrs > povrsine[najveca]:
            najveca = i
    [x, y, w, h] = cv2.boundingRect(konture[najveca])
    isecena = broj[y:y + h + 1, x:x + w + 1]
    isecena = cv2.resize(isecena, (28,28), interpolation=cv2.INTER_AREA)
    return isecena

#######
def ispraviSlova(img):
    velicina = 28
    moments = cv2.moments(img)
    if abs(moments['mu02']) < 1e-2:
        return img.copy()
    nakrivljenost = moments['mu11'] / moments['mu02']
    M = np.float32([[1, nakrivljenost, -0.5 * velicina * nakrivljenost], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (velicina, velicina), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img
#https://www.learnopencv.com/image-classification-using-convolutional-neural-networks-in-keras/

def napravi_model(oblik, broj_klasa=10):
    model = Sequential()
    model.add(Conv2D(28, (3, 3), padding='same', activation='relu', input_shape=oblik))
    model.add(Conv2D(28, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(56, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(56, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(56, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(56, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(broj_klasa, activation='softmax'))

    return model

# IF NAME IS MAIN (ZA trenjiranje Neuronske mreze !!!)

def prepoznajBroj(slika, kontura, klasifikator):
    x, y, w, h = cv2.boundingRect(kontura)
    centarx = int(x + w / 2)
    centary = int(y + h / 2)
    siva = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)
    broj = siva[centary-12:centary+12, centarx-12:centarx+12]
    cv2.imshow("broj", broj)
    (tr, broj) = cv2.threshold(broj, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    broj = ispraviSlova(broj)
   # broj = iseci_broj(broj)
    cv2.imshow("broj finalno", broj)
    cv2.imwrite('slikeBrojeva/brojevi%d.png' % index, broj)
    br = klasifikator.predict_classes(broj.reshape(1, 28, 28, 1))
    return int(br)

def diffImg(t0, t1, t2):  # Function to calculate difference between images.
    # t0 = t0[150:410,190:420]
    # t1 = t1[150:410, 190:420]
    # t2 = t2[150:410, 190:420]
    d1 = cv2.absdiff(t2, t1)
    d2 = cv2.absdiff(t1, t0)
    return cv2.bitwise_and(d1, d2)


klasifikator = napravi_model((28,28,1),10)
klasifikator.load_weights(''
                          'weights.h5')
#####

slika = cv2.imread("frames/video-0.avi0.png")     # 8

upper_red = np.array([255, 100, 100])
lower_red = np.array([90, 0, 0])

mask = cv2.inRange(slika, lower_red, upper_red)
output = cv2.bitwise_and(slika, slika, mask=mask)
plt.imshow(output)
plt.show()

x1, y1, x2, y2 = ZaDetekciju(output)
#print(str(x1) + ' , ' + str(y1) + '  i  ' + str(x2) + ' , ' + str(y2))

beleKonture = kontureBrojeva(slika)

img_GRAY = cv2.cvtColor(beleKonture, cv2.COLOR_RGB2GRAY)  # konvert u grayscale
retSlike, image_bin = cv2.threshold(img_GRAY, 100, 255, cv2.THRESH_OTSU)
blurovana = cv2.GaussianBlur(image_bin, (5, 5), 0)  # ili Gray ovde

img, contours, hierarchy = cv2.findContours(blurovana.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


contours_Brojeva = []  # ovde ce biti samo konture koje pripadaju bar-kodu
index = 0
suma = 0
sumiranje = False
for contour in contours:  # za svaku konturu
    center, size, angle = cv2.minAreaRect(contour)  # pronadji pravougaonik minimalne povrsine koji ce obuhvatiti celu konturu
    width, height = size
    xx, yy, w, h = cv2.boundingRect(contour)
    povrsina = cv2.contourArea(contour)
    #print('xx='+ str(xx) + '  yy=0' + str(yy))
    #if xx >= x1 and xx <= x2 and yy <= y1 and yy >= y2:  # uslov da kontura pripada (trebalo bi preko Krugova)                                yyy:yyy + hhh, xxx:xxx + www
    #if (x1+100 >= xx >= x1 or x2+100 >= xx >= x2) and (y1+100 >= yy >= y1 or y2+100 >= yy >= y2) :  # uslov da kontura pripada (trebalo bi preko Krugova)                                yyy:yyy + hhh, xxx:xxx + www
    if povrsina > 12 and xx > x1 and width >= 2 and height > 7:  # uslov da kontura pripada (trebalo bi preko Krugova)                                yyy:yyy + hhh, xxx:xxx + www
        #cv2.imwrite('slikeBrojeva/brojevi%d.png' % index, img_GRAY[yy:yy + h, xx:xx + w])
        print('Visina =' + str(height) + ' Sirina je: ' + str(width) )
        index = index + 1
        element = prepoznajBroj(beleKonture, contour, klasifikator)
        print( str(element) + "  !!!!!!!!!")
        sumiranje = True
        contours_Brojeva.append(contour)  # ova kontura pripada bar-kodu
        print('Detektovano kontura(linija):  ' + str(len(contours_Brojeva)))
        if sumiranje:
            suma = suma +  element
            sumiranje = False
    # cv2.minAreaRect(contour)
print('Suma identifikovanih brojeva : ' + str(suma) )
    # print('Kordinate duzi su: ' + str(xx) + ',' + str(yy+h) + '  A druge tacke: ' + str(xx+w) +',' + str(yy)+ '  //Sirina je: ' + str(w) + ' __VISINA JE:  ' + str(h) )
img = beleKonture.copy()
cv2.drawContours(img, contours_Brojeva, -1, (255, 0, 0), 2)
plt.imshow(img)
plt.show()