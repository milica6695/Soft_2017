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
    plt.imshow(img_GRAY_gs, 'gray')
    plt.show()
    # image_barcode_bin = cv2.threshold(img_GRAY_gs, 100, 200, cv2.THRESH_BINARY)   #adaptiveThreshold(img_GRAY_gs, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 61,15)

    ####
    retSlike, image_bin = cv2.threshold(img_GRAY_gs, 100, 255, cv2.THRESH_OTSU)
    print("pRAG JE: " + str(retSlike))
    img, contours, hierarchy = cv2.findContours(image_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img = Okvir.copy()
    cv2.drawContours(img, contours, -1, (0, 255, 255), 4)  # Ovde sve konture naznaci
    plt.imshow(img)
    plt.show()

    contours_Tablica = []  # ovde ce biti samo konture koje pripadaju bar-kodu
    for contour in contours:  # za svaku konturu
        center, size, angle = cv2.minAreaRect(
            contour)  # pronadji pravougaonik minimalne povrsine koji ce obuhvatiti celu konturu
        width, height = size
        xx, yy, w, h = cv2.boundingRect(contour)
        x, y = center
        if w > 3 and h > 5 and h < 400:  # uslov da kontura pripada (trebalo bi preko Krugova)
            contours_Tablica.append(contour)  # ova kontura pripada bar-kodu
            print('Detektovano kontura(linija):  ' + str(len(contours_Tablica)))
            print('Kordinate duzi su: ' + str(xx) + ',' + str(yy+h) + '  A druge tacke: ' + str(xx+w) +',' + str(yy)+ '  //Sirina je: ' + str(w) + ' __VISINA JE:  ' + str(h) )
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

# def iseci_broj(broj):
#     _, siva = cv2.threshold(broj, 128, 255, cv2.THRESH_BINARY)
#     _, konture, _ = cv2.findContours(siva, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     #povrsine kontura nadjenih na slici broja
#     povrsine = []
#     for kontura in konture:
#         povrsine.append(cv2.contourArea(kontura))
#     if len(povrsine) == 0:
#         return broj
#     najveca = 0
#     #trazi najvecu
#     for i,povrs in enumerate(povrsine):
#         if povrs > povrsine[najveca]:
#             najveca = i
#     [x, y, w, h] = cv2.boundingRect(konture[najveca])
#     isecena = broj[y:y + h + 1, x:x + w + 1]
#     isecena = cv2.resize(isecena, (28,28), interpolation=cv2.INTER_AREA)
#     return isecena
#
# #######
# def ispraviSlova(img):
#     velicina = 28
#     moments = cv2.moments(img)
#     if abs(moments['mu02']) < 1e-2:
#         return img.copy()
#     nakrivljenost = moments['mu11'] / moments['mu02']
#     M = np.float32([[1, nakrivljenost, -0.5 * velicina * nakrivljenost], [0, 1, 0]])
#     img = cv2.warpAffine(img, M, (velicina, velicina), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
#     return img
# #https://www.learnopencv.com/image-classification-using-convolutional-neural-networks-in-keras/
# def napravi_model(oblik, broj_klasa=10):
#     model = Sequential()
#     model.add(Conv2D(28, (3, 3), padding='same', activation='relu', input_shape=oblik))
#     model.add(Conv2D(28, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#
#     model.add(Conv2D(56, (3, 3), padding='same', activation='relu'))
#     model.add(Conv2D(56, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#
#     model.add(Conv2D(56, (3, 3), padding='same', activation='relu'))
#     model.add(Conv2D(56, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#
#     model.add(Flatten())
#     model.add(Dense(512, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(broj_klasa, activation='softmax'))
#
#     return model
#
#
# if __name__ == '__main__':
#     (tr_slike, tr_labele), (te_slike, te_labele) = mnist.load_data()
#     #
#     broj_klasa = 10
#
#     for i in range(len(te_slike)):
#         isecena = iseci_broj(te_slike[i])
#         te_slike[i] = isecena
#     for i in range(len(tr_slike)):
#         isecena = iseci_broj(tr_slike[i])
#         tr_slike[i] = isecena
#
#     red, kolona = tr_slike.shape[1:]
#     tr_podaci = tr_slike.reshape(tr_slike.shape[0], red, kolona, 1)
#     te_podaci = te_slike.reshape(te_slike.shape[0], red, kolona, 1)
#     oblik = (red, kolona, 1)
#
#     tr_podaci = tr_podaci.astype('float32')
#     te_podaci = te_podaci.astype('float32')
#     # Scale the data to lie between 0 to 1
#     tr_podaci /= 255
#     te_podaci /= 255
#
#     # iz int u kategoricki
#     tr_lab_kat = to_categorical(tr_labele)
#     te_lab_kat = to_categorical(te_labele)
#
#     model = napravi_model(oblik, broj_klasa)
#
#     model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#
#     model.summary()
#
#     istorija = model.fit(tr_podaci, tr_lab_kat, batch_size=256, epochs=0, verbose=1,
#                              validation_data=(te_podaci, te_lab_kat))
#     gubitak, tacnost = model.evaluate(te_podaci, te_lab_kat, verbose=0)
#     model.save_weights('weights.h5')
#     print('Ali Tacnost je: ' +  str(tacnost) )
#
#
#
# def prepoznajBroj(slika, kontura, klasifikator):
#     x, y, w, h = cv2.boundingRect(kontura)
#     centarx = int(x + w / 2)
#     centary = int(y + h / 2)
#     siva = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)
#     broj = siva[centary-12:centary+12, centarx-12:centarx+12]
#     #cv2.imshow("broj", broj)
#     (tr, broj) = cv2.threshold(broj, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#     broj = ispraviSlova(broj)
#     broj = iseci_broj(broj)
#     #cv2.imshow("broj finalno", broj)
#     br = klasifikator.predict_classes(broj.reshape(1, 28, 28, 1))
#     return int(br)
#
# klasifikator = napravi_model((28,28,1),10)
# klasifikator.load_weights(''
#                           'weights.h5')


#####

slika = cv2.imread("images/slika8.jpg")

upper_red = np.array([255, 120, 120])
lower_red = np.array([90, 0, 0])

mask = cv2.inRange(slika, lower_red, upper_red)
output = cv2.bitwise_and(slika, slika, mask=mask)
plt.imshow(output)
plt.show()

x1, y1, x2, y2 = ZaDetekciju(output)
print(str(x1) + ' , ' + str(y1) + '  i  ' + str(x2) + ' , ' + str(y2))

beleKonture = kontureBrojeva(slika)

img_GRAY = cv2.cvtColor(beleKonture, cv2.COLOR_RGB2GRAY)  # konvert u grayscale
retSlike, image_bin = cv2.threshold(img_GRAY, 100, 255, cv2.THRESH_OTSU)
img, contours, hierarchy = cv2.findContours(image_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

contours_Brojeva = []  # ovde ce biti samo konture koje pripadaju bar-kodu
index = 0
suma = 0
sumiranje = False

pts = np.array([[x1,y1],[x2,y2],[x2+90,y2+90],[x1+90,y1+90]], np.int32)
pts = pts.reshape((-1,1,2))
cv2.polylines(beleKonture,[pts],True,(0,255,0))

for contour in contours:  # za svaku konturu
    center, size, angle = cv2.minAreaRect(
        contour)  # pronadji pravougaonik minimalne povrsine koji ce obuhvatiti celu konturu
    width, height = size
    xx, yy, w, h = cv2.boundingRect(contour)
    print('xx='+ str(xx) + '  yy=0' + str(yy))
    #if xx >= x1 and xx <= x2 and yy <= y1 and yy >= y2:  # uslov da kontura pripada (trebalo bi preko Krugova)                                yyy:yyy + hhh, xxx:xxx + www
    #if (x1+100 >= xx >= x1 or x2+100 >= xx >= x2) and (y1+100 >= yy >= y1 or y2+100 >= yy >= y2) :  # uslov da kontura pripada (trebalo bi preko Krugova)                                yyy:yyy + hhh, xxx:xxx + www
    if x1 < xx < x2+90 and y1+90 > yy > y2  :  # uslov da kontura pripada (trebalo bi preko Krugova)                                yyy:yyy + hhh, xxx:xxx + www
        cv2.imwrite('slikeBrojeva/brojevi%d.png' % index, img_GRAY[yy:yy + h, xx:xx + w])
        index = index + 1
       # element = prepoznajBroj(beleKonture, contour, klasifikator)
       # print( str(element) + "  !!!!!!!!!")
        sumiranje = True
        contours_Brojeva.append(contour)  # ova kontura pripada bar-kodu
        print('Detektovano kontura(linija):  ' + str(len(contours_Brojeva)))
    #if sumiranje:
        #suma += element
    # cv2.minAreaRect(contour)
print('Suma identifikovanih brojeva : ' + str(suma) )
    # print('Kordinate duzi su: ' + str(xx) + ',' + str(yy+h) + '  A druge tacke: ' + str(xx+w) +',' + str(yy)+ '  //Sirina je: ' + str(w) + ' __VISINA JE:  ' + str(h) )
img = beleKonture.copy()
cv2.drawContours(img, contours_Brojeva, -1, (255, 0, 0), 2)
plt.imshow(img)
plt.show()


