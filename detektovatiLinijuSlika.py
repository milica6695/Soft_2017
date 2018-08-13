import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage

from skimage import img_as_ubyte
from skimage.measure import label
from skimage.morphology import skeletonize
from skimage.measure import regionprops

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_mldata


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
    cv2.drawContours(img, contours, -1, (0, 255, 255), 4)   #Ovde sve konture naznaci
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
                #print('Kordinate duzi su: ' + str(xx) + ',' + str(yy+h) + '  A druge tacke: ' + str(xx+w) +',' + str(yy)+ '  //Sirina je: ' + str(w) + ' __VISINA JE:  ' + str(h) )
                return xx,yy+h,xx+w,yy;
    # img = Okvir.copy()
    # cv2.drawContours(img, contours_Tablica, -1, (0, 255, 0), 1)
    # plt.imshow(img)
    # plt.show()
    #
    # return img, contours_Tablica, retSlike

def kontureBrojeva(slika):
    upper_red = np.array([255, 255, 255])
    lower_red = np.array([140, 140, 140])   #Posto su brojevi bjeli

    mask = cv2.inRange(slika, lower_red, upper_red)
    output = cv2.bitwise_and(slika, slika, mask=mask)
    plt.imshow(output)
    plt.show()
    return output

# mnistPutanja = "C:\\Users\\Smekac\\Desktop\\IDX3-UBYTE File"
#
# mnist = fetch_mldata('MNIST original')
# ucitanFile = mnist.data
# np.save(os.path.join(mnistPutanja, 'dataSet'), ucitanFile)

from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels

#file = os.path.join('C:/Users/Smekac/Desktop/dataSet', 'dataSet')

with open('C:/Users/Smekac/Desktop/train-images-idx3-ubyte.gz', 'rb') as f:
  train_images = extract_images(f)
  #obukaDataseta(train_images)
 # np.save(os.path.join('C:/Users/Smekac/Desktop/dataSet', train_images), 'dataSet')


#kneCLas = KNeighborsClassifier(n_neighbors=1, algorithm='brute')
#ktrain = kneCLas.fit('dataSet', train_images);

slika = cv2.imread("images/slika8.jpg")

upper_red = np.array([255,120,120])
lower_red = np.array([50,0,0])

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
index =0
for contour in contours:  # za svaku konturu
    center, size, angle = cv2.minAreaRect(contour)  # pronadji pravougaonik minimalne povrsine koji ce obuhvatiti celu konturu
    width, height = size
    xx, yy, w, h = cv2.boundingRect(contour)
    if xx >= x1 and xx <= x2 and yy <= y1 and yy >= y2 :  # uslov da kontura pripada (trebalo bi preko Krugova)
        # rotated = ndimage.rotate(slika, angle, reshape=True)
        # RotiranaBezBoja = ndimage.rotate(slika, angle, reshape=True)
        # imgRotirana, KonturaKadjeSlikaRotirana, retSlike = kontureBrojeva(rotated)
        # xxx, yyy, www, hhh = cv2.boundingRect(KonturaKadjeSlikaRotirana[index])
        # center, size, angle = cv2.minAreaRect(KonturaKadjeSlikaRotirana[index])
        # plt.imshow(imgRotirana[yyy:yyy + hhh, xxx:xxx + www])  # SlikaKola[x:x+w,y:y+h])
        # plt.figure()
        # cv2.imwrite('images/fotka%d.png' % index, RotiranaBezBoja[yyy:yyy + hhh, xxx:xxx + www])
        # plt.imshow(RotiranaBezBoja[yyy:yyy + hhh, xxx:xxx + www])  # SlikaKola[x:x+w,y:y+h])
        # index = index + 1
        # plt.show()                                    yyy:yyy + hhh, xxx:xxx + www
        cv2.imwrite('slikeBrojeva/brjevi%d.png' % index, img_GRAY[yy:yy+h, xx:xx + w])
        
        index = index + 1
        contours_Brojeva.append(contour)  # ova kontura pripada bar-kodu
        print('Detektovano kontura(linija):  ' + str(len(contours_Brojeva)))
     #cv2.minAreaRect(contour)

                #print('Kordinate duzi su: ' + str(xx) + ',' + str(yy+h) + '  A druge tacke: ' + str(xx+w) +',' + str(yy)+ '  //Sirina je: ' + str(w) + ' __VISINA JE:  ' + str(h) )
img = beleKonture.copy()
cv2.drawContours(img, contours_Brojeva, -1, (255, 0, 0), 2)
plt.imshow(img)
plt.show()




