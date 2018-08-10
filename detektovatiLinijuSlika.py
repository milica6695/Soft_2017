import numpy as np
import cv2
import matplotlib.pyplot as plt


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
                print('Ima li:  ' + str(len(contours_Tablica)))
                print(str(size))
    img = Okvir.copy()
    cv2.drawContours(img, contours_Tablica, -1, (0, 255, 0), 2)
    plt.imshow(img)
    plt.show()

    return img, contours_Tablica, retSlike


slika = cv2.imread("images/slika1.jpg")

upper_red = np.array([255,120,120])
lower_red = np.array([50,0,0])

mask = cv2.inRange(slika, lower_red, upper_red)
output = cv2.bitwise_and(slika, slika, mask=mask)
plt.imshow(output)
plt.show()

DataSlika, sveKonture, slikaSaonturama = ZaDetekciju(output)


