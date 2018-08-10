import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from skimage.measure import label



def ZaDetekciju(Okvir):
    img_GRAY_gs = cv2.cvtColor(Okvir, cv2.COLOR_RGB2GRAY)  # konvert u grayscale
    retSlike, image_bin = cv2.threshold(img_GRAY_gs, 100, 255, cv2.THRESH_OTSU)
    img, contours, hierarchy = cv2.findContours(image_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contours_Tablica = []  # ovde ce biti samo konture koje pripadaju bar-kodu
    for contour in contours:  # za svaku konturu
        center, size, angle = cv2.minAreaRect(
            contour)  # pronadji pravougaonik minimalne povrsine koji ce obuhvatiti celu konturu
        width, height = size
        xx, yy, w, h = cv2.boundingRect(contour)
        x, y = center
        if w > 3 and h > 5 and h < 400:  # uslov da kontura pripada (trebalo bi preko Krugova)
                contours_Tablica.append(contour)  # ova kontura pripada bar-kodu
    img = Okvir.copy()
    cv2.drawContours(img, contours_Tablica, -1, (0, 255, 0), 2)
    #plt.imshow(img)
    #plt.show()

    return img, contours_Tablica, retSlike

import os
dir = os.path.dirname(__file__)
filename = os.path.join(dir, 'Videos')

path, dirs, files = os.walk(filename).__next__()
file_count = len(files)


index = 0
while index < len(files):       #Prolazak kroz sve videe iz foldera Videos
    winName = files[index]      # Davanje naziva
    cv2.namedWindow(winName)

    captureVideo = cv2.VideoCapture(path + '/' + files[index])  #'Videos/video-%d.avi' % index)
    trenutniFrame = 0
    brojevi_list = []
    while (captureVideo.isOpened()):
        ret, frame = captureVideo.read()

        if (ret == False):
            break
        ####
        # siva = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('frame', siva)
        # if (trenutniFrame == 0):
        #     edges = cv2.Canny(siva, 50, 150, apertureSize=3)
        #     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
        #     x1 = lines[0][0][0]
        #     y1 = lines[0][0][1]
        #     x2 = lines[0][0][2]
        #     y2 = lines[0][0][3]
        # img = cv2.inRange(siva, 163, 255)
        # clos = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((4, 4), np.uint8))
        # labela = label(clos)
        # regioni = regionprops(labela)
        # for reg in regioni:
        #     if ((reg.bbox[2] - reg.bbox[0]) <= 10 or prolazIzaPrave(reg.bbox) == False):
        #         continue
        #     img_br = nasaoBroj(reg.bbox, siva)
        #    # prepoznat_br = int(ktrain.predict(img_br.reshape(1, 784)))
        #     kreiranjeListeBr(reg.bbox, brojevi_list, 4)
        # trenutniFrame += 1

        ####

        #img = cv2.imread('dave.jpg')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160, apertureSize=5)

        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            # x1 = int(x0 + 10 * (-b))
            # y1 = int(y0 + 10 * (a))
            # x2 = int(x0 - 300 * (-b))
            # y2 = int(y0 - 300 * (a))
       # saKonturama, konture, slika = ZaDetekciju(frame)
        upper_red = np.array([255, 120, 120])
        lower_red = np.array([90, 0, 0])

        mask = cv2.inRange(frame, lower_red, upper_red)
        output = cv2.bitwise_and(frame, frame, mask=mask)

        DataSlika, konturaLinije, slikaSaonturama = ZaDetekciju(output)
        xx, yy, w, h = cv2.boundingRect(konturaLinije[0])
        cv2.line(frame, (xx, yy+h ), (xx+w, yy), (255, 0, 0), 4)

        cv2.imwrite('images\slika%d.jpg' % index, frame)

      #  ZaDetekciju(frame)
        cv2.imshow(winName, frame)
        # Display the resulting frame
        if cv2.waitKey(10) & 0xFF == ord('q'):
            #captureVideo.release()
            break
    suma = 0
    for broj in brojevi_list:
        suma += broj[0]

    print ('Za video ' + (winName) + ' suma brojeva iznosi: ' + str(suma) + '\n')
    cv2.destroyWindow(winName)  # Kad izadje iz whil petlje unistava se
    index +=1
    #break   # proveravamo samo za jedan video

# When everything done, release the capture
captureVideo.release()