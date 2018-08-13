import numpy as np
import cv2
from datetime import datetime
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

def diffImg(t0, t1, t2):  # Function to calculate difference between images.
    # t0 = t0[150:410,190:420]
    # t1 = t1[150:410, 190:420]
    # t2 = t2[150:410, 190:420]

    d1 = cv2.absdiff(t2, t1)
    d2 = cv2.absdiff(t1, t0)
    return cv2.bitwise_and(d1, d2)



import os
dir = os.path.dirname(__file__)
filename = os.path.join(dir, 'Videos')

path, dirs, files = os.walk(filename).__next__()
file_count = len(files)


index = 0
threshold = 8500
timeCheck = datetime.now().strftime('%Ss')

while index < len(files):       #Prolazak kroz sve videe iz foldera Videos
    winName = files[index]      # Davanje naziva
    cv2.namedWindow(winName)

    captureVideo = cv2.VideoCapture(path + '/' + files[index])  #'Videos/video-%d.avi' % index)

    trenutniFrame = 0
    brojevi_list = []


    while (captureVideo.isOpened()):
        ret, frame = captureVideo.read()

        t_minus = cv2.cvtColor(captureVideo.read()[1], cv2.COLOR_RGB2GRAY)
        t = cv2.cvtColor(captureVideo.read()[1], cv2.COLOR_RGB2GRAY)
        t_plus = cv2.cvtColor(captureVideo.read()[1], cv2.COLOR_RGB2GRAY)

        if cv2.countNonZero(diffImg(t_minus, t, t_plus)) > threshold :
            dimg = captureVideo.read()[1]
            # cv2.imwrite(datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f') + '.jpg', dimg)
            cv2.imwrite('frames/' + files[index] +'%d.png' % trenutniFrame, t)
            trenutniFrame+=1;

        if (ret == False):
            break

        #img = cv2.imread('dave.jpg')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160, apertureSize=5)

        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

        upper_red = np.array([255, 120, 120])
        lower_red = np.array([90, 0, 0])

        mask = cv2.inRange(frame, lower_red, upper_red)
        output = cv2.bitwise_and(frame, frame, mask=mask)

        DataSlika, konturaLinije, slikaSaonturama = ZaDetekciju(output)
        xx, yy, w, h = cv2.boundingRect(konturaLinije[0])
       # cv2.line(frame, (xx, yy+h ), (xx+w, yy), (255, 255, 0), 4)

        cv2.imwrite('images\slika%d.jpg' % index, frame)

      #  ZaDetekciju(frame)q
        cv2.imshow(winName, frame)
        # Display the resulting frame
        if cv2.waitKey(90) & 0xFF == ord('q'):
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