import types

import numpy
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Flatten, Dense
import sys
import cv2
import re
import numpy as np
import matplotlib.pyplot as plt
import glob

import cv2
import re
import numpy as np
import matplotlib.pyplot as plt
import glob

img_width, img_height = 28, 28


def create_model():
    model = Sequential()

    model.add(Convolution2D(16, 5, 5, activation='relu', input_shape=(img_width, img_height, 3)))
    model.add(MaxPooling2D(2, 2))

    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(2, 2))

    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))

    model.add(Dense(36, activation='softmax'))

    model.summary()

    return model


# model predict number if true otherwise character
def prediction(model, img, number):
    # img = cv2.imread(sys.argv[1])
    img = cv2.resize(img, (img_width, img_height))
    model = create_model()
    model.load_weights('./weights.h5')
    arr = numpy.array(img).reshape((img_width, img_height, 3))
    arr = numpy.expand_dims(arr, axis=0)
    prediction = model.predict(arr)[0]
    bestclass = ''
    bestconf = -1
    for n in range(10):
        if (prediction[n] > bestconf):
            bestclass = int(n)
            bestconf = prediction[n]
    return str(bestclass)

img_width, img_height = 28, 28


def dilate(image):
    kernel = np.ones((3, 3))
    return cv2.dilate(image, kernel, iterations=1)


def erode(image):
    kernel = np.ones((4, 4))
    return cv2.erode(image, kernel, iterations=1)


def vertical_erode(image):
    cols = image.shape[1]
    vertical_size = cols / 60
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(vertical_size)))
    return cv2.erode(image, verticalStructure)


def resize_region(region):
    return cv2.resize(region, (28, 28,), interpolation=cv2.INTER_NEAREST)


def select_roi(image_color, image_bin):
    img, contours, hierarchy = cv2.findContours(plate_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if w > 15 and w < 90 and h > 30 and h < 100 and area > 600:
         region = image_bin[y:y + h + 1, x:x + w + 1]
         regions_array.append([resize_region(region), (x, y, w, h)])
         cv2.rectangle(image_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
        regions_array = sorted(regions_array, key=lambda item: item[1][0])
        sorted_regions = sorted_regions = [region[0] for region in regions_array]
    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = sorted_regions = [region[0] for region in regions_array]
    return image_color, sorted_regions


def prepare_for_nn(regions):
    ready_for_nn = []
    for region in regions:
        scale = region / 255
        ready_for_nn.append(region.flatten())
    return ready_for_nn

# ucitavanje tekstualnih falova
file_object = open('out.txt', 'r+')

allPlates = glob.glob("slikeBrojeva/brjevi2.png")
print(allPlates)
# for i, plate in enumerate(allPlates):
plate_color = cv2.imread("slikeBrojeva/brjevi1.png")

#plate_color = cv2.resize(plate_color, (450, 105), interpolation=cv2.INTER_NEAREST)
# cv2.imshow('color',plate_color)
plate_color = cv2.GaussianBlur(plate_color, (3, 3), 0)
plate_gray = cv2.cvtColor(plate_color, cv2.COLOR_RGB2GRAY)
cv2.imshow('gray', plate_gray)
plate_bin = cv2.adaptiveThreshold(plate_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 61, 30)
cv2.imshow('binary', plate_bin)

# izdvajanje regiona od interesa sa tablice
selected_regions, chars = select_roi(plate_color, plate_bin)
cv2.imshow('regions', selected_regions)
typess = []
result = []
for i, char in enumerate(chars):
        imgs = cv2.cvtColor(char, cv2.COLOR_GRAY2BGR)
        model = create_model()
        result.append(prediction(model, imgs, typess[i]))

licence_plate = ''.join(result) + '\n'
print('Pre validacije ' + licence_plate)
licence_plates = file_object.readlines()
file_object.write(licence_plate)

file_object.close()
plt.show()
cv2.waitKey()