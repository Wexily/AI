import matplotlib.pyplot as plt
import numpy as np
import cv2
import keras
from datetime import datetime

current_time = datetime.now().strftime("%D-%H:%M:%S")

plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')


def detect_plate(img, text=''):  # the function detects and perfors blurring on the number plate.
    global plate, x, w, y, h
    plate_img = img.copy()
    roi = img.copy()
    plate_rect = plate_cascade.detectMultiScale(plate_img, scaleFactor=1.2, minNeighbors=7)
    for (x, y, w, h) in plate_rect:
        roi_ = roi[y:y + h, x:x + w, :]  # extracting the Region of Interest of license plate for blurring.
        plate = roi[y:y + h, x:x + w, :]
        cv2.rectangle(plate_img, (x + 2, y), (x + w - 3, y + h - 5), (51, 181, 155),
                      3)  # finally representing the detected contours by drawing rectangles around the edges.
    if text != '':
        plate_img = cv2.putText(plate_img, text, (x - w // 2, y - h // 2),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (51, 181, 155), 1, cv2.LINE_AA)

    return plate_img, plate


def display(img_, title=''):
    img = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    ax.imshow(img)
    plt.axis('off')
    plt.title(title)
    plt.show()


img = cv2.imread('DataSet/CAR/6.bmp')
display(img, 'Исходная фотография')

output_img, plate = detect_plate(img)

display(output_img, 'Найденный номерной знак')

display(plate, 'Увеличение выделенной области')


def find_contours(dimensions, img):
    # Find all contours in the image
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]

    # Check largest 5 or  15 contours for license plate or character respectively
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)

    ii = cv2.imread('contour.jpg')

    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs:
        # detects contour in binary image and returns the coordinates of rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        # checking the dimensions of the contour to filter out the characters by contour's size
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
            x_cntr_list.append(intX)

            char_copy = np.zeros((44, 24))
            # extracting each character using the enclosing rectangle's coordinates.
            char = img[intY:intY + intHeight, intX:intX + intWidth]
            char = cv2.resize(char, (20, 40))

            cv2.rectangle(ii, (intX, intY), (intWidth + intX, intY + intHeight), (50, 21, 200), 2)
            plt.imshow(ii, cmap='gray')

            # Make result formatted for classification: invert colors
            char = cv2.subtract(255, char)

            # Resize the image to 24x44 with black border
            char_copy[2:42, 2:22] = char
            char_copy[0:4, :] = 0
            char_copy[:, 0:4] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy)  # List that stores the character's binary image (unsorted)

    # Return characters on ascending order with respect to the x-coordinate (most-left character first)

    plt.show()
    # arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])  # stores character images according to their index
    img_res = np.array(img_res_copy)

    return img_res


def segment_characters(image):
    # Preprocess cropped license plate image
    img_lp = cv2.resize(image, (400, 150))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3, 3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3, 3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Make borders white
    img_binary_lp[0:40, :] = 255
    img_binary_lp[:, 0:30] = 255
    img_binary_lp[130:150, :] = 255
    img_binary_lp[:, 380:400] = 255

    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH / 6,
                  LP_WIDTH / 2,
                  LP_HEIGHT / 10,
                  2 * LP_HEIGHT / 3]
    plt.imshow(img_binary_lp, cmap='gray')
    plt.show()
    cv2.imwrite('contour.jpg', img_binary_lp)

    # Get contours within cropped license plate
    char_list = find_contours(dimensions, img_binary_lp)

    return char_list


char = segment_characters(plate)

for i in range(len(char)):
    plt.subplot(1, 10, i + 1)
    plt.imshow(char[i], cmap='gray')
    plt.axis('off')

plt.show()


# Predicting the output
def fix_dimension(img):
    new_img = np.zeros((28, 28, 3))
    for i in range(3):
        new_img[:, :, i] = img
    return new_img


model = keras.models.load_model('model/Sequential.h5')


def show_results():
    dic = {}
    characters = '0123456789ABCEHKMOPTXY'
    for i, c in enumerate(characters):
        dic[i] = c

    output = []
    for i, ch in enumerate(char):  # iterating over the characters
        img_ = cv2.resize(ch, (28, 28), interpolation=cv2.INTER_AREA)
        img = fix_dimension(img_)
        img = img.reshape(1, 28, 28, 3)  # preparing image for the model
        y_ = model.predict(img)[0]  # predicting the class
        predicted_index = np.argmax(y_)  # get the index of the predicted class
        character = dic[int(predicted_index)]  # convert predicted index to int and get character from the dictionary
        output.append(character)  # storing the result in a list

    plate_number = ''.join(output)

    return plate_number


print(f'Распознанный номер авто: {show_results()}')
print(f'Время фиксации: {current_time}')
